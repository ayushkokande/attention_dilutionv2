"""Identify guardrail attention heads via Direct Logit Attribution (DLA)
onto d_hat at the canonical read-out layer L_read.

For each (layer ell <= L_read, head h):
    DLA(ell, h) = E_p [ < (W_O[h] @ z_h(p))[-1] , d_hat[L_read] > ]

where z_h is the per-head output of the attention block (the slice of the
o_proj input corresponding to head h), and the expectation is over a held-out
pool of harmful prompts. We slice the pre-output-projection input tensor by
head and project each slice onto the residual via the matching W_O[h] block,
then dot with d_hat at the canonical layer.

Top-K heads by DLA score define the guardrail set used by experiment_10
(attention-mass on guardrail heads vs N) and experiment_11 (steering target).

Pool. By default we reuse the held-out pool that defined L_read in
experiment_2 (AdvBench[480:480+N), N=24), so the heads are scored on the
exact prompts that demonstrated post-ablation refusal=0 at L_read. Override
via --start-idx / --n-prompts.

Reads:
    results/<slug>/refusal_direction/d_hat_all_layers.pt
    results/<slug>/refusal_direction/meta.json   (canonical_layer auto-detect)

Writes:
    results/<slug>/guardrail_heads/
        head_attrib.pt        # full [L_read+1, n_heads] tensor (float32, cpu)
        dla_ranked.csv        # all (layer, head, dla_score) sorted desc
        guardrail_heads.json  # top-K plus pool + read-layer metadata
        meta.json
        fig_head_attrib_heatmap.png  (optional, requires matplotlib)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP1 = ROOT / "experiment_1"
for _p in (str(HERE), str(EXP1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # type: ignore  (reused from experiment_1/utils.py)
    format_chat_prompt,
    load_harmful_prompts,
    results_dir,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def slug_from_model(m: str) -> str:
    return m.split("/")[-1].lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--refusal-dir", default=None,
                   help="Override directory containing d_hat_all_layers.pt + meta.json. "
                        "Default: results/<slug>/refusal_direction")
    p.add_argument("--read-layer", type=int, default=None,
                   help="Layer index whose d_hat is used as the read-out direction. "
                        "Default: meta.json:canonical_layer (Exp.2 picked 18 for Qwen3-14B).")
    p.add_argument("--n-prompts", type=int, default=24,
                   help="Held-out harmful pool size for DLA scoring.")
    p.add_argument("--start-idx", type=int, default=480,
                   help="AdvBench start index for the held-out pool. Default 480 reuses "
                        "the same pool that defined L_read in Exp.2 causal sweep.")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--output-dir", default=None,
                   help="Override; default: results/<slug>/guardrail_heads")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the heatmap PNG (avoids matplotlib import).")
    return p.parse_args()


def _resolve_read_layer(refusal_dir: Path, override: int | None) -> int:
    if override is not None:
        return int(override)
    meta_path = refusal_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}; either pass --read-layer N explicitly "
            f"or run experiment_2/refusal_direction.py first."
        )
    meta = json.loads(meta_path.read_text())
    layer = meta.get("canonical_layer", meta.get("default_layer"))
    if layer is None:
        raise KeyError(f"{meta_path} has neither 'canonical_layer' nor 'default_layer'.")
    return int(layer)


def _load_d_hat(refusal_dir: Path, read_layer: int) -> torch.Tensor:
    pt_path = refusal_dir / "d_hat_all_layers.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Missing {pt_path}; run experiment_2/refusal_direction.py first."
        )
    bank = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(bank, torch.Tensor):
        raise TypeError(f"Expected a tensor in {pt_path}, got {type(bank)}.")
    if read_layer < 0 or read_layer >= bank.shape[0]:
        raise IndexError(f"read_layer={read_layer} out of range for {tuple(bank.shape)}.")
    d = bank[read_layer].float()
    return d / d.norm().clamp_min(1e-8)


@torch.no_grad()
def compute_dla(
    model,
    tokenizer,
    prompts: list[str],
    d_hat_read: torch.Tensor,
    read_layer: int,
    batch_size: int,
    device,
) -> torch.Tensor:
    """Return [read_layer + 1, n_heads] DLA matrix on cpu/float32.

    Per-prompt score for head (ell, h) = < W_O[ell, h] @ z_{h, last}, d_hat >
    where z_{h, last} is the slice of the o_proj input at the last real token
    that corresponds to head h. The expectation is computed as a sum across
    prompts and divided at the end (averaging weighting each prompt equally).
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_head = d_model // n_heads
    layers = model.model.layers

    # Pre-slice o_proj weights per head (kept on each layer's device, float32).
    # Linear weight shape is [out_features, in_features] = [d_model, d_model].
    # Reshape to [d_model, n_heads, d_head] so column-block h corresponds to
    # head h's slice of the concat-z input.
    w_per_head: dict[int, torch.Tensor] = {}
    for ell in range(read_layer + 1):
        w = layers[ell].self_attn.o_proj.weight  # [d_model, d_model]
        w_per_head[ell] = (
            w.detach().float().view(d_model, n_heads, d_head).contiguous()
        )

    # Accumulator on cpu / float32.
    totals = torch.zeros(read_layer + 1, n_heads, dtype=torch.float32)
    n_seen = 0

    captured: dict[int, torch.Tensor] = {}

    def make_pre_hook(ell: int):
        def hook(module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            # Detach + cast to float32 immediately to keep dot-product precision.
            captured[ell] = x.detach().float()
        return hook

    handles = []
    for ell in range(read_layer + 1):
        h = layers[ell].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(ell))
        handles.append(h)

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            ).to(device)
            captured.clear()
            model(**enc, use_cache=False)

            # Last real token index per row (left-padding => last column is real).
            # We rely on tokenizer.padding_side="left" being set by the caller.
            for ell in range(read_layer + 1):
                x = captured[ell]                           # [B, T, d_model]
                z_last = x[:, -1, :].view(x.shape[0], n_heads, d_head)  # [B, H, d_head]
                w_h = w_per_head[ell].to(z_last.device)     # [d_model, H, d_head]
                # per-head residual contribution: y_{B,H,d_model}
                y = torch.einsum("bhd,mhd->bhm", z_last, w_h)
                # dot with d_hat (kept on same device as y for the dot product).
                d = d_hat_read.to(y.device)
                scores = (y @ d).sum(dim=0)                 # [n_heads]
                totals[ell] += scores.float().cpu()

            n_seen += enc["input_ids"].shape[0]
            print(f"  forward {n_seen}/{len(prompts)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    return totals / max(n_seen, 1)


def _save_heatmap(head_attrib: torch.Tensor, out_path: Path, slug: str, read_layer: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[fig] matplotlib unavailable ({e}); skipping heatmap.", flush=True)
        return
    arr = head_attrib.numpy()
    vmax = max(abs(arr.min()), abs(arr.max()))
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"DLA onto d_hat ({slug}, read L{read_layer})")
    fig.colorbar(im, ax=ax, label="DLA score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"      {out_path}")


def main() -> None:
    args = parse_args()
    slug = slug_from_model(args.model)
    refusal_dir = (
        Path(args.refusal_dir) if args.refusal_dir
        else (results_dir(slug) / "refusal_direction")
    )
    out_dir = Path(args.output_dir) if args.output_dir else (results_dir(slug) / "guardrail_heads")
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]

    read_layer = _resolve_read_layer(refusal_dir, args.read_layer)

    print(f"Model       : {args.model}")
    print(f"Slug        : {slug}")
    print(f"Refusal dir : {refusal_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Read layer  : {read_layer}")
    print(f"Pool        : AdvBench[{args.start_idx}:{args.start_idx + args.n_prompts}]")
    print(f"GPUs        : {torch.cuda.device_count()}")
    print(f"dtype       : {args.dtype}  | batch_size={args.batch_size}")

    d_hat_read = _load_d_hat(refusal_dir, read_layer)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # Held-out harmful pool. Pull enough rows then slice.
    all_h = load_harmful_prompts(args.start_idx + args.n_prompts)
    pool = all_h[args.start_idx:args.start_idx + args.n_prompts]
    if len(pool) < args.n_prompts:
        raise RuntimeError(
            f"AdvBench yielded only {len(pool)} rows after slicing; "
            f"requested {args.n_prompts} starting at {args.start_idx}."
        )
    pool_p = [format_chat_prompt(tok, p, enable_thinking=args.enable_thinking) for p in pool]

    print("\n[DLA forward]")
    t0 = time.time()
    head_attrib = compute_dla(model, tok, pool_p, d_hat_read, read_layer,
                              args.batch_size, device)
    print(f"  done {time.time() - t0:.1f}s | shape {tuple(head_attrib.shape)}")

    torch.save(head_attrib, out_dir / "head_attrib.pt")

    # Rank.
    n_layers_scored, n_heads = head_attrib.shape
    flat = head_attrib.flatten()
    order = torch.argsort(flat, descending=True)
    ranked = []
    for k, i in enumerate(order.tolist()):
        ell = int(i // n_heads)
        h = int(i % n_heads)
        ranked.append({"rank": k, "layer": ell, "head": h, "dla_score": float(flat[i].item())})

    csv_path = out_dir / "dla_ranked.csv"
    with csv_path.open("w") as f:
        f.write("rank,layer,head,dla_score\n")
        for r in ranked:
            f.write(f"{r['rank']},{r['layer']},{r['head']},{r['dla_score']}\n")

    top_k = max(1, min(args.top_k, len(ranked)))
    top = ranked[:top_k]
    cum = 0.0
    total_pos = sum(r["dla_score"] for r in ranked if r["dla_score"] > 0) or 1.0
    for r in top:
        cum += max(r["dla_score"], 0.0)
        r["cum_pos_mass_frac"] = cum / total_pos

    print(f"\nTop-{top_k} guardrail heads (read L{read_layer}):")
    for r in top:
        print(f"  L{r['layer']:02d}H{r['head']:02d}  dla={r['dla_score']:+.4f}  "
              f"cum_pos_mass={r['cum_pos_mass_frac']:.2f}")

    guard_payload = {
        "model": args.model,
        "read_layer": read_layer,
        "n_layers_scored": n_layers_scored,
        "n_heads": n_heads,
        "n_prompts": len(pool),
        "advbench_pool_range": [args.start_idx, args.start_idx + args.n_prompts],
        "top_k": top_k,
        "guardrail_heads": [
            {"layer": r["layer"], "head": r["head"], "dla_score": r["dla_score"],
             "rank": r["rank"], "cum_pos_mass_frac": r["cum_pos_mass_frac"]}
            for r in top
        ],
    }
    (out_dir / "guardrail_heads.json").write_text(json.dumps(guard_payload, indent=2))

    meta = {
        "model": args.model,
        "slug": slug,
        "read_layer": read_layer,
        "refusal_dir": str(refusal_dir),
        "n_prompts": len(pool),
        "advbench_start_idx": args.start_idx,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "top_k": top_k,
        "head_attrib_shape": list(head_attrib.shape),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if not args.no_figure:
        _save_heatmap(head_attrib, out_dir / "fig_head_attrib_heatmap.png", slug, read_layer)

    print(f"\nSaved {out_dir / 'head_attrib.pt'}")
    print(f"      {csv_path}")
    print(f"      {out_dir / 'guardrail_heads.json'}")
    print(f"      {out_dir / 'meta.json'}")
    print(f"\nNext: experiment_10 reads guardrail_heads.json for the head set.")


if __name__ == "__main__":
    main()
