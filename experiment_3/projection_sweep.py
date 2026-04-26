"""Activation-projection sweep — the proposal's promised Phase 2 metric.

Where context_sweep.py measures *binary* refusal output, this script measures
the *continuous internal signal* that the refusal output is a thresholded
function of: the projection of the last-token residual stream onto the
refusal direction d̂ at each chosen layer.

For each context length L and each prompt:

    proj(L, layer) = residual_at_last_token[layer] · d̂[layer]

Aggregated over the same disjoint AdvBench split context_sweep.py uses (rows
[0, harmful_n)). The ablated arm is omitted by design: with d̂ projected out
at every block, projection at the harmful token is ≈ 0 and uninformative.

Outputs
-------
results/<slug>/projection_sweep.json   raw values + per-cell summary stats
results/<slug>/projection_sweep.png    mean projection vs L, one line per layer

A monotone-decreasing curve at any layer is direct evidence of dilution in
activation space, even when the binary refusal rate is saturated.
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
EXP2 = ROOT / "experiment_2"
for _p in (str(HERE), str(EXP1), str(EXP2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # type: ignore
    format_chat_prompt,
    load_harmful_prompts,
    results_dir,
)
from context_sweep import (  # type: ignore
    SWEEP_HARMFUL_N,
    adaptive_batch_size,
    build_filler,
    slug_from_model,
    wrap_prompt,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

DEFAULT_LENGTHS = [0, 512, 2048, 8192, 16384, 32768]
DEFAULT_LAYERS = [20, 24, 28]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--refusal-dir", required=True,
                   help="Dir with d_hat_all_layers.pt + meta.json from refusal_direction.py")
    p.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="Layer indices into d_hat_all_layers.pt to project against.")
    p.add_argument("--harmful-n", type=int, default=SWEEP_HARMFUL_N)
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Optional cap for quick smoke runs.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override adaptive batch size (applied to all L).")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None,
                   help="Override; default: results/<slug>")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect_projections(
    model,
    tok,
    chat_prompts: list[str],
    layers: list[int],
    d_hat_subset: torch.Tensor,  # (n_chosen_layers, d_model) on CPU/float32
    batch_size: int,
    device,
) -> torch.Tensor:
    """Return (n_chosen_layers, n_prompts) tensor of scalar projections."""
    n_chosen = len(layers)
    n_prompts = len(chat_prompts)
    out = torch.zeros(n_chosen, n_prompts, dtype=torch.float32)

    block_to_slot = {layer_idx: slot for slot, layer_idx in enumerate(layers)}
    cur_offset = [0]  # mutable container so the hook closure can read it

    def make_hook(slot: int):
        d = d_hat_subset[slot]  # (d_model,) cpu/float32

        def hook(module, inp, mod_out):
            h = mod_out[0] if isinstance(mod_out, tuple) else mod_out
            # tokenizer.padding_side == "left" → position -1 is the actual last
            # real token for every row in the batch.
            last = h[:, -1, :].detach().to("cpu", torch.float32)  # (B, d_model)
            proj = last @ d  # (B,)
            start = cur_offset[0]
            out[slot, start:start + last.shape[0]] = proj

        return hook

    handles = []
    for layer_idx, slot in block_to_slot.items():
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(slot))
        handles.append(handle)

    try:
        for start in range(0, n_prompts, batch_size):
            batch = chat_prompts[start:start + batch_size]
            cur_offset[0] = start
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to(device)
            model(**enc, use_cache=False)
            print(f"    forward {start + len(batch)}/{n_prompts}", flush=True)
    finally:
        for h in handles:
            h.remove()

    return out


def summarize(values: torch.Tensor) -> dict:
    v = values.float()
    qs = torch.quantile(v, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    return {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
        "min": float(v.min()),
        "max": float(v.max()),
        "q05": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q95": float(qs[4]),
        "n": int(v.numel()),
        "values": v.tolist(),
    }


def maybe_plot(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not available, skipping plot)")
        return

    lengths = summary["lengths"]
    layers = summary["layers"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer_idx in layers:
        means = []
        stds = []
        for L in lengths:
            cell = summary["cells"][f"L{L}"]["by_layer"][str(layer_idx)]
            means.append(cell["mean"])
            stds.append(cell["std"])
        means_t = torch.tensor(means)
        stds_t = torch.tensor(stds)
        ax.plot(lengths, means_t, marker="o", label=f"layer {layer_idx}")
        ax.fill_between(lengths, means_t - stds_t, means_t + stds_t, alpha=0.15)

    ax.set_xscale("symlog", linthresh=512)
    ax.set_xlabel("context length L (tokens of benign filler)")
    ax.set_ylabel("⟨residual_last · d̂⟩  (mean ± 1σ)")
    ax.set_title("Residual projection onto refusal direction vs context length")
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"  wrote plot {out_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    slug = slug_from_model(args.model)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir(slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]

    refusal_dir = Path(args.refusal_dir)
    d_hat_all = torch.load(refusal_dir / "d_hat_all_layers.pt", map_location="cpu").float()
    meta = json.loads((refusal_dir / "meta.json").read_text())
    n_layers_total = d_hat_all.shape[0]

    for li in args.layers:
        if not (0 <= li < n_layers_total):
            raise ValueError(f"--layers entry {li} out of [0, {n_layers_total})")

    d_hat_subset = torch.stack([d_hat_all[li].contiguous() for li in args.layers], dim=0)

    print(f"Model        : {args.model}")
    print(f"Output dir   : {out_dir}")
    print(f"Layers       : {args.layers}  (of {n_layers_total})")
    print(f"d̂ norms      : "
          f"{[round(d_hat_all[li].norm().item(), 4) for li in args.layers]}  (should all be ≈ 1.0)")
    print(f"Lengths      : {args.lengths}")
    print(f"GPUs         : {torch.cuda.device_count()}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    harmful_all = load_harmful_prompts(args.harmful_n)
    if args.max_prompts is not None:
        harmful_all = harmful_all[: args.max_prompts]
    print(f"Harmful subset (idx 0..{len(harmful_all) - 1}): {len(harmful_all)}")

    summary = {
        "model": args.model,
        "default_layer_from_meta": meta.get("default_layer"),
        "lengths": args.lengths,
        "layers": args.layers,
        "harmful_n": len(harmful_all),
        "harmful_idx_offset": 0,  # disjoint from refusal_direction split
        "enable_thinking": args.enable_thinking,
        "cells": {},
    }

    out_json = out_dir / "projection_sweep.json"
    out_png = out_dir / "projection_sweep.png"

    for L in args.lengths:
        bs = args.batch_size if args.batch_size is not None else adaptive_batch_size(L)
        filler = build_filler(tok, L)
        wrapped = [wrap_prompt(filler, h) for h in harmful_all]
        chat_prompts = [
            format_chat_prompt(tok, w, enable_thinking=args.enable_thinking)
            for w in wrapped
        ]
        actual_filler_tok = (
            len(tok(filler, add_special_tokens=False)["input_ids"]) if filler else 0
        )
        print(f"\n=== L={L}  (batch={bs}, filler_tok={actual_filler_tok}) ===")

        t0 = time.time()
        proj = collect_projections(
            model, tok, chat_prompts,
            layers=args.layers,
            d_hat_subset=d_hat_subset,
            batch_size=bs,
            device=device,
        )
        dt = time.time() - t0

        cell = {
            "L": L,
            "filler_tok": actual_filler_tok,
            "batch_size": bs,
            "wallclock_s": dt,
            "by_layer": {
                str(args.layers[slot]): summarize(proj[slot])
                for slot in range(len(args.layers))
            },
        }
        summary["cells"][f"L{L}"] = cell

        # Persist incrementally so a walltime kill mid-sweep leaves usable data.
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"  wrote {out_json}  ({dt:.1f}s)")
        for slot, layer_idx in enumerate(args.layers):
            s = cell["by_layer"][str(layer_idx)]
            print(f"    layer {layer_idx}: mean={s['mean']:+.3f}  "
                  f"std={s['std']:.3f}  q25={s['q25']:+.3f}  q75={s['q75']:+.3f}")

    print("\n=== Mean projection by (L, layer) ===")
    header = "  L".ljust(10) + "".join(f"layer {li:>2}".rjust(14) for li in args.layers)
    print(header)
    for L in args.lengths:
        row = f"  {L:>6} ".ljust(10)
        for li in args.layers:
            row += f"{summary['cells'][f'L{L}']['by_layer'][str(li)]['mean']:+.3f}".rjust(14)
        print(row)

    if not args.no_plot:
        maybe_plot(summary, out_png)

    print(f"\nWrote: {out_json}")


if __name__ == "__main__":
    main()
