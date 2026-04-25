"""Compute the per-layer refusal direction d̂ for Qwen3-14B.

Difference-of-means (Arditi et al., 2024): for each transformer block, take
the mean last-token residual on harmful prompts minus the mean on harmless
prompts, then unit-normalize. We save d̂ for *every* layer; the sweep script
picks one (default: middle) to project out at every block.

The harmful prompts used here are taken from AdvBench rows
[SWEEP_HARMFUL_N : SWEEP_HARMFUL_N + n_harmful], i.e. *disjoint* from the
rows context_sweep.py runs on, so the direction is not estimated from the
same prompts it will later be evaluated against.
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
    load_harmless_prompts,
    results_dir,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Must match context_sweep.py's SWEEP_HARMFUL_N so the d̂ split and the sweep
# split are disjoint AdvBench rows.
SWEEP_HARMFUL_N = 100


def slug_from_model(m: str) -> str:
    return m.split("/")[-1].lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--n-harmful", type=int, default=256)
    p.add_argument("--n-harmless", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--output-dir", default=None,
                   help="Override; default: results/<slug>/refusal_direction")
    return p.parse_args()


@torch.no_grad()
def collect_last_token_residuals(model, tokenizer, prompts, batch_size, device):
    """Return tensor (n_layers, n_prompts, d_model) on CPU/float32."""
    layers = model.model.layers
    n_layers = len(layers)
    cached: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    def make_hook(i):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # tokenizer.padding_side="left", so the last position is the last
            # real token for every row in the batch.
            cached[i].append(h[:, -1, :].detach().to("cpu", torch.float32))
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
            model(**enc, use_cache=False)
            print(f"  forward {start + len(batch)}/{len(prompts)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    return torch.stack([torch.cat(cached[i], dim=0) for i in range(n_layers)], dim=0)


def main() -> None:
    args = parse_args()
    slug = slug_from_model(args.model)
    out_dir = Path(args.output_dir) if args.output_dir else (results_dir(slug) / "refusal_direction")
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]

    print(f"Model       : {args.model}")
    print(f"Slug        : {slug}")
    print(f"Output      : {out_dir}")
    print(f"GPUs        : {torch.cuda.device_count()}")
    print(f"dtype       : {args.dtype}  | batch_size={args.batch_size}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # Disjoint from the sweep split: skip the first SWEEP_HARMFUL_N rows.
    all_harmful = load_harmful_prompts(SWEEP_HARMFUL_N + args.n_harmful)
    harmful = all_harmful[SWEEP_HARMFUL_N:SWEEP_HARMFUL_N + args.n_harmful]
    harmless = load_harmless_prompts(args.n_harmless)
    print(f"Harmful  (idx {SWEEP_HARMFUL_N}..) : {len(harmful)}")
    print(f"Harmless                        : {len(harmless)}")

    harmful_p = [format_chat_prompt(tok, p, enable_thinking=args.enable_thinking) for p in harmful]
    harmless_p = [format_chat_prompt(tok, p, enable_thinking=args.enable_thinking) for p in harmless]

    print("\n[harmful forward]")
    t0 = time.time()
    h_harmful = collect_last_token_residuals(model, tok, harmful_p, args.batch_size, device)
    print(f"  done {time.time() - t0:.1f}s | shape {tuple(h_harmful.shape)}")

    print("\n[harmless forward]")
    t0 = time.time()
    h_harmless = collect_last_token_residuals(model, tok, harmless_p, args.batch_size, device)
    print(f"  done {time.time() - t0:.1f}s | shape {tuple(h_harmless.shape)}")

    mu_harmful = h_harmful.mean(dim=1)
    mu_harmless = h_harmless.mean(dim=1)
    diff = mu_harmful - mu_harmless                       # (n_layers, d_model)
    norms = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    d_hat = diff / norms                                  # unit vectors per layer

    n_layers, d_model = d_hat.shape
    pt_path = out_dir / "d_hat_all_layers.pt"
    torch.save(d_hat, pt_path)

    meta = {
        "model": args.model,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "harmful_idx_offset": SWEEP_HARMFUL_N,
        "diff_norms_per_layer": norms.squeeze(-1).tolist(),
        "default_layer": n_layers // 2,
        "suggested_layers_to_try": [
            int(round(n_layers * 0.50)),
            int(round(n_layers * 0.60)),
            int(round(n_layers * 0.70)),
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved {pt_path}")
    print(f"      {out_dir / 'meta.json'}")
    print(f"\nNext: python experiment_2/context_sweep.py --refusal-dir {pt_path.parent}")


if __name__ == "__main__":
    main()
