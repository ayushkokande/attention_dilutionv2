"""Compare original d̂ vs covariate-matched d̂* per layer.

Runs anywhere (no GPU). Loads two `d_hat_all_layers.pt` files, computes
cosine similarity per layer, and applies a preregistered decision rule at
the canonical layer (default: 20, the layer used by context_sweep.py and
projection_sweep.py).

Decision rule:
    cos >= 0.90 → CLEAN     : original direction was robust to the
                              verb-class/length confounds; keep using it.
    cos <= 0.70 → CONTAMINATED: confound did meaningful work; adopt d̂*
                              as canonical and re-run projection_sweep.py.
    0.70 < cos < 0.90 → AMBIGUOUS: escalate to the 2x2 factorial ablation
                              described in the validity section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", default="results/qwen3-14b/refusal_direction")
    ap.add_argument("--matched-dir",
                    default="results/qwen3-14b/refusal_direction_matched")
    ap.add_argument("--canonical-layer", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    orig_path = Path(args.orig_dir) / "d_hat_all_layers.pt"
    matched_path = Path(args.matched_dir) / "d_hat_all_layers.pt"
    if not orig_path.exists():
        raise SystemExit(f"missing {orig_path}")
    if not matched_path.exists():
        raise SystemExit(f"missing {matched_path}")

    d = torch.load(orig_path, map_location="cpu").float()
    d_star = torch.load(matched_path, map_location="cpu").float()

    if d.shape != d_star.shape:
        raise SystemExit(f"shape mismatch: {tuple(d.shape)} vs {tuple(d_star.shape)}")

    cos = torch.nn.functional.cosine_similarity(d, d_star, dim=-1)
    n_layers = cos.numel()

    print(f"\nLayers compared : {n_layers}")
    print(f"Canonical layer : {args.canonical_layer}\n")
    print(f"{'layer':>6}  {'cos(d̂, d̂*)':>14}")
    print(f"{'-' * 6}  {'-' * 14}")
    for L in range(n_layers):
        marker = "  <-- canonical" if L == args.canonical_layer else ""
        print(f"{L:>6}  {cos[L].item():>14.4f}{marker}")

    c_canon = float(cos[args.canonical_layer])
    print()
    print(f"Canonical-layer cosine : {c_canon:.4f}")
    if c_canon >= 0.90:
        verdict = ("CLEAN — original d̂ ≈ matched d̂*. The diff-of-means recipe "
                   "was robust to verb-class/length confounds in this dataset. "
                   "Keep using results/qwen3-14b/refusal_direction/ downstream "
                   "and add a one-paragraph validity note in the paper.")
    elif c_canon <= 0.70:
        verdict = ("CONTAMINATED — original d̂ differs materially from matched "
                   f"d̂* at layer {args.canonical_layer}. Adopt d̂_matched as the "
                   "canonical direction: switch context_sweep.py and "
                   "projection_sweep.py to point at "
                   "results/qwen3-14b/refusal_direction_matched/ and re-run.")
    else:
        verdict = ("AMBIGUOUS — 0.70 < cos < 0.90 at the canonical layer. "
                   "Escalate to the 2x2 factorial ablation (cross-style "
                   "generalization test) before deciding which direction to "
                   "use downstream.")
    print(f"Verdict                : {verdict}")

    summary_path = Path(args.matched_dir) / "compare_to_orig.json"
    summary_path.write_text(json.dumps({
        "orig_dir": args.orig_dir,
        "matched_dir": args.matched_dir,
        "canonical_layer": args.canonical_layer,
        "cos_per_layer": cos.tolist(),
        "cos_canonical": c_canon,
        "verdict": verdict,
    }, indent=2))
    print(f"\nWrote {summary_path}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available, skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(range(n_layers), cos.tolist(), marker="o", linewidth=1.2)
    ax.axhline(0.90, color="green", linestyle="--", alpha=0.55, label="clean (≥ 0.90)")
    ax.axhline(0.70, color="red", linestyle="--", alpha=0.55, label="contaminated (≤ 0.70)")
    ax.axvline(args.canonical_layer, color="black", linestyle=":", alpha=0.4,
               label=f"canonical layer {args.canonical_layer}")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine(d̂, d̂*)")
    ax.set_title("Refusal direction: original vs covariate-matched")
    ax.set_ylim(min(-0.05, cos.min().item() - 0.05), 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out_png = Path(args.matched_dir) / "cos_per_layer.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"Wrote plot {out_png}")


if __name__ == "__main__":
    main()
