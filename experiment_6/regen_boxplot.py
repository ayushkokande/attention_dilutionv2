"""Regenerate exp_6 boxplot at canonical layer 18 from per_prompt.jsonl.

Standalone helper. Stale boxplot.png was rendered at L20 before cascade.
No GPU needed — reads cached projections only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP1 = ROOT / "experiment_1"
for _p in (str(HERE), str(EXP1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import results_dir  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-prompt", default=None)
    p.add_argument("--slug", default="qwen3-14b")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--dir-name", default="refusal_direction",
                   help="Which refusal direction key to plot.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.per_prompt:
        pp_path = Path(args.per_prompt)
    else:
        pp_path = results_dir(args.slug) / "topic_decouple" / "per_prompt.jsonl"
    if not pp_path.exists():
        raise FileNotFoundError(f"Missing per_prompt.jsonl: {pp_path}")

    rows = []
    with pp_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    intent = np.array([r["label_intent"] for r in rows])
    topic = np.array([r["label_topic"] for r in rows])
    proj = np.array(
        [r["projections"][args.dir_name][str(args.layer)] for r in rows],
        dtype=np.float64,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    cell_labels = ["harmless\nmundane", "harmless\nedgy",
                   "harmful\nmundane", "harmful\nedgy"]
    data = [proj[(intent == i) & (topic == t)] for (i, t) in cells_order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=cell_labels, showmeans=True)
    ax.set_ylabel(f"projection on {args.dir_name} (L{args.layer})")
    ax.set_title(f"Topic decoupling 2x2 — {args.dir_name} L{args.layer}")
    fig.tight_layout()

    out_path = Path(args.output) if args.output else (
        pp_path.parent / "boxplot.png"
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
