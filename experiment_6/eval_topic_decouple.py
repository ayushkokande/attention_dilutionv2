"""Topic-decoupling validity test for learned refusal directions (experiment 6).

2x2 factorial (intent x topic) over 200 prompts. For each refusal direction
d_hat at chosen layers, projects last-token residuals and computes:

  - per-cell mean / std projection
  - 2-way ANOVA on layer 20 projection: proj ~ intent + topic + intent:topic
    with partial eta-squared per term
  - AUC_intent (collapse topic), AUC_topic (collapse intent)

Decision rule (preregistered, applied to first refusal-dir at layer 20):
  eta2(intent) >= 0.5 AND eta2(topic) <= 0.1            -> CLEAN
  eta2(intent) >= eta2(topic) AND eta2(topic) > 0.1     -> MIXED
  eta2(topic) > eta2(intent)                            -> TOPIC_DOMINANT

Outputs (under results/<slug>/topic_decouple/):
  per_prompt.jsonl, anova.json, cell_means.json,
  boxplot.png, scatter_intent_vs_topic.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP1 = ROOT / "experiment_1"
EXP2 = ROOT / "experiment_2"
for _p in (str(HERE), str(EXP1), str(EXP2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import format_chat_prompt, results_dir  # type: ignore
from refusal_direction import slug_from_model  # type: ignore

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

DEFAULT_LAYERS = [20, 28]
DECISION_LAYER = 20
DECISION_INTENT_HI = 0.5
DECISION_TOPIC_LO = 0.1

POOL_FILES = {
    "harmful_edgy_topic": "harmful_edgy_topic.jsonl",
    "harmful_mundane_topic": "harmful_mundane_topic.jsonl",
    "harmless_edgy_topic": "harmless_edgy_topic.jsonl",
    "harmless_mundane_topic": "harmless_mundane_topic.jsonl",
}


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_pos_ranks = ranks[pos_mask].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--refusal-dirs", nargs="+", required=True)
    p.add_argument("--refusal-dir-names", nargs="+", default=None)
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None,
                   help="Override; default results/<slug>/topic_decouple")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load_pools(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for pool_name, fname in POOL_FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing pool file: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row["prompt"]).strip()
                if not text:
                    continue
                rows.append({
                    "pool": pool_name,
                    "prompt_raw": text,
                    "label_intent": int(row["label_intent"]),
                    "label_topic": int(row["label_topic"]),
                    "row_id": row.get("advbench_idx", row.get("index")),
                })
    return rows


@torch.no_grad()
def collect_projections_one_dir(
    model,
    tok,
    chat_prompts: list[str],
    layers: list[int],
    d_hat_subset: torch.Tensor,
    batch_size: int,
    device,
) -> torch.Tensor:
    n_prompts = len(chat_prompts)
    n_layers_chosen = len(layers)
    proj_row = torch.zeros(n_layers_chosen, n_prompts, dtype=torch.float32)

    block_to_slot = {layer_idx: slot for slot, layer_idx in enumerate(layers)}
    cur_offset = [0]

    def make_hook(slot: int):
        d_vec = d_hat_subset[slot].to(device=device, dtype=torch.float32)

        def hook(module, inp, mod_out):
            h = mod_out[0] if isinstance(mod_out, tuple) else mod_out
            last = h[:, -1, :].detach().float()
            p = (last * d_vec.unsqueeze(0)).sum(dim=-1)
            start = cur_offset[0]
            proj_row[slot, start : start + last.shape[0]] = p.cpu()

        return hook

    handles = [
        model.model.layers[layer_idx].register_forward_hook(
            make_hook(block_to_slot[layer_idx])
        )
        for layer_idx in layers
    ]

    try:
        for start in range(0, n_prompts, batch_size):
            batch = chat_prompts[start : start + batch_size]
            cur_offset[0] = start
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to(device)
            model(**enc, use_cache=False)
            print(f"  forward {start + len(batch)}/{n_prompts}", flush=True)
    finally:
        for h in handles:
            h.remove()

    return proj_row


def two_way_anova(
    proj: np.ndarray, intent: np.ndarray, topic: np.ndarray
) -> dict:
    """Type-II 2-way ANOVA with partial eta-squared per term."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    df = pd.DataFrame({
        "proj": proj.astype(np.float64),
        "intent": intent.astype(int),
        "topic": topic.astype(int),
    })
    model = ols("proj ~ C(intent) + C(topic) + C(intent):C(topic)", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)

    out = {"terms": {}, "residual_df": float(aov.loc["Residual", "df"]),
           "r_squared": float(model.rsquared)}
    ss_resid = float(aov.loc["Residual", "sum_sq"])
    for term in ["C(intent)", "C(topic)", "C(intent):C(topic)"]:
        ss = float(aov.loc[term, "sum_sq"])
        df_t = float(aov.loc[term, "df"])
        f = float(aov.loc[term, "F"])
        p = float(aov.loc[term, "PR(>F)"])
        partial_eta2 = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else float("nan")
        out["terms"][term] = {
            "sum_sq": ss, "df": df_t, "F": f, "p": p,
            "partial_eta2": partial_eta2,
        }
    return out


def cell_stats(proj: np.ndarray, intent: np.ndarray, topic: np.ndarray) -> dict:
    cells = {}
    for i in (0, 1):
        for t in (0, 1):
            mask = (intent == i) & (topic == t)
            sub = proj[mask]
            cells[f"intent={i},topic={t}"] = {
                "n": int(mask.sum()),
                "mean": float(sub.mean()) if sub.size else float("nan"),
                "std": float(sub.std(ddof=1)) if sub.size > 1 else float("nan"),
            }
    return cells


def verdict_from_anova(anova: dict) -> str:
    eta_intent = anova["terms"]["C(intent)"]["partial_eta2"]
    eta_topic = anova["terms"]["C(topic)"]["partial_eta2"]
    if eta_intent >= DECISION_INTENT_HI and eta_topic <= DECISION_TOPIC_LO:
        return "CLEAN — d_hat tracks harmful intent; topic effect negligible."
    if eta_intent >= eta_topic and eta_topic > DECISION_TOPIC_LO:
        return "MIXED — intent dominates but topic non-trivial; orthogonalise as robustness check."
    if eta_topic > eta_intent:
        return "TOPIC_DOMINANT — d_hat tracks harmful-topic, not harm-intent. Reframe required."
    return "INDETERMINATE — manual inspection required."


def maybe_plot_box(
    rows_meta: list[dict],
    proj_layer: np.ndarray,
    dir_name: str,
    layer_idx: int,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    intent = np.array([r["label_intent"] for r in rows_meta])
    topic = np.array([r["label_topic"] for r in rows_meta])
    cells_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    cell_labels = ["harmless\nmundane", "harmless\nedgy", "harmful\nmundane", "harmful\nedgy"]
    data = [proj_layer[(intent == i) & (topic == t)] for (i, t) in cells_order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=cell_labels, showmeans=True)
    ax.set_ylabel(f"projection on {dir_name} (L{layer_idx})")
    ax.set_title(f"Topic decoupling 2x2 — {dir_name} L{layer_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def maybe_plot_scatter_aucs(
    aucs_by_dir: dict, layers: list[int], out_path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    for dname, layer_aucs in aucs_by_dir.items():
        for layer_idx in layers:
            ai = layer_aucs[layer_idx]["auc_intent"]
            at = layer_aucs[layer_idx]["auc_topic"]
            ax.scatter(at, ai, s=60)
            ax.annotate(f"{dname} L{layer_idx}", (at, ai), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.7, linestyle="--")
    ax.axhline(0.5, color="grey", linewidth=0.5)
    ax.axvline(0.5, color="grey", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("AUC_topic (collapse intent)")
    ax.set_ylabel("AUC_intent (collapse topic)")
    ax.set_title("Intent- vs topic-discrimination per d_hat / layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    slug = slug_from_model(args.model)
    out_dir = Path(args.output_dir) if args.output_dir else (results_dir(slug) / "topic_decouple")
    out_dir.mkdir(parents=True, exist_ok=True)

    refusal_dirs = [Path(p) for p in args.refusal_dirs]
    dir_names = args.refusal_dir_names or [p.name for p in refusal_dirs]
    if len(dir_names) != len(refusal_dirs):
        raise ValueError("--refusal-dir-names length must match --refusal-dirs")

    d_hat_list = []
    for rd in refusal_dirs:
        dh = torch.load(rd / "d_hat_all_layers.pt", map_location="cpu").float()
        n_layers_total = dh.shape[0]
        for li in args.layers:
            if not (0 <= li < n_layers_total):
                raise ValueError(f"layer {li} out of range [0,{n_layers_total})")
        subset = torch.stack([dh[li].contiguous() for li in args.layers], dim=0)
        d_hat_list.append(subset)

    rows_meta = load_pools(Path(args.data_dir))
    if not rows_meta:
        raise RuntimeError("No prompts loaded — run build_topic_pools.py first.")
    print(f"Loaded {len(rows_meta)} prompts from 4 pools.")

    intent_arr = np.array([r["label_intent"] for r in rows_meta], dtype=np.int64)
    topic_arr = np.array([r["label_topic"] for r in rows_meta], dtype=np.int64)
    print(f"  intent_pos={int(intent_arr.sum())}  topic_pos={int(topic_arr.sum())}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    chat_prompts = [
        format_chat_prompt(tok, r["prompt_raw"], enable_thinking=args.enable_thinking)
        for r in rows_meta
    ]

    proj_dict: dict[str, torch.Tensor] = {}
    for di, dh_sub in enumerate(d_hat_list):
        print(f"\n-- refusal dir {di} ({dir_names[di]}) --", flush=True)
        proj_dict[str(di)] = collect_projections_one_dir(
            model, tok, chat_prompts, args.layers, dh_sub,
            args.batch_size, device,
        )

    per_lines = []
    for pi, rm in enumerate(rows_meta):
        row = dict(rm)
        row["projections"] = {}
        for di, dname in enumerate(dir_names):
            row["projections"][dname] = {
                str(layer_idx): float(proj_dict[str(di)][slot, pi])
                for slot, layer_idx in enumerate(args.layers)
            }
        per_lines.append(row)

    anova_payload: dict = {
        "decision_constants": {
            "layer": DECISION_LAYER,
            "intent_eta2_hi": DECISION_INTENT_HI,
            "topic_eta2_lo": DECISION_TOPIC_LO,
        },
        "refusal_dirs": [],
    }
    cell_payload: dict = {"refusal_dirs": []}
    aucs_by_dir: dict[str, dict[int, dict[str, float]]] = {}

    for di, dname in enumerate(dir_names):
        dir_aucs: dict[int, dict[str, float]] = {}
        dir_anova_by_layer: dict[str, dict] = {}
        dir_cells_by_layer: dict[str, dict] = {}
        verdict = None

        for slot, layer_idx in enumerate(args.layers):
            scores = proj_dict[str(di)][slot].numpy()
            auc_i = roc_auc_binary(intent_arr, scores)
            auc_t = roc_auc_binary(topic_arr, scores)
            dir_aucs[layer_idx] = {"auc_intent": auc_i, "auc_topic": auc_t}

            anova = two_way_anova(scores, intent_arr, topic_arr)
            anova["auc_intent"] = auc_i
            anova["auc_topic"] = auc_t
            dir_anova_by_layer[str(layer_idx)] = anova
            dir_cells_by_layer[str(layer_idx)] = cell_stats(scores, intent_arr, topic_arr)

            if layer_idx == DECISION_LAYER:
                verdict = verdict_from_anova(anova)

            print(
                f"[{dname}] L{layer_idx}: "
                f"AUC_intent={auc_i:.4f}  AUC_topic={auc_t:.4f}  "
                f"eta2(intent)={anova['terms']['C(intent)']['partial_eta2']:.3f}  "
                f"eta2(topic)={anova['terms']['C(topic)']['partial_eta2']:.3f}"
            )

        aucs_by_dir[dname] = dir_aucs
        anova_payload["refusal_dirs"].append({
            "name": dname,
            "path": str(refusal_dirs[di]),
            "by_layer": dir_anova_by_layer,
            "verdict_layer20": verdict,
        })
        cell_payload["refusal_dirs"].append({
            "name": dname,
            "by_layer": dir_cells_by_layer,
        })

    out_pp = out_dir / "per_prompt.jsonl"
    with out_pp.open("w", encoding="utf-8") as wf:
        for row in per_lines:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "anova.json").write_text(json.dumps(anova_payload, indent=2))
    (out_dir / "cell_means.json").write_text(json.dumps(cell_payload, indent=2))

    if not args.no_plot:
        # boxplot for first dir at decision layer
        first_dir = dir_names[0]
        slot_dec = args.layers.index(DECISION_LAYER) if DECISION_LAYER in args.layers else 0
        proj_dec = proj_dict["0"][slot_dec].numpy()
        maybe_plot_box(
            rows_meta, proj_dec, first_dir,
            args.layers[slot_dec], out_dir / "boxplot.png",
        )
        maybe_plot_scatter_aucs(
            aucs_by_dir, args.layers, out_dir / "scatter_intent_vs_topic.png",
        )

    print(f"\nWrote {out_pp}")
    print(f"Wrote {out_dir / 'anova.json'}")
    print(f"Wrote {out_dir / 'cell_means.json'}")
    print(f"\nVerdict (first dir @ L{DECISION_LAYER}):",
          anova_payload["refusal_dirs"][0]["verdict_layer20"])


if __name__ == "__main__":
    main()
