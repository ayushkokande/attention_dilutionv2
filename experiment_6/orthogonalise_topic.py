"""Orthogonalisation robustness check for exp_6 MIXED verdict.

When 2x2 ANOVA at canonical layer 18 returns MIXED (eta2(topic) > 0.1),
preregistered rule says: orthogonalise topic from projection, retest intent
effect. If intent effect survives, topic-orthogonal harm-intent direction
claim still defensible.

Two analyses, both consume per_prompt.jsonl (no GPU):

  A. Within-topic AUC_intent. Test intent discrimination separately for
     topic=0 (mundane) and topic=1 (edgy) cells. If both AUCs ~1.0,
     intent effect is topic-invariant.
  B. Linear orthogonalisation. Regress projection on topic, take residuals,
     recompute eta2(intent) and AUC_intent on residuals. Survival of
     decision threshold (>=0.5) on residuals = robust.

Output: results/<slug>/topic_decouple/orthogonalisation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP1 = ROOT / "experiment_1"
for _p in (str(HERE), str(EXP1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import results_dir  # type: ignore

DECISION_LAYER = 18
INTENT_HI = 0.5


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


def two_way_anova(proj: np.ndarray, intent: np.ndarray, topic: np.ndarray) -> dict:
    """Type-II 2x2 ANOVA via pure numpy. Assumes balanced cells.

    For balanced 2x2 (n per cell equal), Type I = Type II = Type III sums of
    squares. Returns partial eta-squared per term for compatibility with the
    statsmodels output in eval_topic_decouple.py.
    """
    y = proj.astype(np.float64)
    a = intent.astype(int)
    b = topic.astype(int)

    cell_n = {}
    cell_mean = {}
    for i in (0, 1):
        for t in (0, 1):
            mask = (a == i) & (b == t)
            cell_n[(i, t)] = int(mask.sum())
            cell_mean[(i, t)] = float(y[mask].mean()) if mask.any() else 0.0

    n_per_cell = cell_n[(0, 0)]
    if not all(v == n_per_cell for v in cell_n.values()):
        raise ValueError(f"Unbalanced cells {cell_n} — pure-numpy ANOVA requires balance.")

    grand_mean = y.mean()
    intent_means = {
        i: 0.5 * (cell_mean[(i, 0)] + cell_mean[(i, 1)]) for i in (0, 1)
    }
    topic_means = {
        t: 0.5 * (cell_mean[(0, t)] + cell_mean[(1, t)]) for t in (0, 1)
    }

    ss_intent = sum(
        2 * n_per_cell * (intent_means[i] - grand_mean) ** 2 for i in (0, 1)
    )
    ss_topic = sum(
        2 * n_per_cell * (topic_means[t] - grand_mean) ** 2 for t in (0, 1)
    )
    ss_interaction = sum(
        n_per_cell
        * (cell_mean[(i, t)] - intent_means[i] - topic_means[t] + grand_mean) ** 2
        for i in (0, 1) for t in (0, 1)
    )
    ss_resid = 0.0
    for i in (0, 1):
        for t in (0, 1):
            mask = (a == i) & (b == t)
            ss_resid += float(((y[mask] - cell_mean[(i, t)]) ** 2).sum())

    ss_total = ss_intent + ss_topic + ss_interaction + ss_resid
    df_resid = float(4 * (n_per_cell - 1))

    out = {
        "terms": {},
        "residual_df": df_resid,
        "r_squared": float(1.0 - ss_resid / ss_total) if ss_total > 0 else float("nan"),
    }
    for term, ss, df_t in [
        ("C(intent)", ss_intent, 1.0),
        ("C(topic)", ss_topic, 1.0),
        ("C(intent):C(topic)", ss_interaction, 1.0),
    ]:
        partial_eta2 = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else float("nan")
        ms = ss / df_t
        ms_resid = ss_resid / df_resid if df_resid > 0 else float("nan")
        f_stat = ms / ms_resid if ms_resid > 0 else float("nan")
        out["terms"][term] = {
            "sum_sq": float(ss),
            "df": df_t,
            "F": float(f_stat),
            "partial_eta2": float(partial_eta2),
        }
    return out


def orthogonalise_topic(proj: np.ndarray, topic: np.ndarray) -> np.ndarray:
    """Regress proj on topic indicator, return residuals."""
    t = topic.astype(np.float64)
    p = proj.astype(np.float64)
    t_mean = t.mean()
    p_mean = p.mean()
    cov = ((t - t_mean) * (p - p_mean)).sum()
    var = ((t - t_mean) ** 2).sum()
    beta = cov / var if var > 0 else 0.0
    alpha = p_mean - beta * t_mean
    pred = alpha + beta * t
    return p - pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-prompt", default=None,
                   help="Path to per_prompt.jsonl. Default: "
                        "results/<slug>/topic_decouple/per_prompt.jsonl")
    p.add_argument("--slug", default="qwen3-14b")
    p.add_argument("--layer", type=int, default=DECISION_LAYER)
    p.add_argument("--output", default=None,
                   help="Output JSON path. Default: <per_prompt_dir>/orthogonalisation.json")
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
    print(f"Loaded {len(rows)} rows from {pp_path}")

    intent = np.array([r["label_intent"] for r in rows], dtype=np.int64)
    topic = np.array([r["label_topic"] for r in rows], dtype=np.int64)
    dir_names = sorted(rows[0]["projections"].keys())

    layer_key = str(args.layer)
    payload: dict = {
        "decision_layer": args.layer,
        "intent_hi_threshold": INTENT_HI,
        "n_total": len(rows),
        "n_per_cell": {
            f"intent={i},topic={t}": int(((intent == i) & (topic == t)).sum())
            for i in (0, 1) for t in (0, 1)
        },
        "refusal_dirs": [],
    }

    for dname in dir_names:
        proj = np.array([r["projections"][dname][layer_key] for r in rows],
                        dtype=np.float64)

        baseline_anova = two_way_anova(proj, intent, topic)
        baseline_eta2_intent = baseline_anova["terms"]["C(intent)"]["partial_eta2"]
        baseline_eta2_topic = baseline_anova["terms"]["C(topic)"]["partial_eta2"]
        baseline_auc_intent = roc_auc_binary(intent, proj)

        within_topic = {}
        for t_level in (0, 1):
            mask = topic == t_level
            sub_proj = proj[mask]
            sub_intent = intent[mask]
            within_topic[f"topic={t_level}"] = {
                "n": int(mask.sum()),
                "n_intent_pos": int((sub_intent == 1).sum()),
                "n_intent_neg": int((sub_intent == 0).sum()),
                "auc_intent": roc_auc_binary(sub_intent, sub_proj),
                "mean_intent_pos": float(sub_proj[sub_intent == 1].mean())
                if (sub_intent == 1).any() else float("nan"),
                "mean_intent_neg": float(sub_proj[sub_intent == 0].mean())
                if (sub_intent == 0).any() else float("nan"),
                "delta_intent": float(
                    sub_proj[sub_intent == 1].mean() - sub_proj[sub_intent == 0].mean()
                ),
            }

        proj_resid = orthogonalise_topic(proj, topic)
        resid_anova = two_way_anova(proj_resid, intent, topic)
        resid_eta2_intent = resid_anova["terms"]["C(intent)"]["partial_eta2"]
        resid_eta2_topic = resid_anova["terms"]["C(topic)"]["partial_eta2"]
        resid_auc_intent = roc_auc_binary(intent, proj_resid)

        auc_within_min = min(within_topic["topic=0"]["auc_intent"],
                             within_topic["topic=1"]["auc_intent"])
        survives = (
            resid_eta2_intent >= INTENT_HI
            and auc_within_min >= 0.85
        )
        if survives:
            verdict = (
                "ROBUST — intent effect survives topic orthogonalisation. "
                "d_hat tracks topic-orthogonal harm-intent direction; "
                "headline claim defensible despite MIXED 2x2 verdict."
            )
        else:
            verdict = (
                "NOT_ROBUST — intent effect drops below threshold after "
                "topic orthogonalisation. Reframe required: d_hat may be "
                "harm-and-topic mixture, not pure intent direction."
            )

        payload["refusal_dirs"].append({
            "name": dname,
            "baseline": {
                "eta2_intent": baseline_eta2_intent,
                "eta2_topic": baseline_eta2_topic,
                "auc_intent": baseline_auc_intent,
            },
            "within_topic": within_topic,
            "topic_orthogonalised": {
                "eta2_intent": resid_eta2_intent,
                "eta2_topic": resid_eta2_topic,
                "auc_intent": resid_auc_intent,
            },
            "auc_within_min": auc_within_min,
            "verdict": verdict,
        })

        print(f"\n[{dname}] L{args.layer}")
        print(f"  baseline:        eta2(intent)={baseline_eta2_intent:.3f} "
              f"eta2(topic)={baseline_eta2_topic:.3f} AUC_intent={baseline_auc_intent:.4f}")
        print(f"  within topic=0:  AUC_intent={within_topic['topic=0']['auc_intent']:.4f} "
              f"delta={within_topic['topic=0']['delta_intent']:.2f}")
        print(f"  within topic=1:  AUC_intent={within_topic['topic=1']['auc_intent']:.4f} "
              f"delta={within_topic['topic=1']['delta_intent']:.2f}")
        print(f"  orthogonalised:  eta2(intent)={resid_eta2_intent:.3f} "
              f"eta2(topic)={resid_eta2_topic:.3f} AUC_intent={resid_auc_intent:.4f}")
        print(f"  verdict: {verdict}")

    out_path = Path(args.output) if args.output else (
        pp_path.parent / "orthogonalisation.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
