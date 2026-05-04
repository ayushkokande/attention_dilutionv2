"""Policy-refusal vs harm-refusal validity test for d_hat (experiment 7).

Three pools (harm_refusal, policy_refusal, harmless), 50 prompts each. For
each refusal direction d_hat at chosen layers, projects last-token residuals
and computes:

  - per-pool mean / std projection
  - three pairwise AUCs:
      AUC(harm vs harmless)   sanity, expect ~1.0
      AUC(policy vs harmless) does d_hat fire on policy refusals at all?
      AUC(harm vs policy)     KEY: does d_hat distinguish harm from policy?

Decision rule (preregistered, applied to first refusal-dir at layer 20):
  AUC(harm vs policy) >= 0.85          -> HARM_SPECIFIC (paper headline holds)
  AUC(harm vs policy) in [0.65, 0.85)  -> MIXED (caveat required)
  AUC(harm vs policy) < 0.65           -> REFUSAL_GENERIC (reframe paper)

Outputs (under results/<slug>/policy_validity/):
  per_prompt.jsonl, aucs.json, pool_means.json, boxplot.png
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
DECISION_AUC_HARM_SPECIFIC = 0.85
DECISION_AUC_MIXED_LO = 0.65

POOL_FILES = {
    "harm_refusal": "harm_refusal.jsonl",
    "policy_refusal": "policy_refusal.jsonl",
    "harmless": "harmless.jsonl",
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
                   help="Override; default results/<slug>/policy_validity")
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
                    "label_harm": int(row["label_harm"]),
                    "label_policy": int(row["label_policy"]),
                    "category": row.get("category"),
                    "row_id": row.get(
                        "advbench_idx", row.get("alpaca_idx", row.get("index"))
                    ),
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
        d_vec_cpu = d_hat_subset[slot].to(dtype=torch.float32).cpu()

        def hook(module, inp, mod_out):
            h = mod_out[0] if isinstance(mod_out, tuple) else mod_out
            last = h[:, -1, :].detach().float()
            d_vec = d_vec_cpu.to(device=last.device, non_blocking=True)
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


def pool_stats(proj: np.ndarray, pools: np.ndarray) -> dict:
    out: dict = {}
    for name in POOL_FILES:
        mask = pools == name
        sub = proj[mask]
        out[name] = {
            "n": int(mask.sum()),
            "mean": float(sub.mean()) if sub.size else float("nan"),
            "std": float(sub.std(ddof=1)) if sub.size > 1 else float("nan"),
        }
    return out


def three_aucs(proj: np.ndarray, pools: np.ndarray) -> dict:
    is_harm = pools == "harm_refusal"
    is_policy = pools == "policy_refusal"
    is_harmless = pools == "harmless"

    pair_hh = is_harm | is_harmless
    auc_harm_vs_harmless = roc_auc_binary(
        is_harm[pair_hh].astype(int), proj[pair_hh]
    )

    pair_ph = is_policy | is_harmless
    auc_policy_vs_harmless = roc_auc_binary(
        is_policy[pair_ph].astype(int), proj[pair_ph]
    )

    pair_hp = is_harm | is_policy
    auc_harm_vs_policy = roc_auc_binary(
        is_harm[pair_hp].astype(int), proj[pair_hp]
    )

    return {
        "auc_harm_vs_harmless": auc_harm_vs_harmless,
        "auc_policy_vs_harmless": auc_policy_vs_harmless,
        "auc_harm_vs_policy": auc_harm_vs_policy,
    }


def verdict_from_auc(auc_hp: float) -> str:
    if auc_hp >= DECISION_AUC_HARM_SPECIFIC:
        return ("HARM_SPECIFIC — d_hat distinguishes harm-refusal from "
                "policy-refusal; paper headline holds.")
    if auc_hp >= DECISION_AUC_MIXED_LO:
        return ("MIXED — d_hat partially separates harm from policy refusals; "
                "report caveat in paper.")
    return ("REFUSAL_GENERIC — d_hat does not distinguish harm from policy "
            "refusals; reframe paper as generic refusal direction.")


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

    pools = np.array([r["pool"] for r in rows_meta])
    order = ["harmless", "policy_refusal", "harm_refusal"]
    labels = ["harmless\n(Alpaca)", "policy\nrefusal", "harm\nrefusal"]
    data = [proj_layer[pools == name] for name in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(f"projection on {dir_name} (L{layer_idx})")
    ax.set_title(f"Policy vs harm validity — {dir_name} L{layer_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    slug = slug_from_model(args.model)
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else (results_dir(slug) / "policy_validity")
    )
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
        raise RuntimeError("No prompts loaded — run build_policy_pools.py first.")
    print(f"Loaded {len(rows_meta)} prompts from 3 pools.")
    pool_arr = np.array([r["pool"] for r in rows_meta])
    for name in POOL_FILES:
        print(f"  {name}: {(pool_arr == name).sum()}")

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

    auc_payload: dict = {
        "decision_constants": {
            "layer": DECISION_LAYER,
            "auc_harm_vs_policy_harm_specific": DECISION_AUC_HARM_SPECIFIC,
            "auc_harm_vs_policy_mixed_lo": DECISION_AUC_MIXED_LO,
        },
        "refusal_dirs": [],
    }
    pool_payload: dict = {"refusal_dirs": []}

    for di, dname in enumerate(dir_names):
        dir_aucs_by_layer: dict[str, dict] = {}
        dir_pools_by_layer: dict[str, dict] = {}
        verdict = None

        for slot, layer_idx in enumerate(args.layers):
            scores = proj_dict[str(di)][slot].numpy()
            aucs = three_aucs(scores, pool_arr)
            dir_aucs_by_layer[str(layer_idx)] = aucs
            dir_pools_by_layer[str(layer_idx)] = pool_stats(scores, pool_arr)

            if layer_idx == DECISION_LAYER:
                verdict = verdict_from_auc(aucs["auc_harm_vs_policy"])

            print(
                f"[{dname}] L{layer_idx}: "
                f"harm/harmless={aucs['auc_harm_vs_harmless']:.4f}  "
                f"policy/harmless={aucs['auc_policy_vs_harmless']:.4f}  "
                f"harm/policy={aucs['auc_harm_vs_policy']:.4f}"
            )

        auc_payload["refusal_dirs"].append({
            "name": dname,
            "path": str(refusal_dirs[di]),
            "by_layer": dir_aucs_by_layer,
            "verdict_layer20": verdict,
        })
        pool_payload["refusal_dirs"].append({
            "name": dname,
            "by_layer": dir_pools_by_layer,
        })

    out_pp = out_dir / "per_prompt.jsonl"
    with out_pp.open("w", encoding="utf-8") as wf:
        for row in per_lines:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "aucs.json").write_text(json.dumps(auc_payload, indent=2))
    (out_dir / "pool_means.json").write_text(json.dumps(pool_payload, indent=2))

    if not args.no_plot:
        first_dir = dir_names[0]
        slot_dec = args.layers.index(DECISION_LAYER) if DECISION_LAYER in args.layers else 0
        proj_dec = proj_dict["0"][slot_dec].numpy()
        maybe_plot_box(
            rows_meta, proj_dec, first_dir,
            args.layers[slot_dec], out_dir / "boxplot.png",
        )

    print(f"\nWrote {out_pp}")
    print(f"Wrote {out_dir / 'aucs.json'}")
    print(f"Wrote {out_dir / 'pool_means.json'}")
    print(f"\nVerdict (first dir @ L{DECISION_LAYER}):",
          auc_payload["refusal_dirs"][0]["verdict_layer20"])


if __name__ == "__main__":
    main()
