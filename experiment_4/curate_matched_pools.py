"""Curate covariate-matched harmful/harmless pools for E-prof-2.

Goal: re-run the diff-of-means recipe on prompt pools that are matched on the
two surface confounds the prof flagged (verb class and length), so the
resulting direction d̂* can only encode axes the original AdvBench/Alpaca
contrast did *not* control out — i.e., closer to pure harmfulness.

Outputs:
    experiment_4/data/matched_harmful.jsonl   one {"prompt": "..."} per line
    experiment_4/data/matched_harmless.jsonl  same
    experiment_4/data/curation_stats.json     pre/post-filter counts + verb hist

Restrictions applied to both pools:
    - Disjoint from context_sweep eval split (harmful comes from AdvBench[100:]).
    - Opening verb ∈ PRODUCE_ARTIFACT_VERBS (verb-class match).
    - Existing harmless-content filter applied to Alpaca side.
    - Word-length histogram of harmless is stratify-sampled to mirror harmful,
      so the two pools have identical length distributions by construction.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP1 = ROOT / "experiment_1"
sys.path.insert(0, str(EXP1))
from utils import _HARMLESS_EXCLUDE_SUBSTRINGS  # type: ignore

# "Produce-an-artifact" imperative verbs — the dominant class on the harmful
# side of the original contrast (write/create/develop/provide = 70% of openings).
# Restricting both pools to this set controls the verb-class confound.
PRODUCE_ARTIFACT_VERBS = frozenset({
    "write", "create", "develop", "design", "generate", "produce",
    "compose", "build", "make", "draft", "provide", "give", "outline",
    "construct", "craft",
})

ADV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ADV_CACHE = Path("/tmp/adv.csv")

# Reserve AdvBench[0:100) for context_sweep eval; harmful pool is drawn from [100:].
SWEEP_HARMFUL_N = 100

# Pull a generous Alpaca slice; final pool is much smaller after filtering.
ALPACA_CANDIDATE_ROWS = 20000

OUT_DIR = HERE / "data"


def first_verb(prompt: str) -> str:
    if not prompt.strip():
        return ""
    return prompt.strip().split()[0].lower().rstrip(",.:?!")


def load_advbench() -> list[str]:
    if not ADV_CACHE.exists():
        urllib.request.urlretrieve(ADV_URL, ADV_CACHE)
    with ADV_CACHE.open() as f:
        return [r["goal"] for r in csv.DictReader(f)]


def load_alpaca_harmless(max_rows: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Need the `datasets` package. Activate your venv and run:\n"
            "    pip install datasets"
        )
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out: list[str] = []
    for row in ds:
        if row.get("input"):
            continue
        instr = str(row["instruction"])
        low = instr.lower()
        if any(s in low for s in _HARMLESS_EXCLUDE_SUBSTRINGS):
            continue
        out.append(instr)
        if len(out) >= max_rows:
            break
    return out


def filter_by_verb(pool: list[str], allowed: frozenset[str]) -> list[str]:
    return [p for p in pool if first_verb(p) in allowed]


def length_match(
    harmful: list[str],
    harmless_candidates: list[str],
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Stratify-sample harmless to mirror harmful's word-length histogram.

    For each length bucket L present in the harmful pool, take min(want, have)
    harmless prompts of that exact word length. Trim harmful's L-bucket to the
    same count, so both pools end up with identical length histograms (and
    therefore identical mean and median by construction).
    """
    rng = random.Random(seed)
    ref_hist = Counter(len(p.split()) for p in harmful)

    by_len: dict[int, list[str]] = defaultdict(list)
    for p in harmless_candidates:
        by_len[len(p.split())].append(p)

    harmful_out: list[str] = []
    harmless_out: list[str] = []
    shortfalls: list[tuple[int, int, int]] = []
    for L, n_needed in sorted(ref_hist.items()):
        bucket = by_len.get(L, [])
        rng.shuffle(bucket)
        n_take = min(n_needed, len(bucket))
        if n_take < n_needed:
            shortfalls.append((L, n_needed, n_take))
        harmless_out.extend(bucket[:n_take])
        harmful_at_L = [p for p in harmful if len(p.split()) == L]
        rng.shuffle(harmful_at_L)
        harmful_out.extend(harmful_at_L[:n_take])

    if shortfalls:
        print("  length-bucket shortfalls (harmful want > harmless have):", file=sys.stderr)
        for L, want, got in shortfalls:
            print(f"    L={L}w  want={want}  got={got}", file=sys.stderr)
    return harmful_out, harmless_out


def write_jsonl(prompts: list[str], path: Path) -> None:
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    advbench = load_advbench()
    harmful_pool = advbench[SWEEP_HARMFUL_N:]
    harmful_verb = filter_by_verb(harmful_pool, PRODUCE_ARTIFACT_VERBS)
    print(f"AdvBench rows [{SWEEP_HARMFUL_N}:end] = {len(harmful_pool)}  "
          f"→ {len(harmful_verb)} after verb-filter")

    print(f"Loading up to {ALPACA_CANDIDATE_ROWS} Alpaca harmless candidates "
          f"(may take ~30s)...")
    harmless_pool = load_alpaca_harmless(ALPACA_CANDIDATE_ROWS)
    harmless_verb = filter_by_verb(harmless_pool, PRODUCE_ARTIFACT_VERBS)
    print(f"Alpaca harmless = {len(harmless_pool)}  "
          f"→ {len(harmless_verb)} after verb-filter")

    harmful_matched, harmless_matched = length_match(harmful_verb, harmless_verb)

    def stats(prompts: list[str]) -> dict:
        wl = [len(p.split()) for p in prompts]
        return {
            "n": len(prompts),
            "wl_mean": sum(wl) / len(wl) if wl else 0,
            "wl_median": sorted(wl)[len(wl) // 2] if wl else 0,
            "wl_min": min(wl) if wl else 0,
            "wl_max": max(wl) if wl else 0,
            "top_verbs": Counter(first_verb(p) for p in prompts).most_common(10),
        }

    s_h = stats(harmful_matched)
    s_m = stats(harmless_matched)
    print("\n=== Final length-matched pools ===")
    print(f"{'metric':<22}{'harmful':>14}{'harmless':>14}")
    for k in ["n", "wl_mean", "wl_median", "wl_min", "wl_max"]:
        v_h, v_m = s_h[k], s_m[k]
        print(f"{k:<22}{v_h:>14.2f}{v_m:>14.2f}" if isinstance(v_h, float)
              else f"{k:<22}{v_h:>14}{v_m:>14}")
    print("\nTop verbs (harmful) :", s_h["top_verbs"])
    print("Top verbs (harmless):", s_m["top_verbs"])

    rng = random.Random(0)
    print("\n--- Sample harmful (4 random) ---")
    for p in rng.sample(harmful_matched, min(4, len(harmful_matched))):
        print(f"  - {p}")
    print("\n--- Sample harmless (4 random) ---")
    for p in rng.sample(harmless_matched, min(4, len(harmless_matched))):
        print(f"  - {p}")

    h_out = OUT_DIR / "matched_harmful.jsonl"
    m_out = OUT_DIR / "matched_harmless.jsonl"
    write_jsonl(harmful_matched, h_out)
    write_jsonl(harmless_matched, m_out)

    stats_out = OUT_DIR / "curation_stats.json"
    stats_out.write_text(json.dumps({
        "verb_set": sorted(PRODUCE_ARTIFACT_VERBS),
        "advbench_offset": SWEEP_HARMFUL_N,
        "harmful_after_verb_filter": len(harmful_verb),
        "harmless_after_verb_filter": len(harmless_verb),
        "harmful_final": s_h,
        "harmless_final": s_m,
    }, indent=2, default=list))

    print(f"\nWrote {h_out}\n      {m_out}\n      {stats_out}")


if __name__ == "__main__":
    main()
