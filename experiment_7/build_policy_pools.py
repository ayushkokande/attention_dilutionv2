"""Build the 3 prompt pools for experiment 7 (policy-refusal vs harm-refusal validity).

Pools (50 prompts each, 150 total):
  harm_refusal.jsonl    label_harm=1 label_policy=0   AdvBench[50:100] held-out
  policy_refusal.jsonl  label_harm=0 label_policy=1   SORRY-Bench (non-harm subset)
  harmless.jsonl        label_harm=0 label_policy=0   Alpaca[100:150]

Held-out indices avoid overlap with experiment_2's refusal_direction
construction (AdvBench[100:150]) and experiment_6's harmful_edgy pool
(AdvBench[0:50]) / harmless_mundane pool (Alpaca[0:50]).

The policy pool is drawn from SORRY-Bench (sorry-bench/sorry-bench-202406),
restricted to non-harm policy-refusal categories (legal/medical/financial/etc.).
Use --list-categories first to inspect the actual category strings on HF; pass
the desired ones via --categories. Only `prompt_style == "base"` rows are kept
(no adversarial paraphrases), to keep the policy axis clean.

After this script, run preflight_refusal_rates.py. If a category's refusal
rate < gate, drop it via --categories and rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

OUT_HARM = DATA / "harm_refusal.jsonl"
OUT_POLICY = DATA / "policy_refusal.jsonl"
OUT_HARMLESS = DATA / "harmless.jsonl"

ADV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

CELL_SIZE = 50
ADV_START_DEFAULT = 50
ALPACA_START_DEFAULT = 100

SORRY_BENCH_HF = "sorry-bench/sorry-bench-202406"

# Default non-harm policy categories. These are substring-matched against the
# `category` column of SORRY-Bench (case-insensitive) so minor formatting
# differences in the upstream labels don't break the pipeline. Verify with
# --list-categories before trusting these defaults.
DEFAULT_CATEGORIES = [
    "legal",
    "medical",
    "financial",
    "governance",
    "dangerous machinery",
]

# Per-category target. 5 cats x 10 = 50. Override via --per-category.
PER_CATEGORY_DEFAULT = 10


def fetch_advbench_rows(start: int, n: int) -> list[tuple[int, str]]:
    raw = urllib.request.urlopen(ADV_URL).read().decode("utf-8").splitlines()
    rdr = csv.reader(raw)
    next(rdr)
    rows: list[tuple[int, str]] = []
    for file_row_idx, parts in enumerate(rdr):
        if file_row_idx < start:
            continue
        if len(rows) >= n:
            break
        rows.append((file_row_idx, parts[0].strip()))
    return rows


def fetch_alpaca_rows(start: int, n: int) -> list[tuple[int, str]]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out: list[tuple[int, str]] = []
    skipped = 0
    for i, row in enumerate(ds):
        instr = (row.get("instruction") or "").strip()
        if not instr:
            continue
        if skipped < start:
            skipped += 1
            continue
        out.append((i, instr))
        if len(out) >= n:
            break
    return out


def fetch_sorrybench() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(SORRY_BENCH_HF, split="train")
    out: list[dict] = []
    for row in ds:
        prompt_style = str(row.get("prompt_style", "")).strip().lower()
        if prompt_style and prompt_style != "base":
            continue
        turns = row.get("turns")
        if isinstance(turns, list) and turns:
            prompt = str(turns[0]).strip()
        else:
            prompt = str(row.get("prompt") or row.get("question") or "").strip()
        if not prompt:
            continue
        out.append({
            "question_id": row.get("question_id"),
            "category": str(row.get("category", "")).strip(),
            "prompt": prompt,
        })
    return out


def list_categories(rows: list[dict]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["category"]] += 1
    width = max((len(c) for c in counts), default=0)
    print(f"{'category'.ljust(width)}  count")
    print(f"{'-' * width}  -----")
    for cat, n in sorted(counts.items()):
        print(f"{cat.ljust(width)}  {n}")
    print(f"\nTotal categories: {len(counts)}  Total base prompts: {sum(counts.values())}")


def select_policy_pool(
    sb_rows: list[dict],
    category_filters: list[str],
    per_category: int,
) -> list[dict]:
    """Pick `per_category` deterministic rows per matched category."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in sb_rows:
        cat_lower = r["category"].lower()
        for needle in category_filters:
            if needle.lower() in cat_lower:
                by_cat[needle].append(r)
                break

    selected: list[dict] = []
    missing: list[tuple[str, int]] = []
    for needle in category_filters:
        bucket = sorted(
            by_cat[needle],
            key=lambda r: (r.get("question_id") if r.get("question_id") is not None else r["prompt"]),
        )
        if len(bucket) < per_category:
            missing.append((needle, len(bucket)))
        for r in bucket[:per_category]:
            selected.append({
                "pool": "policy_refusal",
                "source": "sorry-bench-202406",
                "sorry_bench_id": r.get("question_id"),
                "category_raw": r["category"],
                "category": needle,
                "prompt": r["prompt"],
                "label_harm": 0,
                "label_policy": 1,
            })

    if missing:
        msg = ", ".join(f"{c}: {n}" for c, n in missing)
        sys.stderr.write(
            f"ERROR: not enough rows for categories ({msg}); "
            f"each needs >= {per_category}. Drop the category or lower --per-category.\n"
        )
        sys.exit(1)
    return selected


def assert_disjoint_from(
    policy_rows: list[dict],
    other_pools: list[list[dict]],
) -> None:
    """Hash-check policy prompts don't collide with harm/harmless prompts."""
    other_hashes = set()
    for pool in other_pools:
        for r in pool:
            other_hashes.add(hashlib.sha1(r["prompt"].encode("utf-8")).hexdigest())
    collisions = []
    for r in policy_rows:
        h = hashlib.sha1(r["prompt"].encode("utf-8")).hexdigest()
        if h in other_hashes:
            collisions.append(r["sorry_bench_id"])
    if collisions:
        sys.stderr.write(
            f"ERROR: {len(collisions)} policy prompts collide with harm/harmless pools "
            f"(sorry_bench_ids: {collisions[:5]}...). Investigate.\n"
        )
        sys.exit(1)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--list-categories",
        action="store_true",
        help="Print SORRY-Bench category counts and exit (no files written).",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Category substring filters (case-insensitive). Each match contributes "
             "--per-category rows to the policy pool.",
    )
    p.add_argument("--per-category", type=int, default=PER_CATEGORY_DEFAULT)
    p.add_argument("--advbench-start", type=int, default=ADV_START_DEFAULT)
    p.add_argument("--alpaca-start", type=int, default=ALPACA_START_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"Fetching SORRY-Bench ({SORRY_BENCH_HF})...")
    sb_rows = fetch_sorrybench()
    print(f"  Got {len(sb_rows)} base-style rows.")

    if args.list_categories:
        list_categories(sb_rows)
        return

    expected_total = args.per_category * len(args.categories)
    if expected_total != CELL_SIZE:
        sys.stderr.write(
            f"WARN: per_category({args.per_category}) * categories({len(args.categories)}) "
            f"= {expected_total}, not {CELL_SIZE}. Pool size will be {expected_total}.\n"
        )

    print(f"Fetching AdvBench[{args.advbench_start}:{args.advbench_start + CELL_SIZE}]...")
    adv_rows = fetch_advbench_rows(args.advbench_start, CELL_SIZE)
    harm_rows = [
        {
            "pool": "harm_refusal",
            "source": "advbench",
            "advbench_idx": idx,
            "prompt": text,
            "label_harm": 1,
            "label_policy": 0,
        }
        for idx, text in adv_rows
    ]
    write_jsonl(OUT_HARM, harm_rows)
    print(f"Wrote {OUT_HARM} ({len(harm_rows)} rows)")

    print(f"Fetching Alpaca[{args.alpaca_start}:{args.alpaca_start + CELL_SIZE}]...")
    alp = fetch_alpaca_rows(args.alpaca_start, CELL_SIZE)
    harmless_rows = [
        {
            "pool": "harmless",
            "source": "alpaca",
            "alpaca_idx": idx,
            "prompt": text,
            "label_harm": 0,
            "label_policy": 0,
        }
        for idx, text in alp
    ]
    write_jsonl(OUT_HARMLESS, harmless_rows)
    print(f"Wrote {OUT_HARMLESS} ({len(harmless_rows)} rows)")

    print(f"Selecting policy pool: {args.categories} x {args.per_category}")
    policy_rows = select_policy_pool(sb_rows, args.categories, args.per_category)
    assert_disjoint_from(policy_rows, [harm_rows, harmless_rows])
    write_jsonl(OUT_POLICY, policy_rows)
    print(f"Wrote {OUT_POLICY} ({len(policy_rows)} rows)")

    by_cat: dict[str, int] = defaultdict(int)
    for r in policy_rows:
        by_cat[r["category"]] += 1

    print("\nPool counts:")
    print(f"  harm_refusal:   {len(harm_rows)}")
    print(f"  policy_refusal: {len(policy_rows)}  (by category: {dict(by_cat)})")
    print(f"  harmless:       {len(harmless_rows)}")


if __name__ == "__main__":
    main()
