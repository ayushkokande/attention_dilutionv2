"""Build the 3 prompt pools for experiment 7 (policy-refusal vs harm-refusal validity).

Pools (50 prompts each, 150 total):
  harm_refusal.jsonl    label_harm=1 label_policy=0   AdvBench[50:100] held-out
  policy_refusal.jsonl  label_harm=0 label_policy=1   manual_policy_refusal.json
                                                      (categories: medical, legal,
                                                       copyright, identity, brand)
  harmless.jsonl        label_harm=0 label_policy=0   Alpaca[100:150]

The manual file is a JSON list of {index, category, prompt}. Empty `prompt`
fields must be filled before this script will succeed (use --allow-partial
to emit only filled rows for an early dry run).

Held-out indices avoid overlap with experiment_2's refusal_direction
construction (AdvBench[100:150]) and experiment_6's harmful_edgy pool
(AdvBench[0:50]) / harmless_mundane pool (Alpaca[0:50]).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

MANUAL_POLICY = DATA / "manual_policy_refusal.json"

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


def load_manual(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected JSON list")
    out = []
    for row in raw:
        idx = int(row["index"])
        category = str(row.get("category", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        out.append({"index": idx, "category": category, "prompt": prompt})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Skip empty manual rows instead of failing (dry-run mode).",
    )
    p.add_argument("--advbench-start", type=int, default=ADV_START_DEFAULT)
    p.add_argument("--alpaca-start", type=int, default=ALPACA_START_DEFAULT)
    return p.parse_args()


def build_policy_pool(path: Path, allow_partial: bool) -> list[dict]:
    rows_in = load_manual(path)
    out: list[dict] = []
    empty: list[int] = []
    for r in rows_in:
        if not r["prompt"]:
            empty.append(r["index"])
            continue
        out.append({
            "pool": "policy_refusal",
            "index": r["index"],
            "category": r["category"],
            "prompt": r["prompt"],
            "label_harm": 0,
            "label_policy": 1,
        })

    if empty and not allow_partial:
        sys.stderr.write(
            f"{path}: {len(empty)} rows missing non-empty `prompt` "
            f"(first: {empty[:5]}). Fill them or rerun with --allow-partial.\n"
        )
        sys.exit(1)
    if empty and allow_partial:
        sys.stderr.write(
            f"WARN {path}: skipping {len(empty)} empty rows (--allow-partial).\n"
        )

    if not allow_partial and len(out) != CELL_SIZE:
        sys.stderr.write(
            f"{path}: expected {CELL_SIZE} filled rows, got {len(out)}.\n"
        )
        sys.exit(1)
    return out


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"Fetching AdvBench[{args.advbench_start}:{args.advbench_start + CELL_SIZE}]...")
    adv_rows = fetch_advbench_rows(args.advbench_start, CELL_SIZE)
    harm_rows = [
        {
            "pool": "harm_refusal",
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
            "alpaca_idx": idx,
            "prompt": text,
            "label_harm": 0,
            "label_policy": 0,
        }
        for idx, text in alp
    ]
    write_jsonl(OUT_HARMLESS, harmless_rows)
    print(f"Wrote {OUT_HARMLESS} ({len(harmless_rows)} rows)")

    policy_rows = build_policy_pool(MANUAL_POLICY, args.allow_partial)
    write_jsonl(OUT_POLICY, policy_rows)
    print(f"Wrote {OUT_POLICY} ({len(policy_rows)} rows)")

    by_cat: dict[str, int] = {}
    for r in policy_rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1

    print("\nPool counts:")
    print(f"  harm_refusal:   {len(harm_rows)}")
    print(f"  policy_refusal: {len(policy_rows)}  (by category: {by_cat})")
    print(f"  harmless:       {len(harmless_rows)}")


if __name__ == "__main__":
    main()
