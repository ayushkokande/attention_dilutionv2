"""Derive a data-driven “edgy vocabulary” from AdvBench vs Alpaca (Monroe-style log-odds).

Writes experiment_5/data/edgy_lexicon.json — used by build_styled_pools.py and eval_validity.py.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "edgy_lexicon.json"

STOPWORDS = frozenset(
    "a an the to of and or for in on at by from with as is are was were be been being "
    "it its this that these those i you we they he she his her their them my our your "
    "do does did doing done have has had having can could should would will shall may "
    "not no yes how what when where why who which into any some more most less very "
    "than then just only also into about over such".split()
)


def tokenize(text: str) -> list[str]:
    return [
        m.group().lower()
        for m in re.finditer(r"[a-zA-Z]+(?:'[a-z]+)?", text.lower())
        if len(m.group()) >= 2
    ]


def load_advbench_prompts(n: int) -> list[str]:
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_behaviors.csv"
    )
    rows = urllib.request.urlopen(url).read().decode("utf-8").strip().splitlines()
    prompts = []
    for line in rows[1 : n + 1]:
        prompts.append(line.split(",", 1)[0].strip().strip('"'))
    return prompts


def load_alpaca_prompts(target_n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in ds:
        inp = (row.get("instruction") or "").strip()
        if inp:
            prompts.append(inp)
        if len(prompts) >= target_n:
            break
    return prompts[:target_n]


def main() -> None:
    print("Loading AdvBench[:100]...")
    adv = load_advbench_prompts(100)
    print(f"  got {len(adv)} prompts")

    print("Loading Alpaca harmless (target 100)...")
    alp = load_alpaca_prompts(100)
    print(f"  got {len(alp)} prompts")

    ca = Counter()
    cb = Counter()
    for p in adv:
        for w in tokenize(p):
            if w not in STOPWORDS:
                ca[w] += 1
    for p in alp:
        for w in tokenize(p):
            if w not in STOPWORDS:
                cb[w] += 1

    vocab = sorted(set(ca) | set(cb))
    alpha = 0.01  # Dirichlet smoothing per bucket (simple scalar)

    scores = []
    for w in vocab:
        a = ca[w] + alpha
        b = cb[w] + alpha
        na = sum(ca.values()) + alpha * len(vocab)
        nb = sum(cb.values()) + alpha * len(vocab)
        odds = math.log(a / na) - math.log(b / nb)
        scores.append((odds, w))

    scores.sort(reverse=True)
    top_k = [w for _, w in scores[:50]]

    payload = {
        "tokens": top_k,
        "details": [
            {"token": w, "log_odds_adv_over_alp": s, "adv_count": ca[w], "alp_count": cb[w]}
            for s, w in scores[:50]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT} — top tokens: {top_k[:15]}...")


if __name__ == "__main__":
    main()
