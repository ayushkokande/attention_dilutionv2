"""Qwen3-14B baseline refusal rates on AdvBench (harmful) + Alpaca (harmless).

Uses the same stack as the rest of the ayush/ pipeline (HF transformers +
accelerate, device_map="auto" to spread weights across the 2 A100s) so the
baseline numbers come from the exact same model code that the intervention
experiments will hook into. Adds left-padded batched generation so ~1k prompts
finish in reasonable wall-clock without giving up reproducibility.

Each split's JSONL + a partial summary are written *as soon as that split
finishes*, so a walltime kill mid-harmful still leaves usable harmless data
on disk. `--splits harmless|harmful|both` lets you re-run a single split
without redoing the other; existing summary fields are merged, not clobbered.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils import (
    format_chat_prompt,
    load_harmful_prompts,
    load_harmless_prompts,
    looks_like_refusal,
    results_dir,
    strip_think_block,
)


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

SPLIT_CHOICES = ("both", "harmless", "harmful")


def slug_from_model(model: str) -> str:
    return model.split("/")[-1].lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES.keys()))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--alpaca-n", type=int, default=512)
    p.add_argument("--advbench-n", type=int, default=520)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", default="both", choices=list(SPLIT_CHOICES),
                   help="Which split(s) to run. Re-runs are safe: existing "
                        "summary fields for the other split are preserved.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Allow Qwen3 chat template <think> blocks.")
    p.add_argument("--output-dir", default=None,
                   help="Override results dir; default: results/<slug>")
    return p.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_existing_summary(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_summary(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    slug = slug_from_model(args.model)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir(slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]

    run_harmless = args.splits in ("both", "harmless")
    run_harmful = args.splits in ("both", "harmful")

    print(f"Model      : {args.model}")
    print(f"Slug       : {slug}")
    print(f"Output dir : {out_dir}")
    print(f"Splits     : {args.splits}  (harmless={run_harmless}, harmful={run_harmful})")
    print(f"dtype      : {args.dtype}  | batch_size={args.batch_size}  "
          f"max_new_tokens={args.max_new_tokens}  T={args.temperature}")
    print(f"GPUs visible: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    first_device = next(model.parameters()).device

    harmless = load_harmless_prompts(args.alpaca_n) if run_harmless else []
    harmful = load_harmful_prompts(args.advbench_n) if run_harmful else []
    if run_harmless:
        print(f"Harmless prompts (Alpaca):  {len(harmless)}")
    if run_harmful:
        print(f"Harmful prompts (AdvBench): {len(harmful)}")

    do_sample = args.temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.95

    @torch.no_grad()
    def run_split(name: str, prompts: list[str]) -> tuple[list[dict], float]:
        chat_prompts = [
            format_chat_prompt(tokenizer, p, enable_thinking=args.enable_thinking)
            for p in prompts
        ]
        rows: list[dict] = []
        refused = 0
        n_completion_tokens = 0
        t0 = time.time()
        for start, chunk_prompts in batched(chat_prompts, args.batch_size):
            inputs = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(first_device)
            out_ids = model.generate(**inputs, **gen_kwargs)
            new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for j, raw in enumerate(decoded):
                idx = start + j
                response = strip_think_block(raw)
                is_refusal = looks_like_refusal(raw)
                if is_refusal:
                    refused += 1
                comp_tok = int((new_ids[j] != tokenizer.pad_token_id).sum().item())
                n_completion_tokens += comp_tok
                rows.append({
                    "split": name,
                    "index": idx,
                    "prompt": prompts[idx],
                    "response_raw": raw,
                    "response": response,
                    "refused": is_refusal,
                    "n_prompt_tokens": int(inputs["attention_mask"][j].sum().item()),
                    "n_completion_tokens": comp_tok,
                })
            done = start + len(chunk_prompts)
            print(f"  [{name}] {done}/{len(prompts)}  cum_refused={refused}", flush=True)

        dt = time.time() - t0
        rate = refused / len(prompts) if prompts else 0.0
        tps = n_completion_tokens / dt if dt else 0.0
        print(f"[{name}] refused {refused}/{len(prompts)} = {rate:.1%}  "
              f"({dt:.1f}s, {tps:.0f} tok/s)")
        return rows, rate

    alpaca_path = out_dir / "baseline_alpaca.jsonl"
    advbench_path = out_dir / "baseline_advbench.jsonl"
    summary_path = out_dir / "baseline_summary.json"

    summary = load_existing_summary(summary_path)
    summary.update({
        "model": args.model,
        "slug": slug,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
    })

    harmless_rate = None
    harmful_rate = None

    if run_harmless:
        print("\n--- Harmless (Alpaca, empty input) ---")
        harmless_rows, harmless_rate = run_split("harmless", harmless)
        write_jsonl(alpaca_path, harmless_rows)
        summary["alpaca_n"] = len(harmless)
        summary["harmless_refusal_rate"] = harmless_rate
        summary["harmless_compliance_rate"] = 1.0 - harmless_rate
        write_summary(summary_path, summary)
        print(f"Wrote: {alpaca_path}")
        print(f"Wrote: {summary_path}  (partial)")

    if run_harmful:
        print("\n--- Harmful (AdvBench) ---")
        harmful_rows, harmful_rate = run_split("harmful", harmful)
        write_jsonl(advbench_path, harmful_rows)
        summary["advbench_n"] = len(harmful)
        summary["harmful_refusal_rate"] = harmful_rate
        write_summary(summary_path, summary)
        print(f"Wrote: {advbench_path}")
        print(f"Wrote: {summary_path}")

    print("\n=== Summary ===")
    if summary.get("harmful_refusal_rate") is not None:
        print(f"Harmful  refusal rate:    {summary['harmful_refusal_rate']:.1%}  (target: high)")
    if summary.get("harmless_refusal_rate") is not None:
        print(f"Harmless refusal rate:    {summary['harmless_refusal_rate']:.1%}  (target: ~0%)")
        print(f"Harmless compliance rate: {summary['harmless_compliance_rate']:.1%}")


if __name__ == "__main__":
    main()
