# Project Invariants

Single source of truth for cross-experiment knobs. Any drift here breaks numerical comparability across the 14 numbered experiments. Update in this file *before* changing scripts.

## Model

- `MODEL_NAME = "Qwen/Qwen3-14B"` (`experiment_1/utils.py:13`)
- `MODEL_SLUG = "qwen3-14b"`
- bf16 on CUDA, fp32 on MPS, fp32 on CPU (`get_device`)
- 40 transformer blocks, 40 query heads (GQA, 8 KV heads), d_model=5120, 32K RoPE context

## Chat template

- `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)` — see `experiment_1/utils.py:format_chat_prompt`.
- `enable_thinking=False` is mandatory; otherwise Qwen3 emits `<think>...</think>` blocks that pollute residual-stream activations and refusal generation. The Suraj archive validates this explicitly (`Suraj/_archive/experiment.py:819-826`).
- Post-hoc safety net: `experiment_1/utils.py:strip_think_block` removes any leading `<think>...</think>` segment that slips through.

## Refusal detector

**Canonical: `experiment_1/utils.py:looks_like_refusal`**
- 18 markers, lowercased, first 200 chars after `strip_think_block`.

The Suraj archive uses a different list (24 substrings, case-sensitive, `Suraj/_archive/experiment.py:77-84`). Differences:
- Suraj-only: `"I will not"`, `"I do not"`, `"I'm not going to"`, `"I am not going to"`, `"I must decline"`, `"I must refuse"`, `"as a language model"`, `"I refuse"`, `"It is not appropriate"`, `"Sorry"` (Ayush has `"i'm sorry"`/`"i am sorry"` only), `"I'm unable"`.
- Ayush-only: `"violates"`, `"ethical guidelines"`, `"harmful request"`, `"inappropriate"`, `"decline"`, `"refuse"`, `"cannot help"`, `"i am not able"`.

**Decision:** when porting Suraj phases into experiments 3, 10, 11, 13, 14 — call `experiment_1/utils.py:looks_like_refusal`, do not re-import `REFUSAL_SUBSTRINGS` from the archive. Cite the diff in any cell where it changes the verdict.

## Datasets

See `splits.json` for per-experiment row indices. AdvBench source is the LLM-attacks `harmful_behaviors.csv`; Alpaca is `tatsu-lab/alpaca[train]` filtered through `experiment_1/utils._harmless_instruction_is_clean` for empty-input rows only.

Two remaining overlap warnings live in `splits.json:overlap_warnings`: exp_6 and exp_7 each share AdvBench eval space with the exp_8 binary sweep (not d_hat training leaks, just shared eval). The earlier exp_6 / exp_2 Alpaca overlap was a real eval leak for the harmless_mundane 2x2 cell; closed on 2026-05-06 by migrating that cell to Alpaca[200:250) post `_harmless_instruction_is_clean` filter (commits `2c4d65f` + `9d5e0c4` + `abf87b4`).

## Refusal direction

**Canonical: layer 36** (Qwen3-14B, causal-ablation criterion under Suraj-style
24-token / 26-phrase judge, exp_2 reconcile rerun 2026-05-07).

- Step 2 result: `results/qwen3-14b/refusal_direction/meta.json` — causal_best_layer=36, canonical_layer=36, norm_best_layer=39. Six layers tied at ablated_harmful=0.0 on the 24-prompt held-out pool ({18, 26, 28, 30, 32, 36}); tiebreak (`abs(layer - norm_best_layer)`, see `experiment_2/refusal_direction.py:338-345`) picks L36 (closest to peak-norm L39).
- Selection-judge config: `causal_judge_config.judge = "suraj_substring_match"`, `max_new_tokens = 24`, `n_phrases = 26`. Stricter than the project-default behavioral judge (`experiment_1/utils.py:looks_like_refusal`, 17 phrases, 128-token window) which over-counts tail-refusal at causally-effective layers and biases the canonical pick toward early layers. The reconciliation rationale and the L36-vs-L18 numbers are recorded in commit history (`a90d2c6` / `381d05e`).
- Prior canonical values for the record: 128-token-judge causal pick was layer 18 (uniquely zero post-ablate under that judge, no tiebreak fired); norm-criterion is layer 39; old Suraj 14B-archive causal was layer 36 (matches current); old 1.7B mid-sem causal was layer 22.
- d_hat artifact: `results/qwen3-14b/refusal_direction/d_hat_best_layer.pt` (`{V_refusal, layer}`); per-layer bank: `d_hat_all_layers.pt`.
- L22 anomaly: independent of judge or pool, `d_hat[22]` fails causally (~0.7 post-ablate harmful refusal) while neighbors L20 / L24 succeed. Real layer-22 brittleness in Qwen3-14B's diff-of-means geometry; both Ayush and Suraj sweeps reproduce.

Cascade impact: layer changed 18 → 36 (reconcile rerun), so downstream consumers re-run against the new d_hat: experiments 3 (head identification, done — top heads L36H31 / L35H38 / L33H35 reproduce Suraj's circuit), 4 (matched d_hat*), 5/6/7 (validity battery; preregistered DECISION_LAYER constants in `experiment_6/eval_topic_decouple.py`, `experiment_6/orthogonalise_topic.py`, `experiment_7/eval_policy_vs_harm.py` were updated 18 → 36 — frame any paper writeup as "primary analysis at reconciled canonical L36"), 8 (binary sweep, ABLATION_LAYER), 9 (continuous projection), 10 (attention mass), 11 (steering), 13 (mitigation training data), 14 (capability cost ablation arm).

## Bloat content (context-scaling experiments only)

**Status: TBD pending step 5 decision.**

- Ayush `BENIGN_SEED_PASSAGE` (Wikipedia-style non-instructional, repeated and token-sliced) — used in exp_2 context_sweep, exp_3 projection_sweep.
- Suraj creative-writing "Willowbrook" passage — used in Suraj Phase 2.

Headline conflict: Ayush sees flat 94-98pp gap across L; Suraj sees N=128 dip to 0.77 with creative-writing prose. Likely driven by content type, not method.

**Greg ext 2 resolves this** — exp_12 sweeps content type at fixed prefix position; both becomes rows. exp_8 binary sweep canonical baseline TBD.

## N-grid (context-scaling experiments)

Merged grid for exp_8/9/10:
`{0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768}`
Adds N=128 (catches Suraj's dip) without dropping any of Ayush's existing points. H1 attention-mass measurement disabled above N=4096 (single-A100 OOM ceiling).

## Cluster

- NYU Greene, account `csci_ga_3033_131-2026sp`, partition `c24m170-a100-2`, 2x A100-80GB.
- Singularity overlay path documented in `~/.claude/.../memory/reference_cluster.md`.

## Numbered-experiment chronology

Final order (project-narrative chronology, not authorship chronology):

1. baseline (existing exp_1)
2. V_refusal extraction (existing exp_2/refusal_direction.py)
3. **NEW** — Guardrail Heads (DLA top-K, from Suraj P1)
4. matched d_hat* (existing exp_4)
5. style/vocab validity (existing exp_5)
6. topic 2x2 validity (existing exp_6)
7. policy vs harm validity (existing exp_7)
8. binary refusal vs N — split out of exp_2 (context_sweep.py + sbatch)
9. continuous projection vs N (existing exp_3)
10. **NEW** — attention-mass on Guardrail Heads vs N (Suraj P2 H1 + P6 merged)
11. **NEW** — activation steering rescue (Suraj P3)
12. **NEW** — content-type ablation (Greg ext 2)
13. **NEW** — long-context safety mitigation: LoRA SFT + DPO (Greg ext 1)
14. **NEW** — capability cost MMLU/GSM8K (Suraj P4 extended)
