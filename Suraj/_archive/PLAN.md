# Attention Dilution Project — Execution Plan

## TL;DR phase budget

| Block | Hours | Deliverable |
|---|---|---|
| Setup + model load | 0.5 | Qwen3-14B loaded in `transformer_lens` on 1× or 2× A100-80GB (NYU Greene) |
| **Phase 1** — Find V_refusal + Guardrail Heads | 2.5 | `V_refusal.pt`, ranked top-10 head list, layer-norm + heatmap figures |
| **Phase 2** — Triage + dense context-scaling sweep | 2.5 | `phase2_scaling.csv` of (N × H1 × H2 × refusal_rate), main scaling figure |
| **Phase 3** — Activation steering intervention | 2.0 | Rescue grid (α × N) and MMLU sanity sweep |
| **Phase 4** — Capability cost | 1.0 | 200 MMLU + 50 GSM8K, intact vs ablated |
| **Phase 5** — 2x2 multi-format grid | 1.5 | Refusal rate across 6 bloat formats × intact/ablated |
| **Phase 6** — Attribution maps | 1.0 | Top-K head z-hooks (no full attn_result), source-flow figure |
| Write-up + figures | 1.5 | Results section ready to paste into paper |

## Model choice — fixed

**Qwen3-14B.** Reasons:
- Safety-tuned (refuses AdvBench prompts out-of-the-box at ~94% — required for V_refusal extraction)
- 40 transformer blocks, 40 query heads (GQA, 8 KV heads), d_model=5120, 32K RoPE context
- bf16 weights ~28 GB; fits on 1× A100-80GB with `n_ctx=16000` and `max_attn_n=4096`; spreads cleanly over 2× A100-80GB for longer-N attention hooks
- 32K context window → can test "long" contexts without truncation artifacts up to the OOM ceiling on attention pattern hooks (currently N=4096)

Backup if Qwen misbehaves: a Llama-family safety-tuned model at comparable scale, used as a robustness check, not as a substitute for the headline run.

## Hypothesis

Original (from proposal): *long-context jailbreaks succeed because massive benign context dilutes attention to Guardrail Heads, starving V_refusal of attribution.*

**Issue raised by literature.** Zhao et al. (NeurIPS 2025) show harmfulness and refusal are encoded at *different token positions* (t_inst vs t_post-inst). Pure attention-mass dilution may not be the full story.

**Three-part hypothesis:**
- **H1 (attention dilution):** As context length N grows, the fraction of attention mass that Guardrail Heads allocate from the final generation position to the harmful-request tokens decreases monotonically.
- **H2 (representational dilution):** The projection of the residual stream onto V_refusal at the harmful-request position decreases as N grows.
- **H3 (intervention rescue):** Injecting α · V_refusal at the post-refusal-computation layer restores refusal behavior, with the minimum effective α increasing with N.

Three possible positive findings instead of one. Even if H1 is weak, H2 or H3 may yield the paper.

## Methodology cheat sheet

### Phase 1: Finding V_refusal (difference-of-means, not IG)

**Drop IG. Use difference-of-means (DiffMean).** IG on attention heads is slow, noisy, and overkill. Arditi et al.'s method is the field standard:

```
V_refusal^(l) = mean(h^(l) | harmful prompts) − mean(h^(l) | harmless prompts)
```

computed at the last token position of the instruction, for each layer l. Pick the layer where the direction is most causally effective (ablating it maximizes non-refusal on held-out harmful prompts). For Qwen3-14B this lands at layer 24 (out of 40, ~60% depth).

**Guardrail Heads identification:** For each head (l, h), compute its contribution to V_refusal via **direct logit attribution**: project head output at last-token position onto V_refusal at the read-off layer. Top-k heads = Guardrail Heads. Top-1 in this run is L22H7.

### Phase 2: Context scaling sweep

Sweep N ∈ {0, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096} tokens of benign bloat (creative-writing passage by default; alternative formats — prefix/suffix/sandwich/multi_turn/distractor/many_shot — covered in Phase 5) prepended before the harmful request. For each N:
- Measure fraction of attention from the *last token* of the prompt that goes to the *harmful-request span* at each Guardrail Head → tests H1
- Measure cosine(residual_stream_at_harmful_span, V_refusal) at the refusal-computation layer → tests H2
- Measure refusal rate on a held-out AdvBench subset → behavioral outcome

Plot all three on a single x-axis (log-scale N). Crossover point = jailbreak threshold (none observed yet — refusal never crosses 0.5×baseline within sweep).

### Phase 3: Activation steering (same-architecture rescue)

At the layer where V_refusal is "read off" (~60% depth = layer 24 for Qwen3-14B), inject:

```
h^(l) ← h^(l) + α · V_refusal_unit
```

at every token position during inference on the diluted prompt. Sweep α ∈ {0, 1, 2, 4, 8, 16}. Report:
- Refusal rate vs α for each N (`phase3_rescue_grid.csv`)
- Capability preservation on a 50-item MMLU subset (`phase3_mmlu.csv`) — over-steering breaks the model

### Phase 4: Capability cost

Independent of steering, compare intact-model vs d̂-ablated-model on 200 MMLU + 50 GSM8K items at every N in the dense sweep. Validates that ablating V_refusal does not silently break general capability.

### Phase 5: Multi-format 2×2 grid

Repeat the (intact, ablated) × (harmful, harmless) refusal-rate measurement across all six bloat formats (prefix / suffix / sandwich / multi_turn / distractor / many_shot) at the dense N grid. Tests format-dependence of the dilution effect.

### Phase 6: Attribution maps

For the top-K Guardrail Heads from Phase 1, run selective z-hooks (not full `attn_result`, which OOMs at long N) at a coarse N grid. Output: `phase6_top_head_attn_fraction.csv` plus source-flow figure.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qwen3-14B refuses too weakly → V_refusal is noisy | Empirically 94% refusal at N=0; well-conditioned. If a future run drops, switch to a Llama-family safety-tuned model |
| Attention-pattern hooks OOM above N≈4k | `max_attn_n=4096`; report status='OOM_measure' rows in CSVs; H1 measurement disabled above 4k by design |
| No visible dilution effect at long N (H1 weak) | Phase 2 already shows non-monotonic behavioral curve — write up the recovery as a finding, not a failure |
| Phase 3 over-steering breaks generation | Sweep α fine-grained in [0, 4] first; use unit vector, not raw V_refusal; MMLU sanity in Phase 3 catches over-steer |
| Long-context jailbreak doesn't succeed at all | Use a stronger preamble (MSJ-style many-shot, in Phase 5) and report negative result honestly |

## File layout

```
experiment.py            — single-file pipeline; all six phases + packaging
PLAN.md                  — this file
README.md                — paper-style writeup of method, results, limitations
results_v2/              — CSVs and figures per phase
```

## Citations to include in the paper

- Arditi, A. et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717
- Jin, Z. et al. (2024). JailbreakLens. (attention head locations for refusal/affirmation)
- Wollschläger et al. (2025). The Geometry of Refusal in Large Language Models. arXiv:2502.17420 (multi-directional refusal — cite for nuance)
- Zhao, J. et al. (2025). LLMs Encode Harmfulness and Refusal Separately. NeurIPS 2025. arXiv:2507.11878
- Anthropic (2024). Many-Shot Jailbreaking.
- Liu, N.F. et al. (2024). Lost in the Middle.
- Turner, A. et al. (2023). Activation Addition.
