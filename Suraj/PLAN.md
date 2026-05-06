# Attention Dilution Project — Same-Day Execution Plan

## TL;DR timeline (work budget ~8–10 hrs)

| Block | Hours | Deliverable |
|---|---|---|
| Setup + model load | 0.5 | Colab A100/T4 with Qwen2.5-1.5B-Instruct loaded in `transformer_lens` |
| **Phase 1** — Find V_refusal + Guardrail Heads | 2.5 | `refusal_direction.pt`, ranked list of top-10 heads, attention pattern plots |
| **Phase 2** — Context scaling sweep | 2.5 | CSV of (context_len × head × attribution × ASR), main scaling plot |
| **Phase 3** — Activation steering intervention (stretch) | 2.0 | Rescue-rate curve: α vs. refusal recovery |
| Write-up + figures | 1.5 | 3-figure results section, ready to paste into paper |

## Model choice — pick one, not three

Use **Qwen2.5-1.5B-Instruct**. Reasons:
- Safety-tuned (refuses AdvBench prompts out-of-the-box — required for V_refusal extraction)
- Small enough to run in Colab free tier with full `transformer_lens` hooks (28 layers, 12 heads, d_model=1536)
- 32K context window → you can actually test "long" contexts without truncation artifacts
- Pythia is NOT safety-tuned — skip it, you'll have no refusal behavior to dilute
- Llama-3.2-1B-Instruct is a backup if Qwen misbehaves

## Hypothesis — revised for rigor

Original hypothesis (from proposal): *long-context jailbreaks succeed because massive benign context dilutes attention to Guardrail Heads, starving V_refusal of attribution.*

**Issue found in literature:** Zhao et al. (NeurIPS 2025) show harmfulness and refusal are encoded at *different token positions* (t_inst vs t_post-inst). Pure attention-mass dilution may not be the full story.

**Revised hypothesis (3-part):**
- **H1 (attention dilution):** As context length N grows, the fraction of attention mass that Guardrail Heads allocate from the final generation position to the harmful-request tokens decreases monotonically.
- **H2 (representational dilution):** The projection of the residual stream onto V_refusal at the harmful-request position decreases as N grows.
- **H3 (intervention rescue):** Injecting α · V_refusal at the post-refusal-computation layer restores refusal behavior, with the minimum effective α increasing with N.

This gives you **three possible positive findings** instead of one. Even if H1 is weak, H2 or H3 will yield a paper.

## What the project knowledge already gives you for free

From Jin et al. 2024 (JailbreakLens, in your project files):
- In **Llama2-7b**: L21H14 = refusal head, L26H04 = affirmation head
- In **Llama2-13b**: L31H35 = refusal head, L37H37 = affirmation head
- Instruction tuning barely moves these locations
- Use this as a **prior** for Phase 1: even if your model is Qwen, expect Guardrail Heads to live in the **middle-to-late** layers (roughly 60–80% depth), consistent with Arditi et al.

## Methodology cheat sheet

### Phase 1: Finding V_refusal (difference-of-means, not IG)

**Drop IG. Use difference-of-means (DiffMean) instead.** IG on attention heads is slow, noisy, and overkill. Arditi et al.'s method is the field standard:

```
V_refusal^(l) = mean(h^(l) | harmful prompts) - mean(h^(l) | harmless prompts)
```

computed at the last token position of the instruction, for each layer l. Pick the layer where the direction is most causally effective (ablating it maximizes non-refusal on held-out harmful prompts). This takes ~5 min of compute vs. hours for IG.

**Guardrail Heads identification:** For each head (l, h), compute its contribution to V_refusal via **direct logit attribution**: project head output at last-token position onto V_refusal at the read-off layer. Top-k heads = Guardrail Heads.

### Phase 2: Context scaling sweep

Sweep N ∈ {0, 128, 512, 1k, 2k, 4k, 8k, 16k} tokens of benign bloat (e.g., Alice-in-Wonderland text or roleplay preamble) prepended before the harmful request. For each N:
- Measure fraction of attention from the *last token* of the prompt that goes to the *harmful-request span* at each Guardrail Head → tests H1
- Measure cosine(residual_stream_at_harmful_span, V_refusal) at the refusal-computation layer → tests H2
- Measure refusal rate on a held-out AdvBench subset → behavioral outcome

Plot all three on a single x-axis (log-scale N). Crossover point = jailbreak threshold.

### Phase 3: Activation steering (same-architecture rescue)

At the layer where V_refusal is "read off" (typically ~70% depth), inject:

```
h^(l) ← h^(l) + α · V_refusal_unit
```

at every token position during inference on the diluted prompt. Sweep α ∈ {0, 1, 2, 4, 8, 16}. Report:
- Refusal rate vs α for each N
- Capability preservation on MMLU subset (sanity check — over-steering breaks the model)

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qwen refuses too weakly → V_refusal is noisy | Switch to Llama-3.2-1B-Instruct or use a stronger harmful prompt set |
| 16k context OOMs Colab | Use bfloat16 + flash attention; cap at 8k if needed — still enough for a scaling curve |
| No visible dilution effect (H1 null) | That's still a publishable finding; H2/H3 will likely show effects |
| Phase 3 over-steering breaks generation | Sweep α fine-grained in [0, 4]; use unit vector, not raw V_refusal |
| Long-context jailbreak doesn't succeed at all | Use a stronger roleplay preamble (MSJ-style), not just lorem ipsum |

## File layout

```
phase12_notebook.ipynb  — Phases 1 & 2, primary deliverable
phase3_steering.py      — Phase 3 rescue intervention
PLAN.md                 — this file
```

## Citations to include in your paper

- Arditi, A. et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717
- Jin, Z. et al. (2024). JailbreakLens. (attention head locations for refusal/affirmation)
- Wollschläger et al. (2025). The Geometry of Refusal in Large Language Models. arXiv:2502.17420 (multi-directional refusal — cite for nuance)
- Zhao, J. et al. (2025). LLMs Encode Harmfulness and Refusal Separately. NeurIPS 2025. arXiv:2507.11878
- Anthropic (2024). Many-Shot Jailbreaking.
- Liu, N.F. et al. (2024). Lost in the Middle.
- Turner, A. et al. (2023). Activation Addition.
