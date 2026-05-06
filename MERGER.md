# Suraj × Ayush merger — progress and plan

Living doc for `reconcile-experiments-suraj` branch. Tracks how Suraj's
six-phase archive (`Suraj/_archive/`) folds into Ayush's numbered
`experiment_*/` chronology, what has run, what's pending.

Authoritative cross-experiment knobs live in `INVARIANTS.md`. This file
references but never duplicates them — if constant moves, edit `INVARIANTS.md`,
not here.

---

## Onboarding for Suraj — why this merger exists

**Background.** Mid-semester check-in (`submission_408271332.pdf`) was
three-author (Suraj + Ayush + Aria) on old repo `attention_dilution/` running
**Qwen3-1.7B**. After mid-sem, Ayush forked `attention_dilutionv2/` solo on
**Qwen3-14B** with bigger samples (n=100 per cell), reran Phase 1 + Phase 2
binary sweep, and added validity work the prof flagged. Your archive
(`Suraj/_archive/experiment.py`, six phases, `results_v2/`) is the 14B work
you ran independently around same time — it never got integrated into v2's
numbered layout.

**Why merge now.** Two reasons.

1. **Final paper needs one chronology.** v2 grew bottom-up: exp_1 baseline,
   exp_2 d_hat, then validity battery (exp_4/5/6/7) bolted on after prof's
   feedback. Your archive has Guardrail Heads, attn-mass measurement, steering
   rescue, capability cost, attribution maps — everything v2 lacks for the
   mechanistic story. Without merging, paper is two disconnected halves.
2. **Headline conflict needs resolution.** Your P2 sweep on creative-writing
   bloat shows N=128 dip → 0.77 → recovery. Ayush's exp_8 sweep on
   Wikipedia-style bloat shows flat 94–98pp gap across L=0..32k.
   Same model, opposite shape. Can't ship paper claiming dilution behavior
   while two arms of own data disagree. Resolution = exp_12 (see Greg below).

**What changed under your archive while you were away.**

- **Canonical refusal layer 20 → 18.** Your archive used L24 (norm criterion
  on 14B). Ayush's exp_2 first reran at L20 (norm). Step-2 causal-ablation
  rerun this week landed L18 — at L18, ablation drives harmful refusal to
  0.00 on 24-prompt held-out pool (`refusal_direction/meta.json`). L18 is
  now canonical. Every downstream consumer (exp_3/5/6/7/8/9 + your phases)
  must use L18 d_hat.
- **Refusal detector divergence.** Your archive uses 24 substrings,
  case-sensitive. v2 canonical is `experiment_1/utils.py:looks_like_refusal`
  — 18 markers, lowercased, first 200 chars after `strip_think_block`.
  Diff documented in `INVARIANTS.md` §Refusal detector. **Use Ayush's, not
  yours, in any ported code.**
- **Disjoint splits enforced via `splits.json`.** Your phases used ad-hoc
  ranges; v2 has central row-index registry per experiment with three known
  overlap warnings flagged. Ports must register their slice there.

---

## Prof's mid-sem feedback (Ayush addressed solo, baked into exp_4–7)

Prof: *"AdvBench and Alpaca prompts look quite different and so this
direction doesn't necessarily reflect harmfulness but instead reflects more
general domain shift."*

Three-experiment validity battery built in response, all share preregistered
decision rules:

- **exp 4 — matched d_hat\*.** Verb-class- and word-length-matched
  harmful/harmless pools, re-extract d_hat\*, compute per-layer
  cosine(d_hat, d_hat\*). Decision: ≥0.90 = clean, ≤0.70 = contaminated
  (adopt d_hat\*), else escalate. Result: `cos_canonical=0.971` at L20 → CLEAN.
  Needs recheck at L18.
- **exp 6 — topic 2x2 ANOVA.** intent × topic factorial, partial η²(intent)
  must be ≥ 0.5 and η²(topic) ≤ 0.1. Result at L20: η²(intent)=0.894,
  η²(topic)=0.089 → CLEAN. Needs rerun at L18.
- **exp 7 — policy-vs-harm AUC.** d_hat must distinguish harm-refusal from
  policy-refusal (model declines for compliance reasons, not safety).
  SORRY-Bench used as policy-refusal pool. Decision: AUC(harm vs policy) ≥ 0.85
  = harm-specific. Result at L20: 0.926 → HARM_SPECIFIC. Needs rerun at L18.

These three together answer prof: d_hat tracks harm intent, not domain
shift, not topic, not policy-decline.

---

## Greg's feedback (advisor) — sources of new exp_12 and exp_13

Greg is project advisor. Two extensions added to plan after he reviewed
mid-sem report and v2 progress:

### Greg ext 1 → **exp 13** (long-context safety mitigation: LoRA SFT + DPO)

**Why.** Mechanistic story (exp 8 → 11) only diagnoses the failure mode.
Paper without mitigation is half a paper. Greg pushed for fix-side experiment:
can we recover refusal at long N via lightweight finetuning, without breaking
capability?

**Plan (scoping pending).** Build long-context refusal training data
(harmful instruction wrapped in benign N-token prefix, paired with refusal
completions). LoRA SFT first pass; DPO second pass with non-refusal as
negative. Eval against held-out long-N AdvBench split + capability check
(piggybacks on exp 14 MMLU/GSM8K).

### Greg ext 2 → **exp 12** (content-type ablation)

**Why.** Resolves the Suraj-vs-Ayush headline conflict. Greg pointed out
that bloat *content type* (Wikipedia exposition vs. creative-writing prose
vs. code vs. multi-turn vs. many-shot) is a confounded variable across our
two sweeps. Without controlling for it, can't say whether the N=128 dip
is real or template-driven.

**Plan.** Fixed prefix position, fixed N grid, vary content type as rows:
(Ayush-Wiki, Suraj-Willowbrook-creative, code, multi-turn, distractor,
many-shot). Same harmful split, same detector. Output decides exp_8
canonical baseline content (currently TBD in `INVARIANTS.md` §Bloat content)
and either confirms N=128 dip is a creative-writing artefact or promotes it
to a real finding worth chasing.

This subsumes your old Phase 5 multi-format 2×2 — same idea, cleaner factor
design.

---

## TL;DR status

- Canonical refusal layer changed **20 → 18** after step-2 causal-ablation rerun
  (`results/qwen3-14b/refusal_direction/meta.json`: `causal_best_layer=18`,
  `norm_best_layer=39`). All sbatches (exp_3/5/6/7) updated to `LAYERS=18 …`
  on commit `1b1138b`. **Existing exp_6/7 result JSONs are still stamped at
  layer 20** — those evals need a rerun against the new d_hat.
- Final 14-experiment chronology defined in `INVARIANTS.md` §Numbered-experiment
  chronology. Steps 0–2 of the merge are landed (commits `01d2a2e` →
  `bb84478` → `1b1138b` → `a80fb87`).
- Suraj phases 1, 2, 3, 4, 5, 6 will become experiments 3, 8/10, 11, 14, 12,
  10 respectively (see mapping table below). His `experiment.py` and Phase
  CSVs are frozen in `Suraj/_archive/` and serve as a reference oracle, not
  a build target.

## Experiment status table

| # | Experiment | Source | d_hat layer in artefact | Status | Result path |
|---|---|---|---|---|---|
| 1 | baseline refusal rates | Ayush | n/a | done | `results/qwen3-14b/baseline_summary.json` (harmful 0.937, harmless 0.014) |
| 2 | V_refusal extraction (DiffMean) | Ayush | **18** (causal) | **done — rerun** | `results/qwen3-14b/refusal_direction/{meta.json, d_hat_all_layers.pt}` |
| 3 | Guardrail Heads (DLA top-K) | NEW (Suraj P1 port) | — | **pending** | (planned: `results/qwen3-14b/guardrail_heads/`) |
| 4 | matched d_hat* | Ayush | 20 | done; `cos_canonical=0.971`, "CLEAN" verdict — but verdict computed pre-cascade; recheck at layer 18 | `results/qwen3-14b/refusal_direction_matched/` |
| 5 | style/vocab validity | Ayush | sbatch=18 | **not yet run** (validity dir empty) | `results/qwen3-14b/validity/` (empty) |
| 6 | topic 2x2 ANOVA | Ayush | result stamped 20 | **stale — needs rerun at L18** | `results/qwen3-14b/topic_decouple/anova.json` |
| 7 | policy vs harm AUC | Ayush | result stamped 20 | **stale — needs rerun at L18** | `results/qwen3-14b/policy_validity/aucs.json` |
| 8 | binary refusal vs N | Ayush (split from old exp_2) | ablation_layer=20 | **stale — needs rerun at L18** | `results/qwen3-14b/context_sweep/summary.json` |
| 9 | continuous projection vs N | Ayush exp_3 | layers=[20,24,28] | **stale — needs rerun at L18** | `results/qwen3-14b/projection_sweep.{json,png}` |
| 10 | attention-mass on Guardrail Heads vs N | NEW (Suraj P2 H1 + P6 merged) | — | pending — depends on exp 3 | — |
| 11 | activation steering rescue | NEW (Suraj P3) | — | pending — depends on exp 2,9 | — |
| 12 | content-type ablation (Greg ext 2) | NEW | — | pending | — |
| 13 | long-context safety mitigation (LoRA SFT + DPO, Greg ext 1) | NEW | — | pending | — |
| 14 | capability cost MMLU/GSM8K (Suraj P4 extended) | NEW | — | pending | — |

## Per-experiment detail

Each entry: **goal · inputs · method · output · decision rule · deps · owner**. Scope frozen — this section enriches descriptions, does not add or remove experiments. Cross-ref `INVARIANTS.md` for shared knobs.

### exp 1 — baseline refusal rates (DONE)

- **Goal.** Establish unconditioned refusal floor on intact Qwen3-14B before any intervention. Sanity-check that refusal detector and chat template behave on the two pools.
- **Inputs.** AdvBench harmful split (`splits.json` exp_1 indices) + Alpaca harmless split. n per pool per `experiment_1/baseline_benchmark.py`.
- **Method.** Greedy decode w/ `enable_thinking=False`, `strip_think_block`, `looks_like_refusal` on first 200 chars. No hooks, no ablation.
- **Output.** `results/qwen3-14b/baseline_summary.json` — harmful refusal 0.937, harmless refusal 0.014.
- **Decision rule.** None — descriptive baseline. Used as "intact" reference for exp 8/14.
- **Deps.** None.
- **Owner.** Ayush. Reused as-is.

### exp 2 — V_refusal extraction (DONE, post-cascade)

- **Goal.** Extract single direction `d_hat` in residual stream that mediates refusal.
- **Inputs.** AdvBench[100:200) harmful + Alpaca[0:100) harmless (training pool — disjoint from exp 1 baseline, NOT disjoint from exp 6 eval — see overlap warning).
- **Method.** DiffMean on last-token residual at every layer (0..39). For each layer compute (a) norm criterion, (b) causal-ablation criterion: project off `d_hat_l` from harmful prompts and measure refusal drop on 24-prompt held-out pool.
- **Output.** `results/qwen3-14b/refusal_direction/{meta.json, d_hat_best_layer.pt, d_hat_all_layers.pt}`. `causal_best_layer=18`, `norm_best_layer=39`. At L18, ablation drives ablated_harmful=0.00, ablated_harmless=0.00.
- **Decision rule.** Pick layer minimising post-ablation harmful refusal subject to harmless refusal staying near 0. → L18.
- **Deps.** None upstream; downstream feeds exp 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14.
- **Owner.** Ayush. Suraj P1 V_refusal section is the methodological reference but Ayush's rerun is canonical.

### exp 3 — Guardrail Heads via DLA (NEW — Suraj P1 port, PENDING)

- **Goal.** Identify the small subset of attention heads whose direct logit attribution onto `d_hat` is largest — the "guardrail heads" hypothesised to write refusal into the residual.
- **Inputs.** L18 `d_hat` from exp 2; held-out harmful pool (TBD slice in `splits.json`).
- **Method.** Port DLA section of `Suraj/_archive/experiment.py:phase1` (~lines 930–1229). For each (layer, head): compute head's per-token output written to residual, dot with `d_hat`, average over harmful prompts. Rank heads. Suraj's L24 top-1 was L22H7 — provisional, must redo at L18.
- **Output.** `results/qwen3-14b/guardrail_heads/` — ranked CSV (layer, head, dla_score), top-K head set (K TBD, likely 8–16), heatmap fig.
- **Decision rule.** Top-K cutoff by elbow on cumulative DLA mass (Suraj convention). Cite Suraj P1 ranking diff as a robustness check.
- **Deps.** exp 2 (L18 d_hat). Blocks exp 10 (needs head set), exp 11 (target layer for steering).
- **Owner.** NEW. Port code into `experiment_3/guardrail_heads.py` (do NOT clone `Suraj/_archive/experiment.py` wholesale — reuse `experiment_1/utils.py` helpers, drop Suraj's REFUSAL_SUBSTRINGS, use `looks_like_refusal`).

### exp 4 — matched d_hat\* (DONE; RECHECK at L18)

- **Goal.** Falsify "d_hat is just verb-class / length artefact" by re-extracting on verb-matched, length-matched harmful/harmless pools and comparing to canonical d_hat.
- **Inputs.** Curated matched pools (`experiment_4/data/`, built by `curate_matched_pools.py`).
- **Method.** Repeat exp 2 DiffMean on matched pools → `d_hat*`. Per-layer cosine(d_hat, d_hat\*).
- **Output.** `results/qwen3-14b/refusal_direction_matched/`. Result at L20: `cos_canonical=0.971` → CLEAN. **Recheck needed at L18.**
- **Decision rule (preregistered).** ≥0.90 = clean (keep canonical); ≤0.70 = contaminated (adopt d_hat\*); else escalate.
- **Deps.** exp 2.
- **Owner.** Ayush.

### exp 5 — style/vocab validity (NOT YET RUN)

- **Goal.** Falsify "d_hat tracks edgy lexicon / register, not harm intent" by varying surface style at fixed intent.
- **Inputs.** Styled pools built by `experiment_5/build_styled_pools.py` + `build_edgy_lexicon.py` (Edgy/Polite × harmful/harmless).
- **Method.** Project each style cell onto L18 d_hat; compare projections within harmful (Edgy vs Polite) and within harmless.
- **Output.** `results/qwen3-14b/validity/` (currently empty).
- **Decision rule.** Within-intent style effect must be small relative to intent effect (specific η² thresholds analogous to exp 6 — pin in script).
- **Deps.** exp 2.
- **Owner.** Ayush. sbatch already targets L18.

### exp 6 — topic 2x2 ANOVA (STALE — RERUN at L18)

- **Goal.** Falsify "d_hat is domain shift between AdvBench and Alpaca." Confirms intent dominates topic.
- **Inputs.** intent × topic factorial pools (`experiment_6/data/`): {harmful_violent, harmful_mundane, harmless_violent, harmless_mundane}.
- **Method.** 2x2 ANOVA on projection-onto-d_hat. Partial η²(intent) and η²(topic).
- **Output.** `results/qwen3-14b/topic_decouple/anova.json`. L20 result: η²(intent)=0.894, η²(topic)=0.089 → CLEAN. Stale at L18.
- **Decision rule (preregistered).** η²(intent) ≥ 0.5 AND η²(topic) ≤ 0.1.
- **Deps.** exp 2. **Caveat:** harmless_mundane Alpaca[0:50) overlaps exp 2 d_hat training Alpaca[0:100) — eval leak, must flag in paper or rerun w/ disjoint slice.
- **Owner.** Ayush.

### exp 7 — policy vs harm AUC (STALE — RERUN at L18)

- **Goal.** Falsify "d_hat is a generic compliance/decline direction." Show it discriminates harm-refusal from policy-refusal.
- **Inputs.** Harm-refusal pool (AdvBench[50:100) — declines for safety) + policy-refusal pool (SORRY-Bench, model declines for compliance/format/identity reasons, not harm).
- **Method.** Project both pools onto L18 d_hat; compute ROC-AUC of harm vs policy.
- **Output.** `results/qwen3-14b/policy_validity/aucs.json`. L20 result: 0.926 → HARM_SPECIFIC. Stale at L18.
- **Decision rule (preregistered).** AUC ≥ 0.85 = harm-specific.
- **Deps.** exp 2. Same shared-eval caveat as exp 6 (AdvBench[50:100) ⊂ exp 8 eval space).
- **Owner.** Ayush.

### exp 8 — binary refusal vs N (STALE — RERUN at L18, FILL N grid)

- **Goal.** Headline phenomenon: does refusal rate degrade as benign context is prepended? Both intact and ablated arms.
- **Inputs.** Harmful prompt + length-N benign bloat prefix; intact model vs L18 ablation.
- **Method.** For each N in `{0,128,512,1024,2048,4096,8192,16384,32768}`, generate completions for harmful prompts with N-token bloat prefix, measure refusal rate w/ `looks_like_refusal`. Two arms: intact, ablated (project off d_hat at L18 across all 40 layers).
- **Output.** `results/qwen3-14b/context_sweep/summary.json`. Current file stamped layer 20 and missing N=128, 1024, 4096.
- **Decision rule.** Falsifiability check: ablated arm should stay flat-low across N (sanity that ablation does what it claims). Intact arm shape is the result, not a pass/fail.
- **Deps.** exp 2, exp 12 (canonical bloat content TBD).
- **Owner.** Ayush. Bloat content currently `BENIGN_SEED_PASSAGE` — Wikipedia-style; final choice waits on exp 12.

### exp 9 — continuous projection vs N (STALE — RERUN at L18)

- **Goal.** Continuous version of exp 8: instead of binary refusal verdict, measure scalar projection onto d_hat as function of N. Catches sub-threshold dilution exp 8 misses.
- **Inputs.** Same prompts + N grid as exp 8.
- **Method.** Run prompt + N-bloat through model, take last-token residual at probe layers, dot with d_hat. Plot per-layer projection vs N. Suraj P2 H2 collapses into this (cosine to V_refusal at last token).
- **Output.** `results/qwen3-14b/projection_sweep.{json,png}`. Current file uses layers=[20,24,28] — rerun w/ L18 (and keep neighbours for context).
- **Decision rule.** Descriptive — feeds exp 11 design (decide where steering helps).
- **Deps.** exp 2.
- **Owner.** Ayush.

### exp 10 — attention-mass on Guardrail Heads vs N (NEW — Suraj P2 H1 + P6 merged, PENDING)

- **Goal.** Mechanistic explanation for exp 8/9: do guardrail heads' attention mass shift OFF the harmful tokens AS bloat grows, diluting the refusal signal at the read site?
- **Inputs.** Guardrail head set from exp 3; merged N-grid up to N=4096 (OOM ceiling on single A100).
- **Method.** Merge `Suraj/_archive/experiment.py` Phase 2 H1 (mean attention mass on harmful-span tokens for top-K heads) + Phase 6 (per-head attribution map, source-flow visualisation). Reuse Suraj's z-hook reduce-on-the-fly trick to avoid materialising full `[B, H, T, T]` patterns.
- **Output.** `results/qwen3-14b/attn_mass/` — per-head per-N attention fraction CSV; attribution heatmaps; source-flow figures (orange=harmful span). Rows above N=4096 marked `status='OOM_measure'` rather than dropped.
- **Decision rule.** Predicts a monotone decrease in attention-mass-on-harmful as N grows for guardrail heads, NOT for control heads. Pre-register direction; test signed.
- **Deps.** exp 3 (head set), exp 9 (continuous projection — co-plotted).
- **Owner.** NEW. New code in `experiment_10/`.

### exp 11 — activation steering rescue (NEW — Suraj P3 port, PENDING)

- **Goal.** Can we *recover* refusal at long N by adding `α · d_hat` back at a target layer? If yes, confirms d_hat is a control knob, not just a probe.
- **Inputs.** L18 d_hat; harmful prompt + long-N bloat (test grid pinned to where exp 8 shows degradation).
- **Method.** Forward hook adds `α · d_hat` to residual at target layer (Suraj used `BEST_LAYER` — adapt to L18). α-grid sweep × N-grid. Sanity: 50-question MMLU subset to confirm steering doesn't tank capability.
- **Output.** `results/qwen3-14b/steering_rescue/` — α × N heatmap of refusal rate; MMLU-50 sanity table.
- **Decision rule.** Find α that restores intact-baseline refusal at long N without dropping MMLU accuracy by >X pp (X TBD, follow Suraj's threshold).
- **Deps.** exp 2, exp 9. **Scope note:** Suraj's original P3 saw only few-pp uplift because his creative-writing bloat already left behaviour near baseline at long N. Against Ayush's flat 94–98pp gap, headroom may be near-zero. After exp 12 fixes canonical bloat, exp 11 may reduce to a sanity demo. Don't preallocate scope — decide post-exp 12.
- **Owner.** NEW. New code in `experiment_11/`.

### exp 12 — content-type ablation (NEW — Greg ext 2, PENDING)

- **Goal.** Resolve the headline Suraj-vs-Ayush conflict: is the N=128 dip real, or a creative-writing-prose artefact? Choose canonical bloat content for exp 8.
- **Inputs.** Single harmful split, single L18 d_hat, fixed prefix position. Bloat *content type* varied as factor: {Ayush-Wiki `BENIGN_SEED_PASSAGE`, Suraj-Willowbrook-creative, code, multi-turn, distractor-instructions, many-shot}.
- **Method.** For each (content_type × N) cell, measure refusal rate (binary) and projection (continuous). Same N-grid as exp 8/9.
- **Output.** `results/qwen3-14b/content_ablation/` — content × N matrix.
- **Decision rule.** (a) If N=128 dip appears only on creative-writing → flag as content artefact, drop from headline. (b) If dip generalises across types → real finding, promote. Either way: pick exp 8 canonical bloat (likely the type with cleanest monotone signal).
- **Deps.** exp 2. Subsumes Suraj P5 multi-format 2×2 (cleaner factor design).
- **Owner.** NEW. New code in `experiment_12/`.

### exp 13 — long-context safety mitigation (NEW — Greg ext 1, PENDING)

- **Goal.** Fix the failure mode exp 8–11 diagnose: train a lightweight adapter that preserves refusal at long N without trashing capability.
- **Inputs.** Long-N refusal training data: harmful instruction wrapped in benign N-token prefix, paired with refusal completion. Held-out long-N AdvBench split for eval. Capability set piggybacks on exp 14.
- **Method.** Two passes. (1) LoRA SFT on (long-N harmful prefix → refusal) pairs. (2) DPO with non-refusal completions as negative. Targeted at attention modules near L18 (scoping pending).
- **Output.** `results/qwen3-14b/mitigation/` — LoRA weights, eval metrics on long-N AdvBench, capability deltas.
- **Decision rule.** Refusal at long-N must reach intact-baseline-N=0 floor; capability drop ≤ exp 14 threshold (TBD).
- **Deps.** exp 2 (d_hat for analysis, not training signal), exp 8 (failure points to target), exp 14 (capability eval).
- **Owner.** NEW. Scoping pending.

### exp 14 — capability cost MMLU/GSM8K (NEW — Suraj P4 extended, PENDING)

- **Goal.** Quantify capability tax of L18 ablation, both at N=0 and across long-N. Suraj's P4 only ran intact-vs-ablated at N=0 — extend to N as a row factor.
- **Inputs.** MMLU (200 items) + GSM8K (50 items), grown larger for held-out reliability. Intact arm + ablated arm, across merged N-grid.
- **Method.** For each (arm × N × benchmark) cell, accuracy via Suraj's `grade_capability` on greedy generations.
- **Output.** `results/qwen3-14b/capability_cost/` — capability vs N curves per arm.
- **Decision rule.** Descriptive — feeds exp 11 sanity gate and exp 13 acceptable-drop threshold.
- **Deps.** exp 2.
- **Owner.** NEW. Port from `Suraj/_archive/experiment.py:phase4` (lines ~1922–2058) but extend N axis.

---

## Suraj → numbered-experiment mapping

| Suraj phase | Old artefact | New home | What carries over | What changes |
|---|---|---|---|---|
| **P1** V_refusal + Guardrail Heads | `phase1_layer_sweep.csv`, `phase1_validation.csv`, `fig_phase1_*.png` | exp 2 (V_refusal — already redone by Ayush) + **NEW exp 3** (Guardrail Heads via DLA) | DiffMean recipe, DLA-onto-d_hat method, top-K head ranking | Layer 24 → 18 (causal criterion). L22H7 top-1 was provisional at L24; **redo DLA at L18** before promoting any head as canonical. Don't reuse `experiment.py` Phase 1 directly — port the DLA section against `experiment_1/utils.py` helpers. |
| **P2** context scaling | `phase2_scaling.csv`, `fig_phase2_scaling.png` | exp 8 (binary), exp 9 (continuous projection), exp 10 (H1 attn-mass) | N-grid shape, two-pass measurement+generation, OOM-marker convention | N-grid merged: `{0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768}` (your 128 + Ayush's longer points). H2 (cosine to V_refusal) folded into exp 9 as continuous projection. Bloat content TBD — pending exp 12. |
| **P3** activation steering | `phase3_rescue_grid.csv`, `phase3_mmlu.csv`, `fig_phase3_*.png` | **NEW exp 11** | α-sweep grid, MMLU-50 sanity guardrail | Re-eval against Ayush's bloat (flat 94–98pp gap) — your few-pp uplift was because behaviour already near baseline at long N with creative-writing bloat. With flat-baseline bloat there may be nothing to rescue at all; if so, exp 11 reduces to a sanity demo. Decide scope after exp 12 fixes the bloat. |
| **P4** capability cost | `phase4_capability.csv`, `fig_phase4_capability.png` | **NEW exp 14** (extended) | Intact-vs-ablated MMLU/GSM8K design | Extend item counts (200 MMLU + 50 GSM8K → larger held-out). Add long-N rows so capability cost reads as function of N, not just intact-vs-ablated at N=0. |
| **P5** multi-format 2×2 | `phase5_refusal_2x2.csv`, `fig_phase5_refusal_2x2.png` | folds into **NEW exp 12** | Format-as-factor design intuition | Greg ext 2 supersedes — exp 12 varies content type as a row factor at fixed prefix position with the merged N-grid. Cleaner than your 2×2. |
| **P6** attribution maps | `phase6_top_head_attn_fraction.csv`, `fig_phase6_*.png` | **NEW exp 10** (merged with P2 H1) | z-hook reduce-on-the-fly trick, source-flow viz | Use L18 d_hat for read-off; head set comes from exp 3 redo. OOM ceiling N=4096 stays. |

## Headline conflict — numbers for the record

Side-by-side, same model:

| Source | Bloat content | N=0 | N=128 | N=512 | N=2048 | N=8192 | N=32768 |
|---|---|---|---|---|---|---|---|
| Ayush exp_8 (Wiki) | `BENIGN_SEED_PASSAGE` | 0.99 | — | 0.95 | 0.95 | 0.95 | 0.95 |
| Suraj P2 (creative) | "Willowbrook" passage | 0.94 | 0.77 | 0.86 | 0.94 | OOM | OOM |
| Ablated arm (Ayush) | Wiki | 0.01 | — | 0.00 | 0.01 | 0.01 | 0.01 |

Falsifiability check passes on Ayush's arm: ablated stays flat-low end-to-end.
N=128 dip is creative-writing-only. Resolves via **exp 12** (Greg ext 2) —
both bloats become rows. exp_8 canonical baseline content TBD pending that
result (`INVARIANTS.md` §Bloat content).

## Other open issues

- **Eval leak in exp_6 harmless_mundane cell.** `splits.json:overlap_warnings` flags exp_2 d_hat training Alpaca[0:100) overlapping exp_6 raw harmless pool Alpaca[0:50). Decide: flag in paper, or rerun exp_6 with disjoint slice (e.g. [200:250)). Same shared-eval caveat (not training leak) for exp_6/exp_8 and exp_7/exp_8 AdvBench overlaps.
- **Refusal detector divergence.** Ayush's 18-marker `looks_like_refusal` differs from Suraj's 24-substring list. `INVARIANTS.md` §Refusal detector pins Ayush canonical. Cite the diff in any cell where verdict changes.
- **Attention hook OOM.** H1 measurement (exp 10) disabled above N=4096 on a single A100-80GB. Exp_8/9 N-grid extends to 32768 because they don't need the full `[B, H, T, T]` pattern. Mark `status='OOM_measure'` rows in CSV rather than dropping.
- **N-grid alignment.** `INVARIANTS.md` §N-grid: merged grid `{0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768}` adds Suraj's 128 (catches the dip) without dropping Ayush's points. exp_8 current `summary.json` is missing 128, 1024, 4096 — rerun to fill.

## Immediate next steps (in order)

1. Cascade rerun against L18 d_hat: exp_4 recheck, exp_5 first run, exp_6 rerun, exp_7 rerun, exp_8 rerun (also fill missing N grid), exp_9 rerun. None of these need new code — sbatches already updated.
2. Build exp 3 (Guardrail Heads): port `Suraj/_archive/experiment.py` Phase 1 DLA section, retarget to Ayush's L18 d_hat, write `results/qwen3-14b/guardrail_heads/`.
3. Build exp 10 (attn-mass vs N): merge Suraj P2 H1 + P6 hooks; depends on exp 3 head list.
4. Build exp 11 (steering rescue): port Suraj P3 against L18 d_hat.
5. exp 12 (content-type ablation) decides exp_8 canonical bloat.
6. exp 13, 14: scoping later.

## Branch conventions

- Commit prefix: `cascade:` for L20→L18 cascade work, `reconcile:` for chronology refactors, `Ayush |` / `Suraj |` for author-tagged single-experiment work (existing convention).
- Never edit `Suraj/_archive/` — it's the frozen reference. New code lives in `experiment_N/`.
- Update `INVARIANTS.md` *before* changing scripts that touch a shared knob.
