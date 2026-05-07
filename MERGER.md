# Suraj × Ayush merger — progress and plan

Living doc for `reconcile-experiments-suraj` branch. Tracks how Suraj's
six-phase archive (`Suraj/experiment.py` + `Suraj/results_v3/`) folds into Ayush's numbered
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
(`Suraj/experiment.py`, six phases, `Suraj/results_v3/`) is the 14B work
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

- **Canonical refusal layer 20 → 18 → 36.** Your archive used L24 (norm
  criterion on 14B). Ayush's exp_2 first reran at L20 (norm), then L18 under a
  128-token behavioral judge. **Reconcile rerun 2026-05-07** under stricter
  Suraj-style 24-token / 26-phrase judge: six layers tied at
  ablated_harmful=0.0 ({18,26,28,30,32,36}); tiebreak `abs(layer -
  norm_best_layer=39)` → **L36**. Old Suraj 14B-archive causal was also L36 →
  matches. **L36 is now canonical.** Every downstream consumer
  (exp_3/4/5/6/7/8/9 + your phases) must use L36 d_hat. Authoritative knob:
  `INVARIANTS.md` §Refusal direction.
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

## Greg's feedback (advisor) — source of new exp_12

Greg is project advisor. One extension added to plan after he reviewed
mid-sem report and v2 progress.

(Greg ext 1 — long-context mitigation via LoRA SFT + DPO — was scoped as
exp 13 but **dropped from scope 2026-05-07**. Project is diagnosis-only;
mechanistic story stops at exp 11 steering probe. exp 11 carries the only
"can we move it?" lever and promotes from sanity demo to mechanistic-knob
result. Flag with Greg in next advisor comms.)

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

- Canonical refusal layer **20 → 18 → 36** (reconcile rerun 2026-05-07 under
  Suraj-style 24-token / 26-phrase judge; tiebreak picked L36, closest to
  norm_best_layer=39). `results/qwen3-14b/refusal_direction/meta.json`:
  `causal_best_layer=36`, `canonical_layer=36`, `norm_best_layer=39`. Cascade
  L18→L36 landed across exp_4–8 (`7a21d61`) and exp_9 (`96b8dd0`).
- exp_1, exp_2, exp_3, exp_4, exp_5, exp_6, exp_7, exp_8, exp_9 all have
  artefacts on disk at canonical L36. exp_3 ran on cluster — top heads
  **L36H31 / L35H38 / L33H35** reproduce Suraj's circuit.
- Stale code constants from L18→L36 cascade — **all patched 2026-05-07**:
  - `experiment_5/eval_validity.py` `DECISION_LAYER = 18 → 36`,
    `DEFAULT_LAYERS = [18, 28] → [36, 28]`, verdict key
    `verdict_first_dir_layer18 → verdict_first_dir_layer{DECISION_LAYER}`,
    docstring updated. `validity/auc.json` rewritten in place: verdict
    `SEMANTIC` (AUC_intent=1.0 ≥ 0.85 AND AUC_vocab=0.5974 ∈ [0.35, 0.65]),
    `decision_constants.layer = 36`. No model rerun (projections unchanged).
  - `experiment_6/eval_topic_decouple.py:7,11` docstring L18 → L36.
  - `experiment_7/eval_policy_vs_harm.py` JSON key `verdict_layer18 →
    verdict_layer{DECISION_LAYER}`; `policy_validity/aucs.json` rewritten:
    both refusal dirs now under `verdict_layer36`.
  - `experiment_3/sbatch_guardrail_heads.sh:3,22` comment L18 → L36
    (reconciled-judge note added).
- Final 14-experiment chronology defined in `INVARIANTS.md` §Numbered-experiment
  chronology. exp_10/11/12/13 are NEW (Suraj P2-H1+P6, P3, Greg ext 2, P4)
  and not yet ported.
- Suraj phases 1, 2, 3, 4, 5, 6 → experiments 3, 8/10, 11, 13, 12, 10
  respectively (see mapping table below). His `experiment.py` and Phase
  CSVs are frozen in `Suraj/results_v3/` and serve as reference oracle, not
  build target.

## Experiment status table

| # | Experiment | Source | d_hat layer in artefact | Status | Result path |
|---|---|---|---|---|---|
| 1 | baseline refusal rates | Ayush | n/a | done | `results/qwen3-14b/baseline_summary.json` (harmful 0.937, harmless 0.014) |
| 2 | V_refusal extraction (DiffMean) | Ayush | **36** (causal, Suraj-style 24-tok / 26-phrase judge) | **done (reconciled 2026-05-07)** | `results/qwen3-14b/refusal_direction/{meta.json, d_hat_all_layers.pt}` (causal_best=36, norm_best=39) |
| 3 | Guardrail Heads (DLA top-K) | NEW (Suraj P1 port) | 36 | **done — top-3 L36H31 / L35H38 / L33H35; reproduces Suraj's circuit** | `results/qwen3-14b/guardrail_heads/{guardrail_heads.json, meta.json}` (top-12) |
| 4 | matched d_hat* | Ayush | 36 | **done at L36 — `cos_canonical=0.9748` → CLEAN** | `results/qwen3-14b/refusal_direction_matched/{meta.json, compare_to_orig.json}` |
| 5 | style/vocab validity | Ayush | 36 | **done at L36 — verdict `SEMANTIC` (AUC_intent=1.0, AUC_vocab=0.5974 in band)** | `results/qwen3-14b/validity/auc.json` (verdict_first_dir_layer36 set; decision_constants.layer=36) |
| 6 | topic 2x2 ANOVA | Ayush | 36 | **done at L36 — η²(intent)=0.9626, η²(topic)=0.0942 → CLEAN. Matched arm: η²(topic)=0.137 → MIXED, robustness via orthogonalisation** | `results/qwen3-14b/topic_decouple/{anova.json, orthogonalisation.json, cell_means.json}` |
| 7 | policy vs harm AUC | Ayush | 36 | **done at L36 — AUC(harm vs policy)=0.9812 → HARM_SPECIFIC** | `results/qwen3-14b/policy_validity/aucs.json` |
| 8 | binary refusal vs N | Ayush (split from old exp_2) | 36 | **done — full N-grid filled (0,128,512,1024,2048,4096,8192,16384,32768). Baseline flat 0.95–0.99; ablated 0.25–0.30 (NOT 0.0 — selection-pool / sweep-pool gap, see open issues)** | `results/qwen3-14b/context_sweep/summary.json` |
| 9 | continuous projection vs N | Ayush exp_9 | layers=[32, 36, 39] | **done at L36 (kept neighbours 32 + 39 for context)** | `results/qwen3-14b/projection_sweep.{json,png}` |
| 10 | attention-mass on Guardrail Heads vs N | NEW (Suraj P2 H1 + P6 merged) | — | pending — exp_3 unblocked, ready to port | — |
| 11 | activation steering rescue | NEW (Suraj P3) | — | pending — depends on exp 2, 9 | — |
| 12 | content-type ablation (Greg ext 2) | NEW | — | pending | — |
| 13 | capability cost MMLU/GSM8K (Suraj P4 extended) | NEW | — | pending | — |

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

### exp 2 — V_refusal extraction (DONE, reconciled 2026-05-07)

- **Goal.** Extract single direction `d_hat` in residual stream that mediates refusal.
- **Inputs.** AdvBench[100:200) harmful + Alpaca[0:100) harmless (training pool — disjoint from exp 1 baseline, NOT disjoint from exp 6 eval — see overlap warning).
- **Method.** DiffMean on last-token residual at every layer (0..39). For each layer compute (a) norm criterion, (b) causal-ablation criterion: project off `d_hat_l` from harmful prompts and measure refusal drop on 24-prompt held-out pool. **Reconcile selection-judge:** Suraj-style 24-token / 26-phrase substring match (replaces 128-token `looks_like_refusal` which over-counted tail-refusal).
- **Output.** `results/qwen3-14b/refusal_direction/{meta.json, d_hat_best_layer.pt, d_hat_all_layers.pt}`. `causal_best_layer=36`, `canonical_layer=36`, `norm_best_layer=39`. Six layers tied at ablated_harmful=0.0 ({18,26,28,30,32,36}); tiebreak `abs(layer - norm_best_layer)` selected L36 (closest to peak-norm).
- **Decision rule.** Pick layer minimising post-ablation harmful refusal subject to harmless refusal staying near 0; tiebreak by proximity to norm-best peak. → L36.
- **L22 anomaly.** Independent of judge or pool, `d_hat[22]` fails causally (~0.7 post-ablate harmful refusal) while neighbours L20/L24 succeed. Real layer-22 brittleness in Qwen3-14B's diff-of-means geometry; both Ayush and Suraj sweeps reproduce.
- **Deps.** None upstream; downstream feeds exp 3, 4, 5, 6, 7, 8, 9, 10, 11, 13.
- **Owner.** Ayush. Suraj P1 V_refusal section is methodological reference; Ayush's reconciled rerun is canonical. Ayush canonical L36 matches Suraj's old 14B-archive causal pick.

### exp 3 — Guardrail Heads via DLA (DONE — Suraj P1 port)

- **Goal.** Identify the small subset of attention heads whose direct logit attribution onto `d_hat` is largest — the "guardrail heads" hypothesised to write refusal into the residual.
- **Inputs.** L36 `d_hat` from exp 2; held-out harmful pool AdvBench[480:504) (n=24, registered in splits.json).
- **Method.** Ported DLA section of `Suraj/experiment.py:phase1`. For each (layer, head) ∈ (0..36, 0..39): compute head's per-token output written to residual, dot with `d_hat`, average over harmful prompts. Top-12 retained. Reused `experiment_1/utils.py` helpers; **dropped** Suraj's REFUSAL_SUBSTRINGS in favour of `looks_like_refusal`.
- **Output.** `results/qwen3-14b/guardrail_heads/{guardrail_heads.json, meta.json}`. Top-3: **L36H31** (DLA=14.72), **L35H38** (DLA=6.85), **L33H35** (DLA=4.41). Cumulative pos-mass to top-12 = 0.31. Reproduces Suraj's circuit (his old 14B top-1 was L36-region).
- **Decision rule.** Top-K=12 fixed; elbow on cumulative DLA mass falls within top-12 → safe cutoff for exp 10 head set.
- **Deps.** exp 2 (L36 d_hat). Unblocks exp 10 (head set ready), exp 11 (target layer for steering).
- **Owner.** Ayush port. `experiment_3/guardrail_heads.py`. Sbatch comment at line 3 still says "L18" — purely cosmetic; runtime read `canonical_layer` from `refusal_direction/meta.json`.

### exp 4 — matched d_hat\* (DONE at L36)

- **Goal.** Falsify "d_hat is just verb-class / length artefact" by re-extracting on verb-matched, length-matched harmful/harmless pools and comparing to canonical d_hat.
- **Inputs.** Curated matched pools (`experiment_4/data/`, built by `curate_matched_pools.py`).
- **Method.** Repeat exp 2 DiffMean on matched pools → `d_hat*`. Per-layer cosine(d_hat, d_hat\*).
- **Output.** `results/qwen3-14b/refusal_direction_matched/{meta.json, compare_to_orig.json}`. **At L36: `cos_canonical=0.9748` → CLEAN.** Per-layer cosine sequence rises from 0.69 (L0) to plateau ≥0.97 from L19 onward; canonical L36 sits well inside the plateau.
- **Decision rule (preregistered).** ≥0.90 = clean (keep canonical); ≤0.70 = contaminated (adopt d_hat\*); else escalate.
- **Deps.** exp 2.
- **Owner.** Ayush.

### exp 5 — style/vocab validity (RAN; verdict logic stale, must repatch)

- **Goal.** Falsify "d_hat tracks edgy lexicon / register, not harm intent" by varying surface style at fixed intent.
- **Inputs.** Styled pools built by `experiment_5/build_styled_pools.py` + `build_edgy_lexicon.py` (Edgy/Polite × harmful/harmless), 50 prompts per cell.
- **Method.** Project each style cell onto d_hat at probed layers [36, 28]; compute AUC(intent), AUC(vocab); decision constants `intent_hi=0.85`, `vocab_auc_lo=0.35`, `vocab_auc_hi=0.65`.
- **Output.** `results/qwen3-14b/validity/auc.json`. **At L36: AUC(intent)=1.0, AUC(vocab)=0.5974, Δintent=596.16, Δvocab=33.85** (matched arm: AUC(vocab)=0.5357). Both AUCs sit in the no-effect band → CLEAN on vocab axis. Verdict `SEMANTIC` written to `verdict_first_dir_layer36` (2026-05-07 patch — `DECISION_LAYER` flipped 18 → 36; verdict block rerun against existing projections, no model rerun).
- **Decision rule.** Within-intent style effect must be small relative to intent effect (`vocab_auc_lo` ≤ AUC(vocab) ≤ `vocab_auc_hi` AND AUC(intent) ≥ `intent_hi`).
- **Deps.** exp 2.
- **Owner.** Ayush.

### exp 6 — topic 2x2 ANOVA (DONE at L36; matched arm MIXED)

- **Goal.** Falsify "d_hat is domain shift between AdvBench and Alpaca." Confirms intent dominates topic.
- **Inputs.** intent × topic factorial pools (`experiment_6/data/`): {harmful_violent, harmful_mundane, harmless_violent, harmless_mundane}, 50 per cell, harmless_mundane migrated to Alpaca[200:250) post `_harmless_instruction_is_clean` filter to close earlier eval leak (commits `2c4d65f` + `9d5e0c4` + `abf87b4`).
- **Method.** 2x2 ANOVA on projection-onto-d_hat. Partial η²(intent), η²(topic), η²(intent:topic). Probed layers [36, 28].
- **Output.** `results/qwen3-14b/topic_decouple/{anova.json, orthogonalisation.json, cell_means.json}`.
  - **L36 canonical d_hat:** η²(intent)=0.9626, η²(topic)=0.0942, η²(intent:topic)=0.0018, AUC(intent)=1.0, AUC(topic)=0.6197 → **CLEAN**.
  - **L36 matched d_hat\*:** η²(intent)=0.9571, η²(topic)=0.1369, AUC(topic)=0.6256 → **MIXED** (topic effect non-trivial; orthogonalise as robustness check — see `orthogonalisation.json`).
- **Decision rule (preregistered).** η²(intent) ≥ 0.5 AND η²(topic) ≤ 0.1.
- **Deps.** exp 2. Eval leak closed; no remaining caveat for harmless_mundane.
- **Stale docstring** (patched 2026-05-07): `experiment_6/eval_topic_decouple.py:7,11` now reads "layer 36". `DECISION_LAYER = 36` was already set.
- **Owner.** Ayush.

### exp 7 — policy vs harm AUC (DONE at L36)

- **Goal.** Falsify "d_hat is a generic compliance/decline direction." Show it discriminates harm-refusal from policy-refusal.
- **Inputs.** Harm-refusal pool (AdvBench[50:100) — declines for safety) + policy-refusal pool (SORRY-Bench, model declines for compliance/format/identity reasons, not harm).
- **Method.** Project both pools onto L36 d_hat; compute ROC-AUC of harm vs policy. Preflight gate (`preflight_refusal_rates.py`) checks both pools refuse at sufficient rates before AUC.
- **Output.** `results/qwen3-14b/policy_validity/aucs.json`.
  - **L36 canonical:** AUC(harm vs harmless)=1.0, AUC(policy vs harmless)=0.9976, **AUC(harm vs policy)=0.9812** → **HARM_SPECIFIC**.
  - **L36 matched:** AUC(harm vs policy)=0.9812 → HARM_SPECIFIC (robustness confirmed).
- **Decision rule (preregistered).** AUC ≥ 0.85 = harm-specific.
- **Deps.** exp 2. Shared-eval caveat (AdvBench[50:100) ⊂ exp 8 eval space) — flag in paper, not training leak.
- **Cosmetic note (patched 2026-05-07):** JSON key renamed `verdict_layer18` → `verdict_layer36` in `policy_validity/aucs.json` (both refusal dirs); `eval_policy_vs_harm.py` writes `verdict_layer{DECISION_LAYER}` so future reruns stay in sync.
- **Owner.** Ayush.

### exp 8 — binary refusal vs N (DONE at L36, full N-grid)

- **Goal.** Headline phenomenon: does refusal rate degrade as benign context is prepended? Both intact and ablated arms.
- **Inputs.** Harmful prompt + length-N benign bloat prefix (`BENIGN_SEED_PASSAGE`, Wikipedia-style); intact model vs L36 ablation. n_harmful=100, max_new_tokens=256, greedy decode (T=0).
- **Method.** For each N in `{0,128,512,1024,2048,4096,8192,16384,32768}`, generate completions, measure refusal rate via `looks_like_refusal`. Two arms: intact (`baseline`), ablated (project off d_hat at L36 across all 40 layers).
- **Output.** `results/qwen3-14b/context_sweep/summary.json`. **All cells filled.**

  | N | baseline refusal | ablated refusal |
  |---|---|---|
  | 0 | 0.99 | 0.29 |
  | 128 | 0.97 | 0.25 |
  | 512 | 0.95 | 0.19 |
  | 1024 | 0.96 | (filled) |
  | 2048 | 0.95 | 0.27 |
  | 4096 | 0.95 | (filled) |
  | 8192 | 0.95 | 0.30 |
  | 16384 | 0.95 | 0.26 |
  | 32768 | 0.95 | 0.25 |

  Baseline flat 0.95–0.99 across all N (no creative-writing dip — Wiki bloat behaves like Ayush's earlier L18 sweep, not Suraj's creative-writing dip).
- **Falsifiability concern.** Ablated arm hovers **0.19–0.30**, **NOT flat at 0.0** as exp_2's 24-prompt held-out predicted (`refusal_direction/meta.json` reports ablated_harmful=0.0 at L36). Selection-pool (24 prompts, AdvBench[480:504)) → broad-sweep-pool (100 prompts) generalisation gap. Two readings:
  - Selection judge (Suraj 24-tok / 26-phrase) is stricter than `looks_like_refusal` used in the sweep — same ablation, different verdict.
  - Generalisation: 24-prompt pool was a tight-fit selector; 100-prompt sweep exposes residual refusal directions ablation doesn't catch.

  **Decision (2026-05-07): option (a) — cite both.** Paper frames as "selection-judge ablation is complete on the selection pool (24-prompt held-out, Suraj-style 24-tok / 26-phrase judge → ablated_harmful=0.0); broad-pool sweep (100-prompt AdvBench, `looks_like_refusal` 18-marker / 200-char window) retains ~25–30% refusal — d_hat is the dominant but not sole refusal direction at L36." No ablated-arm rerun under matching judge.
- **Decision rule.** Intact arm shape is the result, not a pass/fail. Falsifiability check (ablated stays low) **partially holds**: ablation drops 0.95–0.99 → 0.19–0.30 (~70 pp drop) at every N — large effect, not zeroed.
- **Deps.** exp 2, exp 12 (canonical bloat content TBD).
- **Owner.** Ayush. Bloat content currently Wiki; final choice waits on exp 12.

### exp 9 — continuous projection vs N (DONE at L36, neighbours [32, 39])

- **Goal.** Continuous version of exp 8: instead of binary refusal verdict, measure scalar projection onto d_hat as function of N. Catches sub-threshold dilution exp 8 misses.
- **Inputs.** Same harmful prompts + merged N grid as exp 8.
- **Method.** Run prompt + N-bloat through model, take last-token residual at probe layers, dot with d_hat. Plot per-layer projection vs N. Suraj P2 H2 collapses into this (cosine to V_refusal at last token).
- **Output.** `results/qwen3-14b/projection_sweep.{json,png}`. **Probe layers: [32, 36, 39]** (L36 canonical + neighbour 32 + norm-best 39 for context).
- **Decision rule.** Descriptive — feeds exp 11 design (decide where steering helps).
- **Deps.** exp 2.
- **Owner.** Ayush.

### exp 10 — attention-mass on Guardrail Heads vs N (NEW — Suraj P2 H1 + P6 merged, READY TO PORT)

- **Goal.** Mechanistic explanation for exp 8/9: do guardrail heads' attention mass shift OFF the harmful tokens AS bloat grows, diluting the refusal signal at the read site?
- **Inputs.** Guardrail head set from exp 3 (top-12 at L36, lead heads **L36H31 / L35H38 / L33H35**); merged N-grid up to N=4096 (OOM ceiling on single A100).
- **Method.** Merge `Suraj/experiment.py` Phase 2 H1 (mean attention mass on harmful-span tokens for top-K heads) + Phase 6 (per-head attribution map, source-flow visualisation). Reuse Suraj's z-hook reduce-on-the-fly trick to avoid materialising full `[B, H, T, T]` patterns.
- **Output.** `results/qwen3-14b/attn_mass/` — per-head per-N attention fraction CSV; attribution heatmaps; source-flow figures (orange=harmful span). Rows above N=4096 marked `status='OOM_measure'` rather than dropped.
- **Decision rule.** Predicts a monotone decrease in attention-mass-on-harmful as N grows for guardrail heads, NOT for control heads. Pre-register direction; test signed.
- **Deps.** exp 3 (head set, **DONE**), exp 9 (continuous projection — co-plotted, **DONE**). **Unblocked.**
- **Owner.** NEW. Port code into `experiment_10/`. Reuse `experiment_1/utils.py` helpers; **drop** Suraj's REFUSAL_SUBSTRINGS, use canonical `looks_like_refusal`. Retarget L24 → L36, head set from exp_3 JSON.

### exp 11 — activation steering rescue (NEW — Suraj P3 port, PENDING)

- **Goal.** Can we *recover* refusal at long N by adding `α · d_hat` back at a target layer? If yes, confirms d_hat is a control knob, not just a probe.
- **Inputs.** L36 d_hat; harmful prompt + long-N bloat (test grid pinned to where exp 8 shows degradation).
- **Method.** Forward hook adds `α · d_hat` to residual at target layer (Suraj used `BEST_LAYER` — retarget to L36). α-grid sweep × N-grid. Sanity: 50-question MMLU subset to confirm steering doesn't tank capability.
- **Output.** `results/qwen3-14b/steering_rescue/` — α × N heatmap of refusal rate; MMLU-50 sanity table.
- **Decision rule.** Find α that restores intact-baseline refusal at long N without dropping MMLU accuracy by >X pp (X TBD, follow Suraj's threshold).
- **Deps.** exp 2, exp 9.
- **Scope note (revised after exp_8 done).** Two-track motivation now:
  - **Track A (intact rescue):** Ayush's exp_8 baseline arm is already flat 0.95–0.99 across N=0..32k → no rescue headroom on this baseline. If exp_12 picks a content type that *does* show degradation, run rescue on that content. Else demote to sanity demo.
  - **Track B (ablated rescue):** ablated arm sits at 0.19–0.30 (NOT 0.0). Steering should be able to push ablated → baseline by adding back α·d_hat. This is a clean control-knob demo regardless of exp_12 outcome.
- **Owner.** NEW. New code in `experiment_11/`.

### exp 12 — content-type ablation (NEW — Greg ext 2, PENDING)

- **Goal.** Resolve the headline Suraj-vs-Ayush conflict: is the N=128 dip real, or a creative-writing-prose artefact? Choose canonical bloat content for exp 8.
- **Inputs.** Single harmful split, single L36 d_hat, fixed prefix position. Bloat *content type* varied as factor: {Ayush-Wiki `BENIGN_SEED_PASSAGE`, Suraj-Willowbrook-creative, code, multi-turn, distractor-instructions, many-shot}.
- **Method.** For each (content_type × N) cell, measure refusal rate (binary) and projection (continuous). Same N-grid as exp 8/9.
- **Output.** `results/qwen3-14b/content_ablation/` — content × N matrix.
- **Decision rule.** (a) If N=128 dip appears only on creative-writing → flag as content artefact, drop from headline. (b) If dip generalises across types → real finding, promote. Either way: pick exp 8 canonical bloat (likely the type with cleanest monotone signal).
- **Deps.** exp 2. Subsumes Suraj P5 multi-format 2×2 (cleaner factor design).
- **Owner.** NEW. New code in `experiment_12/`.

### exp 13 — capability cost MMLU/GSM8K (NEW — Suraj P4 extended, PENDING)

- **Goal.** Quantify capability tax of L36 ablation, both at N=0 and across long-N. Suraj's P4 only ran intact-vs-ablated at N=0 — extend to N as a row factor.
- **Inputs.** MMLU (200 items) + GSM8K (50 items), grown larger for held-out reliability. Intact arm + ablated arm (project off L36 d_hat across all 40 layers, matching exp 8 ablation), across merged N-grid.
- **Method.** For each (arm × N × benchmark) cell, accuracy via Suraj's `grade_capability` on greedy generations.
- **Output.** `results/qwen3-14b/capability_cost/` — capability vs N curves per arm.
- **Decision rule.** Descriptive — feeds exp 11 sanity gate.
- **Deps.** exp 2.
- **Owner.** NEW. Port from `Suraj/experiment.py:phase4` (lines ~1922–2058) but extend N axis.

(Old exp 13 — Greg ext 1 long-context mitigation via LoRA SFT + DPO — dropped from scope 2026-05-07. Diagnosis-only project.)

---

## Suraj → numbered-experiment mapping

| Suraj phase | Old artefact | New home | What carries over | What changes |
|---|---|---|---|---|
| **P1** V_refusal + Guardrail Heads | `phase1_layer_sweep.csv`, `phase1_validation.csv`, `fig_phase1_*.png` | exp 2 (V_refusal — Ayush rerun, reconciled to L36) + **exp 3 DONE** (Guardrail Heads via DLA at L36) | DiffMean recipe, DLA-onto-d_hat method, top-K head ranking | Layer 24 → 36 (causal criterion under stricter judge; matches your old 14B-archive causal pick). DLA top-3 at L36: **L36H31 / L35H38 / L33H35**. Ported into `experiment_3/guardrail_heads.py`; reuses `experiment_1/utils.py`, drops Suraj REFUSAL_SUBSTRINGS. |
| **P2** context scaling | `phase2_scaling.csv`, `fig_phase2_scaling.png` | exp 8 (binary), exp 9 (continuous projection), exp 10 (H1 attn-mass) | N-grid shape, two-pass measurement+generation, OOM-marker convention | N-grid merged: `{0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768}` (your 128 + Ayush's longer points). H2 (cosine to V_refusal) folded into exp 9 as continuous projection. Bloat content TBD — pending exp 12. |
| **P3** activation steering | `phase3_rescue_grid.csv`, `phase3_mmlu.csv`, `fig_phase3_*.png` | **NEW exp 11** | α-sweep grid, MMLU-50 sanity guardrail | Retarget to L36. Two tracks: (A) intact rescue — Ayush exp_8 baseline arm flat 0.95–0.99 → no headroom on Wiki bloat, depends on exp_12 picking content with degradation; (B) ablated rescue — ablated arm sits 0.19–0.30, clean control-knob demo by adding back α·d_hat. Track B viable today, track A waits on exp_12. |
| **P4** capability cost | `phase4_capability.csv`, `fig_phase4_capability.png` | **NEW exp 13** (extended) | Intact-vs-ablated MMLU/GSM8K design | Extend item counts (200 MMLU + 50 GSM8K → larger held-out). Add long-N rows so capability cost reads as function of N, not just intact-vs-ablated at N=0. |
| **P5** multi-format 2×2 | `phase5_refusal_2x2.csv`, `fig_phase5_refusal_2x2.png` | folds into **NEW exp 12** | Format-as-factor design intuition | Greg ext 2 supersedes — exp 12 varies content type as a row factor at fixed prefix position with the merged N-grid. Cleaner than your 2×2. |
| **P6** attribution maps | `phase6_top_head_attn_fraction.csv`, `fig_phase6_*.png` | **NEW exp 10** (merged with P2 H1) | z-hook reduce-on-the-fly trick, source-flow viz | Use L36 d_hat for read-off; head set comes from exp 3 (top-12 at L36, lead heads L36H31 / L35H38 / L33H35). OOM ceiling N=4096 stays. |

## Headline conflict — numbers for the record (post-cascade L36)

Side-by-side, same model:

| Source | Bloat content | N=0 | N=128 | N=512 | N=2048 | N=8192 | N=32768 |
|---|---|---|---|---|---|---|---|
| Ayush exp_8 baseline (Wiki) | `BENIGN_SEED_PASSAGE` | 0.99 | 0.97 | 0.95 | 0.95 | 0.95 | 0.95 |
| Suraj P2 (creative, archive L24) | "Willowbrook" passage | 0.94 | 0.77 | 0.86 | 0.94 | OOM | OOM |
| Ayush exp_8 ablated (L36, Wiki) | `BENIGN_SEED_PASSAGE` | 0.29 | 0.25 | 0.19 | 0.27 | 0.30 | 0.25 |

**Updates after cascade rerun:**
- Baseline arm now full N-grid; flat 0.95–0.99 — no creative-writing dip on Wiki.
- Ablated arm at L36 sits 0.19–0.30 (under `looks_like_refusal`), **not** 0.0
  as exp_2's selection-judge sweep on the 24-prompt held-out reports. Two
  candidate explanations: (i) judge mismatch — selection used Suraj-style
  24-tok / 26-phrase, sweep uses 128-tok `looks_like_refusal`; (ii) genuine
  generalisation gap from 24→100-prompt pool. Decide for paper.
- N=128 dip remains creative-writing-only on this evidence.

Resolves via **exp 12** (Greg ext 2) — both bloats become rows. exp_8 canonical
baseline content TBD pending that result (`INVARIANTS.md` §Bloat content).

## Other open issues

- **Eval leak in exp_6 harmless_mundane cell — CLOSED 2026-05-06.** Migrated to Alpaca[200:250) post `_harmless_instruction_is_clean` filter (commits `2c4d65f` + `9d5e0c4` + `abf87b4`). Two shared-eval caveats remain (not training leaks): exp_6/exp_8 and exp_7/exp_8 AdvBench overlaps — flag in paper.
- **Selection-pool vs broad-sweep-pool gap (exp_8 ablated arm) — RESOLVED 2026-05-07.** Option (a) chosen: cite both numbers in paper. exp_2 reports ablated_harmful=0.0 on 24-prompt held-out under Suraj-style judge; exp_8 reports 0.19–0.30 on 100-prompt pool under `looks_like_refusal`. Frame: "d_hat dominant but not sole refusal direction at L36." No ablated-arm rerun under matching judge.
- **Refusal detector divergence.** Ayush's 18-marker `looks_like_refusal` (project default) differs from Suraj's 24-substring list (used in exp_2 selection judge after reconcile). `INVARIANTS.md` §Refusal detector pins behavioral canonical = `looks_like_refusal`. Cite the diff in any cell where verdict changes.
- **Attention hook OOM.** H1 measurement (exp 10) disabled above N=4096 on a single A100-80GB. Exp_8/9 N-grid extends to 32768 because they don't need the full `[B, H, T, T]` pattern. Mark `status='OOM_measure'` rows in CSV rather than dropping.
- **N-grid alignment — CLOSED.** Merged grid `{0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768}` filled in exp_8 `summary.json`.
- **Stale code constants from L18→L36 cascade — RESOLVED 2026-05-07.**
  - `experiment_5/eval_validity.py` `DECISION_LAYER = 18 → 36`,
    `DEFAULT_LAYERS = [18, 28] → [36, 28]`, verdict key →
    `verdict_first_dir_layer{DECISION_LAYER}`, docstring updated.
    `validity/auc.json` rewritten in place: verdict `SEMANTIC`,
    `decision_constants.layer = 36`. No model rerun.
  - Docstrings at `experiment_5/eval_validity.py:4,10` and
    `experiment_6/eval_topic_decouple.py:7,11` updated L18 → L36.
  - `experiment_3/sbatch_guardrail_heads.sh:3,22` comment L18 → L36 (with
    reconciled-judge note).
  - `experiment_7/eval_policy_vs_harm.py` writes
    `verdict_layer{DECISION_LAYER}` into `aucs.json`; existing
    `policy_validity/aucs.json` keys renamed `verdict_layer18 →
    verdict_layer36` for both refusal dirs.

## Immediate next steps (in order, post 2026-05-07 cascade)

**Cleanup — DONE 2026-05-07:**
1. ~~Patch `experiment_5/eval_validity.py` DECISION_LAYER 18→36 + rerun verdict block.~~ Done — verdict `SEMANTIC` written to `validity/auc.json`.
2. ~~Patch stale docstrings + sbatch comment.~~ Done — exp_5/exp_6 docstrings, exp_3 sbatch, exp_7 verdict key.

**Selection-vs-sweep judge mismatch — RESOLVED 2026-05-07:** option (a) chosen — cite both numbers, frame "d_hat dominant but not sole." No ablated-arm rerun.

**New experiments (Suraj phase ports):**
3. **exp 10** (attn-mass vs N): merge Suraj P2 H1 + P6 hooks, retarget L36 + exp_3 head set, OOM ceiling N=4096. Unblocked.
4. **exp 11** (steering rescue): port Suraj P3 against L36 d_hat. Track B (ablated rescue) viable today; track A (intact rescue) waits on exp_12.
5. **exp 12** (content-type ablation, Greg ext 2): build content × N matrix at L36; subsumes Suraj P5; decides exp_8 canonical bloat.
6. **exp 13** (capability cost): port `Suraj/experiment.py:phase4` against L36, extend N axis.

## Branch conventions

- Commit prefix: `cascade:` for layer-cascade work (L20→L18→L36), `reconcile:` for chronology refactors, `Ayush |` / `Suraj |` for author-tagged single-experiment work (existing convention).
- Update `INVARIANTS.md` *before* changing scripts that touch a shared knob.

## Suraj port source-of-truth (2026-05-07 pin)

Only two artefacts under `Suraj/` are valid port sources for exp 10/11/12/13:

- **`Suraj/experiment.py`** — function-level port surface. Phase entrypoints:
  - `phase1` (~L934) → already absorbed into exp 2 (V_refusal) + exp 3 (Guardrail Heads).
  - `phase2_triage` (~L1239), `phase2_dense` (~L1333), `phase2_5` (~L1569) → **exp 10** (attn-mass H1).
  - `phase3_rescue` (~L1603), `phase3_mmlu` (~L1826) → **exp 11** (steering rescue + MMLU sanity).
  - `phase4` (~L1926) → **exp 13** (capability cost).
  - `phase5` (~L2066) → **exp 12** (content-type ablation, subsumes Suraj P5 multi-format).
  - `phase6` (~L2240) → **exp 10** (attribution map / source-flow viz, merged with phase2).
  - Helpers reusable across ports: `BloatBuilder` (L334), `ModelHelpers` (L437), `make_ablation_hooks` (L505), `make_steering_hook` (L515), `cache_last_token_resid` (L904), `load_capability_set` (L1738), `format_capability` (L1817), `grade_capability` (L271). Reuse — do not rewrite.
  - **Drop on port**: `is_refusal` (L260) — Suraj's 24-substring detector. Replace with `experiment_1/utils.py:looks_like_refusal`.

- **`Suraj/results_v3/`** — frozen reference oracle. Compare numbers, never overwrite.
  - **exp 10 sources**: `phase2_focal_distractor.csv` (attn_to_harmful, cos_refusal, cos_refusal_readout, cos_refusal_harmful_span × format × seed × N), `phase2_jailbreak_thresholds.csv`, `phase2_triage.csv`, `phase6_attn_fraction_perprompt.csv`, `phase6_top_head_attn_fraction.csv`, figs `fig_phase2_dense_distractor.png`, `fig_phase6_attribution_*.png`.
  - **exp 11 sources**: `phase3_rescue_distractor.csv` (refusal vs α × N × format), `phase3_mmlu_steering.csv` (capability sanity), figs `fig_phase3_*.png`.
  - **exp 12 sources**: `phase5_multi.csv` (format × setting × prompt_type × seed × N × refusal_rate, all six content types), `phase5_killer_comparison.csv`, fig `fig_phase5_multi.png`. Note: distractor + ablated → refusal_rate ≈ 0.0 across N — confirms ablation works under matching format.
  - **exp 13 sources**: `phase4_capability.csv` (intact+ablated × N=[0, 4096] grid — Suraj already had partial N axis, not just N=0; extend to merged grid), `capability_set.json` (full MMLU + GSM8K eval pack — port directly), fig `fig_phase4_capability.png`.

- **Out of port scope (do not touch):** `Suraj/phase123_v2.ipynb*` (notebook + 3 backups, ~1.5MB; drift state vs `experiment.py` not tracked), `Suraj/results_v2/` (older results pre-`v3`), `Suraj/PLAN.md`, `Suraj/README.md` (Suraj-side plan, not project plan), `Suraj/environment.yml`, `Suraj/requirements.txt` (use project `requirements.txt`).

- **Out of current 13-experiment scope (decide later):** `Suraj/results_v3/phase7_circuit_tracing/` (`phase7_circuit_metrics.csv`, `run_circuit_tracer_commands.sh`, prompt manifest) and `Suraj/results_v3/phase7b_path_patching/` (`layer_sweep.csv`, `head_zoom.csv`). `experiment.py:phase7_analyze_graphs` (L2525), `phase7_path_patching` (L2746), `phase7_circuit_tracing` (L3103). Mechanistic depth beyond exp 10. Flag with Greg if circuit-level depth becomes a paper requirement; else freeze.

## Per-exp port checklist (10/11/12/13)

For each new experiment_N port — extract from `experiment.py` only, validate against `results_v3/` numbers:

1. Create `experiment_N/` dir.
2. Extract relevant phase function(s) from `Suraj/experiment.py`. Reuse helper classes (`BloatBuilder`, `ModelHelpers`). Drop `is_refusal`, import `looks_like_refusal` from `experiment_1/utils.py`.
3. Retarget `BEST_LAYER` / `BEST_DIR_LAYER` → 36; load d_hat from `results/qwen3-14b/refusal_direction/d_hat_all_layers.pt`.
4. For exp 10: load head set from `results/qwen3-14b/guardrail_heads/guardrail_heads.json` (top-12).
5. Register dataset slice in `splits.json`.
6. Write outputs under `results/qwen3-14b/<exp_name>/` — never under `Suraj/`.
7. Sanity-compare numbers vs corresponding `Suraj/results_v3/phaseN_*.csv` row — flag any large delta as judge-mismatch or layer-mismatch artefact.
8. Sbatch retarget per `feedback_single_launcher.md` — edit canonical in place.
