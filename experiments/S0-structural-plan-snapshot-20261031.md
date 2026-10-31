# Structural plan: from 0.86687 to the winning score

Written 2026-10-31 at the user's request: "we need a structural change. can
you devise a new plan?" Goal (GOAL.md): **quiz 0.8554** (test 0.8567). With the
measured offset (test ≈ dev CV − 0.0074) that is **≈0.8628 dev CV**, a gap of
**0.0041** from R9-LAT-A2h's 0.8668706.

## Diagnosis: why incremental bets stopped working

1. **Our base models are weaker than the winners'.** After converting their
   quiz numbers to probe-equivalent (GAP_AUDIT.md §0):
   - Pragmatic Theory's best single predictor (integrated model → polynomial
     post-processing → kNN chain) is ≈0.879 probe.
   - BellKor's timeSVD++ with frequency factors is ≈0.883–0.885.
   - **Our best single column, B4 Bayes, is 0.8897**; K1 is 0.894.

   The winners blended hundreds of predictors that were each 0.005–0.010
   better than ours. We tried to make up for weak bases with more blend
   columns of the same weak families, and those saturate (ROUND6, R10–R13, LC, HR, E-bets).
2. **We select models the wrong way for an ensemble.** Every column's epoch is
   chosen by standalone P2 RMSE. BellKor and BigChaos deliberately over-trained
   and stopped at the best *blend* RMSE (GAP_AUDIT §0.2). We have never done
   that systematically.
3. **Our one unique strength is under-scaled.** A2h, the column the winners
   did not have, is a 2-layer, d_model-64 transformer. Its history length was
   still paying when we stopped: 384→768 gave −0.0053 standalone, 768→1536
   gave −0.0066. 3072 failed only on an MPS INT_MAX limit, not on accuracy.
   The one width/depth test (A2c) was compared at matched early epochs with
   unchanged hyperparameters. That is not a scaling study.
4. **The headroom map rules out shortcuts.** The user-level shift is absorbed
   by Stage C (A7), the movie side is empty, simple proxies add nothing, and
   choice history (N1) shrank to −0.00017 in the full bank (HEADROOM.md). The
   remaining gain has to come from **better models**, not new side information.

## The plan: four lanes, each gated

Rough gain ranges are planning guesses from the cited gaps and our history of
how standalone gains shrink in the blend. They are not forecasts.

### Lane 1 — Winner-grade base models (CPU, C++; weeks 1–3)

Rebuild the base layer to published single-model quality, then chain each
model the way the winners did.

- **1a. Pragmatic Theory integrated model ("BK4"):** low-rank time-dependent
  user biases, frequency terms and integrated neighbourhood. GAP_AUDIT lists
  it as not tried, with the largest single-model gap (≈0.010).
- **1b. BellKor timeSVD++ with frequency factors at large f**, trained past
  the standalone optimum.
- **1c. Chains on every strong model:** polynomial post-processing ("nlpp")
  and kNN on its residuals ("kNN on every model": BigChaos §5.2, PT kNN3).
- **Gate per model:** standalone within 0.003 of the paper's probe-equivalent,
  then ≥0.0005 in the full-bank dev CV (all five folds).
- **Range if it works:** 0.0010–0.0020.

### Lane 2 — Scale the transformer properly (GPU; weeks 1–4, parallel)

- **2a. Long context.** Implement chunked or memory-efficient attention so
  histories of 3072+ fit on MPS. Only 5.3% of probe rows change context at
  3072, so test it on them first.
- **2b. A real width/depth scaling study.** Run d_model 64/128/256 × layers
  2/4/6 with the learning rate, warmup and dropout re-tuned per size on P2,
  trained to convergence. The current family has never had its optimiser
  scaled with its size.
- **2c. Fold the paying side signals into one model:** choice tokens (N1),
  date tokens and the distribution loss (E3) go into the scaled model rather
  than separate small models.
- **2d.** Seeds and snapshots of the best size for the bank.
- **Compute.** The M3 Max runs A2h at about 3 h per 10 epochs. A scaling study
  needs roughly 10–20× that. **Renting cloud GPUs is the main lever here and
  needs your approval and budget.** Without them, lane 2 is limited to about
  two to three sizes.
- **Gate:** each size must beat A2h standalone on P2 by ≥0.002 before a
  full-bank test, then ≥0.0005 in the bank on all folds.
- **Range if it works:** 0.0010–0.0020.

### Lane 3 — Blend-aware training and selection (CPU, cheap; week 1 onward)

- **3a.** For each strong family, save every epoch or sweep and pick the one
  that maximises **P2 blend gain** over the bank, not standalone RMSE (BigChaos
  Alg. 1). This is P2 only, so no probe leakage.
- **3b.** Over-trained variants of the factor models and RBMs, kept *only* if
  they add in the blend.
- **3c.** Re-fit the final blender (LightGBM ⊕ NN2) on the rebuilt bank, with
  per-user-cluster blending features (PT, BigChaos bins).
- **Range:** 0.0005–0.0010.

### Lane 4 — Production pipeline and offset recalibration (CPU + GPU; weeks 2–5)

Nothing counts without it.

- **4a.** Run the full Stage C retrain (train + probe) for every retained
  column, and produce qualifying predictions end to end. PRODUCTION_READINESS.md
  covers the recipes, but the pipeline has never been run in full.
- **4b.** Re-measure the Stage C offset on the *final* bank. GAP_AUDIT §0.5
  says it may differ by ±0.0005 from 0.0074 for a bank this different. That is
  calibration, not a gain claim.
- **Output:** a qualifying prediction file and an estimated test score with
  its uncertainty.

## Order and decision points

| Week | Work | Decision at end |
|---|---|---|
| 0 (1–2 days) | **Scoping.** (i) Diversity curve: blend random subsets of our 79 columns and see how RMSE falls with count. (ii) Quality-substitution check: swap each family's best column for a weaker sibling and measure the blend change per unit of standalone RMSE. That converts "this model is 0.010 better standalone" into an expected bank gain. | Which of lanes 1 and 2 has more expected value per week; whether 0.0041 is reachable on paper |
| 1–3 | Lane 1 (1a first), lane 3a in parallel; lane 2a/2b on GPU | Kill any lane whose first model misses its standalone gate |
| 3–4 | Lane 2c/2d, lane 1c chains, lane 3b/3c | Full-bank dev CV re-measured weekly against ≈0.8628 |
| 4–5 | Lane 4 | Estimated quiz/test score with uncertainty |

**Stop rule.** After week 3, if the bank has not improved by at least 0.0015
dev CV (more than a third of the gap), report that the winning score is out
of reach on this compute and evaluate before continuing.

## Honesty constraints

- **No hidden-test verification.** Qualifying labels are not available
  locally. Every reported score is an estimate (dev CV − measured offset).
- **The quiz number is not fully comparable.** The winners' quiz 0.8554
  partly reflects quiz-set feedback blending (BigChaos §7), which has no
  analogue without leaderboard access. Their test score of 0.8567 is the
  fairer comparison. We keep 0.8554 as the goal by your choice and report
  both.
- **One shared program.** Both sessions on this branch (Claude, Codex)
  claim lanes in GOAL.md before starting, and every gate is written down
  before its score.
