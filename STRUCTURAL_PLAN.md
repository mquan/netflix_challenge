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

## Timeline on the current machine (no cloud GPUs — user decision, 2026-10-31)

Hardware: one Apple M3 Max (10 performance + 4 efficiency CPU cores, 30-core
GPU, 96 GB unified memory). Measured costs from `experiments/*.json`:

| Job | Time |
|---|---:|
| A2h select + Stage C refit | 7.7 h + 4.9 h |
| N1 refit | 8.2 h |
| Bayes r128 (B3/B4), one run at 4 threads | ≈12.5 h |
| K1 SGD / RBM, one run | ≈1 h / ≈2 h |
| Full-bank comparison (tree s3 + NN2) | ≈1 h per bank |

| Lane | Estimate | Wall-clock |
|---|---|---|
| 0. Scoping | blends on existing columns | 1–2 days |
| 1. Base models (CPU) | BK4 3–5 days of code + 10–20 runs of 2–6 h; timeSVD++ at f 1000–2000 ≈8 h/run; chains hours each | 2–3 weeks |
| 2. Transformer (GPU) | long-context attention 2–4 days of code; d128 L4 ≈20–25 h select + ≈12 h refit + pilots; d256 ≈3–4 days per run; merged choice/E3 model one more large run | **4–6 weeks (critical path)** |
| 3. Blend-aware selection | ridge on P2 + ≈1 h full-bank checks | ~1 week, overlapped |
| 4. Production | ≈40–60 GPU-hours + 100–150 CPU core-hours of Stage C retrains, then qualifying predictions and offset re-measurement | 1–1.5 weeks |

**Total ≈7–10 weeks**, or **≈5–7 weeks** if lane 2 stops at d128 (lower
ceiling). The single GPU serialises lane 2, the other session's GPU work and
lane 4's GPU refits. d256 runs may need the GPU alone: two A2c runs once hit
110 GB. CPU work runs 2–3 jobs in parallel within the ~10-thread rule. The
week-3 stop rule (≥0.0015 dev-CV bank gain) still applies.

## Week-0 scoping result (2026-10-31)

`scripts/scoping_bank.py` → `artifacts/scoping/S0-bank-scoping.json`. It uses
a ridge proxy over the 79 prediction columns of R9-LAT-A2h (hash-verified),
with 5 user folds, cross-fitted, on the original probe, as a diagnostic.
Proxy full-bank RMSE is 0.86968 vs production 0.86687, so read shapes, not
absolute gains.

**Breadth comes from families, not columns.** Mean RMSE by number of whole
families included (random orders):

| Families | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RMSE | 0.8947 | 0.8828 | 0.8774 | 0.8768 | 0.8736 | 0.8723 | 0.8712 | 0.8704 | 0.8697 |

The 8th and 9th families still add ≈0.0007–0.0008 each. Leave-one-family-out:

| a2 | bayes | masked history | gp/local | rbm | integrated | knn | autorec | factor sgd |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00173 | 0.00127 | 0.00079 | 0.00067 | 0.00047 | 0.00024 | 0.00021 | 0.00009 | 0.00004 |

About 30 kNN and SGD-factor columns together are worth ≤0.0002. A power law
fitted to random column subsets (a + b·k^−c, c = 0.32) extrapolates to
−0.0018 at 160 columns and −0.0046 at 800. That assumes new columns are as
novel as random existing ones, which the family results contradict unless
they come from new families.

**Quality converts poorly into blend gain.** Blend gain per 1.0 of standalone gain when a
family's representative is swapped (bank without the family plus one member):

| Family | Slope | Members' standalone spread |
|---|---:|---:|
| bayes | 0.14 (3 members, noisy) | 0.003 |
| gp/local | 0.10 | 0.006 |
| a2 transformer | 0.028 | 0.013 |
| masked history | 0.025 | 0.019 |
| integrated / knn / factor sgd / rbm / autorec | ≈0 | — |

### Revised lane ranking

1. **New model families first** (≈0.0007–0.001 each in the proxy). This
   is the only lever with a demonstrated per-unit size that could sum to
   0.0041: about 4–6 new families, fewer if some are strong. The next step
   is a short audit of which families the winners or the literature used that
   are absent here, ranked by expected error-structure novelty.
2. **Quality upgrades only in high-slope families**: Bayes (MCMC) and GP/local
   nonlinear residual models. A ≈0.007–0.010 standalone gain there would be
   worth ≈0.001 in the blend.
3. **Lane 1a (PT integrated model) and plain transformer scaling (lane 2b)
   are demoted.** Their families convert standalone gain at ≈0 and 0.028, so a
   0.010 standalone improvement would add ≲0.0003. They pay only if they
   change error structure. A scaled transformer that adds a new *input*
   (choice tokens, long context) is the version worth testing, not width alone.
4. Lane 3 (blend-aware selection) and lane 4 (production) are unchanged.

**Caveat.** Slopes rest on few members and the ridge proxy; the production
blender extracts more nonlinear value from each column. These estimates rank
the lanes. They do not forecast gains.

## Lane 1 closed (2026-10-31)

User decision after L1 (+0.0000023 beyond the bank): the new-family lane is closed for good. See FAMILY_AUDIT.md.
