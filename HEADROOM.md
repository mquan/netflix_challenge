# Headroom map (STRATEGY.md, part A)

Started 2026-10-31. Tool: `scripts/headroom_map.py`. Lower RMSE is better;
negative Δ is a gain.

## Method

1. Fit the base predictor out of fold by user (5 user folds, ridge).
2. For each **information source**, pool residuals of the target row's group,
   always leaving the target row out. Groups: user, user×month/week/day,
   movie, movie×month/week/day, calendar day. Features are the pooled sum over
   (n+k) for k ∈ {1, 5, 25}, an indicator for n>0, and log(1+n).
3. Add the features to the fixed ridge gate (fit on folds 2–4, pick the
   penalty on fold 1 from {1, 10, 100, 1000}, refit on 1–4, evaluate on fold 0).
4. **Placebo:** the same features with rows reassigned to random groups of
   the same size distribution. That is the floor for adding columns at all.

These are **oracles**: the features read other held-out labels of the scored
set. They estimate how much error is *shared within a source*, an approximate
ceiling for any model of that source. They are not deployable predictors.

## Results

| Source (oracle Δ RMSE) | H1: P2, bank11 ridge | H2a: probe, core9 **a** (no P2) | H2b: probe, core9 **b** (Stage C analogue) | H3: probe, **promoted R9-LAT-A2h blend** |
|---|---:|---:|---:|---:|
| user (all other held-out rows) | −0.00371 | −0.00509 | −0.00380 | **−0.00318** |
| user × month | −0.00332 | −0.00455 | −0.00347 | −0.00298 |
| user × week | −0.00301 | −0.00403 | −0.00303 | −0.00260 |
| user × day | −0.00292 | −0.00383 | −0.00286 | −0.00245 |
| movie | −0.00013 | −0.00039 | −0.00032 | −0.00007 |
| movie × week | −0.00005 | — | — | −0.00003 |
| calendar day | −0.00005 | −0.00016 | −0.00014 | −0.00001 |
| placebo (user-day sizes) | +0.000003 | +0.000008 | +0.000007 | +0.000006 |
| all user sources | −0.00391 | −0.00570 | −0.00431 | −0.00364 |
| all movie sources | −0.00016 | −0.00054 | −0.00046 | −0.00010 |
| everything | −0.00407 | −0.00621 | −0.00476 | −0.00375 |

Artifacts: `artifacts/headroom/H1-p2-oracle.json`,
`H2-core-{a,b}-probe-oracle.json` and `H3-R9-LAT-A2h-probe-oracle.json`
(the promoted out-of-fold blend, `avg_rmse` 0.8668706425). H2 and H3 read
original-probe labels **for diagnosis only**; no model, epoch or setting is
selected from them.

## Reading

1. **The remaining shared error is user-level and current.** It is spread
   across the user's days, not confined to a session (user ≈ −0.0032 vs
   user×day −0.0025 on the promoted blend). The movie side, the target of E2
   (other raters as context), has almost no oracle headroom: ≤0.0001 on the
   promoted blend. E2 should be deprioritised unless its mechanism is user-side.
2. **Stage C harvests only about a quarter of it.** Retraining with the
   user's recent labelled ratings cuts the user-level oracle from −0.0051 to
   −0.0038 (core9 a → b). The rest is shared among the held-out rows
   themselves, which no labelled data can reach.
3. **Implication for the goal.** On the promoted blend the user-level oracle
   is ~0.0032–0.0036 on dev CV. Scaling by the b/a ratio (~0.75) for the
   overlap with the Stage C offset gives ~0.0024 after Stage C, about equal
   to the 0.0022 gap. Reaching 10% therefore needs a model that infers most
   of a user's *current* shift from unlabelled information about the held-out
   rows (which movies, which dates, how many). A partial capture of a
   0.0024 ceiling will not be enough unless it is a large fraction.
4. Next measurement (H4): how much of that user-level shift is predictable
   from information available at test time.

## H4: label-free proxies (deployable features) — nothing left

`scripts/headroom_proxies.py`: LightGBM on the cross-fitted residual with
test-time features only. These are held-out set sizes, the mean prediction of
the user's other held-out movies, the gap since the last training rating,
recent-minus-overall training mean, user mean/std/count, unlabelled implicit
choices (total and same-day) and popularity. Fit on folds 2–4, early-stopped on
fold 1, evaluated on fold 0.

| Base | Δ RMSE |
|---|---:|
| P2, bank11 ridge | −0.00081 |
| original probe, promoted blend | **−0.00003** (95% ΔMSE interval includes 0) |

The production blender already extracts everything these features know. The
user-level shift is not predictable from counts, gaps, drift or popularity.

## H5: N1 is orthogonal to the user-level shift

On P2, adding the N1 saved epoch-13 column to bank11 improves the base by
0.00121, matching N1's recorded gate (−0.00121 beyond A2h, >10 cluster SE). But
the user-level oracle is unchanged: −0.00369 with N1 vs −0.00371 without. N1's
choice-history gain is row-specific information and is real and deployable.
The shared user shift is a separate, untouched source.

## H6: probe siblings carry the shift — a deployable lever

In the real data the probe and qualifying sets are a random split of the same
held-out pool. A qualifying row's user therefore has **labelled sibling**
probe rows from the same period, and dev-CV models never trained on them.
`scripts/sibling_pooling.py` rehearses this on the original probe:
- split each user's probe rows at random into half A (labelled) and half B
  (stand-in for qualifying);
- pool the promoted blend's honest residuals over A (user, user-week and
  user-day);
- fit coefficients on A by leave-one-out within A, reading no B label;
- apply them to B.

| Half B (704k rows) | RMSE |
|---|---:|
| promoted blend (dev CV) | 0.86598 |
| + pooled half-A sibling residuals | 0.86412 |
| **Δ** | **−0.00186**, all five folds (−0.0017 … −0.0022), 95% ΔMSE interval [−0.0035, −0.0030] |

Round 13's P2 → probe residual transfer was dead (−0.00005). P2 is ordinary
training ratings near probe dates, not siblings from the held-out pool. The
shift belongs to the held-out period itself.

**Open question, now the most important one in the project.** Stage C retrains on
the probe, the siblings themselves, which share days with the qualifying
rows. The models' daily user terms may already absorb much of this signal.
That would also mean the true Stage C offset differs from the 0.0074 measured
with P2, which is not from the held-out pool. Next: a **sibling-split Stage C
rehearsal**. Retrain on T + half A, score half B, and compare dev,
dev + pooling, retrained, and retrained + pooling.

## A7: sibling-split Stage C rehearsal — fixed protocol (2026-10-31, before any score)

**Data.** `artifacts/sibling/T-plus-A.bin` (`scripts/sibling_split_dataset.py`):
training is T plus probe half A (703,524 rows, same split as H6), and the
probe is half B (704,871 rows). Implicit arrays are unchanged.

**Columns.** The K1 family first, because it is CPU-only:
- **dev:** the existing Phase 0 b columns (B-K1, B-knn256td + support,
  B-knn64), trained on T only, restricted to B (and to A for fitting);
- **stageC:** `scripts/a7_sibling_chain.sh build/netflix-ordinal`, which
  retrains on T + A with the Phase 0 recipes and scores B.

**Arms on half B** (blender fit on A only, then applied to B, as the real
pipeline fits on probe and applies to qualifying):
1. dev columns;
2. dev + sibling pooling (coefficients fit on A by leave-one-out, as H6);
3. stageC columns, with the same A-fitted blender;
4. stageC + sibling pooling with the *same A-fitted* coefficients (the
   deployable variant);
5. diagnostic only: stageC + pooling with coefficients refit on B by user
   folds. This gives the ideal shrink factor and selects nothing.

**Reported.**
- Sibling Stage C offset (1 − 3), compared with the P2-measured 0.0074 for this family.
- Pooling gain on top of Stage C (3 − 4), with folds and a user-cluster interval.

**Decision.** If arm 4 beats arm 3 by ≥0.0005 on all five folds, extend the
rehearsal to the Bayes and RBM columns, then the full bank, and adopt sibling
pooling as a production step. If not, the shift is already harvested by
Stage C; record the sibling offset as the better estimate of the Stage C
offset either way.
