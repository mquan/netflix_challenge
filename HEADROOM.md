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
