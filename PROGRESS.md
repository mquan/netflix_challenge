---
status: in-progress
branch: main
timestamp: 2026-09-28T05:06:47Z
best_score: 0.8858460517
score_metric: five-fold held-out-user probe blend RMSE
---

# Netflix Prize progress checkpoint

The best recorded blend is B7 at **0.8858460517** probe user-fold CV RMSE
(0.8857544990 fit RMSE). Its ten prediction columns include the frozen N2i
128-neighbor epoch-5 column. All five held-out user folds improve over B6's
0.8863207420 CV RMSE. These are development scores: base-model epoch selection
uses probe labels, so they are not an independent test result.

Completed work includes quadratic ridge blending (B5), a 256-factor joint
checkpoint and blend (J5/B6), the 128-neighbor graph and model (G2/N2), B7,
temporal neighborhood decay (N3), and paired frequency-factor and continuation
pilots (J6/J7). Experiment manifests, logs, timestamped events, source snapshots,
and the regenerated `experiments/score_timeline.csv` preserve the evidence.
`make test` passed at this checkpoint.

Next, compare N3's temporal-neighborhood prediction with B7 under the same
five user folds. Then test whether J6 adds more to the blend than J7, freezing
each prediction column before fitting weights. The saved N2i manifest identifies
the exact checkpoint used by B7; do not substitute a later N2 artifact.
