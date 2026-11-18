# Nonlinear blending of our freshly trained bank

Branch: `research/fresh-nonlinear-blend`. Local laptop only; the second laptop
owns new-family training on `research/fresh-models-12h`.

Freeze the existing 62 fresh prediction columns and training/query context.
Reproduce the latest **0.866286461848646** five-user-fold ProbeX control before
training. Downloaded prediction vectors, published epoch weights, and recovered
qualifying outcomes are excluded. Development labels may be read from the
verified row-aligned development artifact; these are not prediction features.

## Fixed comparisons, recorded before results

1. Existing contextual ridge control, lambda 1000, unpenalized intercept.
2. Ridge plus a LightGBM residual correction: three seeds 2026/7/13, at most
   2000 rounds, learning rate .03, 31 leaves, minimum leaf 1000, feature fraction
   .9, bagging fraction .8 every five rounds, stopping patience 75.
3. Ridge plus a small neural residual correction: the same three seeds, two
   64-unit GELU layers, dropout .2, AdamW learning rate .0003, weight decay .01,
   batch 8192, at most 40 epochs, stopping patience four.
4. Fixed equal average of the clipped tree and neural predictions. No fitted
   second-stage weights or selection of an averaging coefficient.

No hyperparameter or feature sweeps follow outer-fold feedback in this round.
Record every result, including improvements too small for past promotion gates,
and retain the control. A positive development gain is not a 10% target claim.

## Isolation through both learning stages

Use the existing outer user folds with seed 20261116. The inner validation
users are those with `user_folds(user*7+19)==0`, restricted to outer training.
Each inner stopping fit has its own ridge and, for the neural network, its own
scaler fitted only on inner-training users. Include zero correction as a
stopping candidate; a worse inner learner need not be refitted.

After fixing each seed's stopping budget, refit its ridge and nonlinear
correction on all outer-training users. The full neural scaler likewise uses
only outer-training users. Residual targets are based on these fold-local
ridges, not a global out-of-fold vector whose training predictions may depend
on the scoring fold. Each nonlinear model sees prediction disagreements,
unclipped component features, five fixed context features and its fold-local
ridge prediction. Context interactions in the ridge remain restricted to the
original fourteen predictions. Rating inputs clip to [0,6]; components retain
native values. Output clipping remains [1.05,4.95].

The source base models have historical development-selection exposure. This
protocol prevents new blender leakage; it cannot make the base bank an
untouched or selection-adjusted test.

## Execution and deliverables

`scripts/upstream_nonlinear.py` freezes verified prepared arrays, input/code
hashes, package versions and recipe before training. LightGBM and PyTorch run
in separate CPU processes with two threads each to avoid their historical
OpenMP runtime collision. There is no base-model training or GPU requirement.
Peak RAM and timing are checked after startup; do not duplicate a live runner.

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
.venv/bin/python -u scripts/upstream_nonlinear.py
```

Run under a detached process with macOS sleep prevention. Family logs and
per-fold/seed stopping budgets live in `experiments/upstream-nonlinear/`.
Models, fold-local ridge weights/scalers, prepared data and predictions live in
ignored `artifacts/upstream-nonlinear/`. Each completed fold has a receipt;
relaunch verifies its output hashes before skipping it. Partial folds refuse
silent overwrite and require an explicit preserved recovery. Do not change
the frozen recipe/code to resume an existing experiment.

Tests must exercise both actual backends with outer-label poisoning, including
stopping and refitting, and compare fold-local ridge with direct fitting.
Verify that changing excluded features cannot affect fitted scaler parameters.
The final receipt reports float32-rescored overall/fold scores, paired changes
from the control, and prediction hashes. All three comparisons run to completion
regardless of another arm's score; runtime failures stop the runner and preserve
artifacts. Further claims require a separate production/refit audit.
