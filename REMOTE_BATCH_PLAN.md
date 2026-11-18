# Evaluate the completed remote batch against our fresh-only control

Branch: `research/fresh-remote-batch`. Remote completion `f130172` is merged.
The retained control is the verified 64-column fixed nonlinear average:
**0.8636584760350996** ProbeX RMSE. Its corresponding ridge is
0.8655052917304086. Preserve both and all earlier artifacts.

## Fixed protocol before reading ensemble results

Verify every entry in the 25-export index, all sixteen native receipt bindings,
the exact pinned command for each job, the transferred native.out files, and
all four ProbeX metadata file hashes. Independently recompute each raw
standalone RMSE. Deduplicate the terminal/epoch-5 bknbrx-100 pair by content,
and confirm its two unique vectors exactly match those already integrated.
There are 22 new unique vectors; never append the old two again.

Run these fixed ridge comparisons against the retained 64-column ridge:

- Temporal: tsvdx5-150o terminal and kept epochs 6/16/24 (four columns).
- Neighborhood: bknbrx-50 terminal and kept epochs 5/9/17/21 (five).
- MF: mf-60 and its knn3 chain (two).
- ALS: als8-8 and its knn3 chain (two).
- DNN: the six predefined residual-chain endpoints in the remote handoff.
- Main combination: all nineteen columns above (83 prediction columns total).
- Intermediate-parent sensitivity: the three fresh DNN parents absent from
  the released bank, separately, then together with the main combination
  (86 total). These two sensitivity arms are **ridge-only** in this round.

No epoch or bundle selection follows outer-fold feedback. Refit the **main
83-column bank** using the same three-seed tree and neural residual recipes
and their fixed 50/50 average, regardless of the signs of the ridge screens.
All original nonlinear feature columns and ridge/context columns remain
unchanged; append the nineteen new clipped rating features centered at 3.6.
The intercept remains last and unpenalized. Keep lambda 1000, five user folds,
original fourteen-only context interactions, and output clip [1.05,4.95].

Use the tested upstream_nonlinear fitting engine through a separate adapter;
do not edit old frozen scripts/manifests. Outer scoring users' labels must be
excluded from ridge residual fitting, inner stopping, scaler fitting and all
nonlinear refits. Zero correction remains an inner stopping candidate.

## Independently check the remote's reported reproduction

Freeze prepared **fresh** features before accessing released reference arrays.
Then hash-check the locally pinned diagnostic fixtures, independently rescore
their differences from the fresh exports, and check agreement with the remote
fixture reports. Report unavailable fixtures and expected Hogwild differences;
do not substitute downloaded vectors. Return diagnostic numbers only, never
reference arrays to the training path. Native fulltrain/qual files remain on
the sender, where their hashes were verified; do not claim local rehashing.

The remote handoff's statement that all large temporal recipes exceed 96 GB
is not supported by our source-derived bounds: 300/400/4800lm lower bounds
are about 49.7/65.0/35.3 GB before overhead. Those jobs were **not trained** in
this run. Their actual time/peak RAM remain unknown and are not new results.

## Execution and deliverables

Run upstream and remote-supervisor checks, then prepare and launch
`scripts/upstream_remote_batch.py`. Keep one tree worker and one NN worker,
each two CPU threads, in separate processes. Save per-fold receipts, models,
ridge weights/scalers, immutable code/input/version hashes and all results.
Do not duplicate active workers or silently overwrite partial folds.

Reports: `experiments/upstream-remote-batch/`; cached arrays/models:
`artifacts/upstream-remote-batch/`. After all five folds, require bitwise saved
model replay, independent compensated-sum scoring, exact fixed-average checks,
paired fold changes against the 0.86365848 control, and conditional user-cluster
uncertainty. Publish compressed predictions, logs and completion evidence to
this branch, including negative family/sensitivity outcomes.

These are exploratory ProbeX development comparisons on historically selected
source recipes, not official/untouched Test scoring or proof of the 10% target.
Every model feature is a freshly trained output; source reuse and historical
recipe selection are disclosed.
