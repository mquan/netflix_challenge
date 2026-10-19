# Final-production readiness audit

2026-10-19; current promoted reference remains R9-LAT-A2h at 0.8668706425
development CV. This audit supports the final retraining step in
`PLAN_TO_TARGET.md`; it neither changes that score nor establishes the 10% goal.
The LC6 score experiment continues independently on the same branch.

## Verified current artifacts

`scripts/production_inventory.py` scanned 2,646 JSON records from the primary
and relevant research worktrees and verified the promoted bank against its
frozen input hashes. Results: `experiments/production-bank-inventory.json`.

| Evidence | Prediction columns |
|---|---:|
| Included in promoted bank | 79 |
| Present and hash-matched | 79 |
| Unique prediction hashes | 77 |
| Explicit matching output-digest record found | 33 |
| Matching output-digest record also contains a command | 21 |
| No explicit output-digest record found by this scan | 46 |

The last row does **not** mean those columns are irrecoverable or that training
commands do not exist. Some historical records have only paths, some selected
epoch files lack historical digests, and some recipes live in shell scripts or
git history. They need further provenance recovery. Input-hash mentions in
blender reports are deliberately not counted as evidence of how a column was
produced. A command record alone does not establish that every dependency is
available or that fixed-epoch qualifying inference is implemented.

Two duplicate prediction pairs are already in the promoted bank:

- `other-branch/joint256-snap.f32` and `artifacts/joint256-reg01.f32`.
- `other-branch/knn64-j256snap.f32` and `artifacts/r4/knn-J6-p1.f32`.

Preserve them in score comparisons: removing duplicate input columns can alter
regularization and nonlinear blending even though their values are identical.
They represent potential shared production work, not a new score claim.

## Required before a final production claim

1. Reach the development milestone under the declared promotion protocol, then
   freeze the retained bank and training settings. The working 0.86465 target
   remains calibrated from the approximate 0.0074 transfer offset.
2. Recover and verify each retained prediction's full training recipe, source,
   fixed epoch/sweep count, and dependencies, including graph/base-model inputs.
   Inventory support features and their train+probe transformations as well.
3. Use the verified train+probe data and qualifying-order mapping described
   below. Query ratings are unknown; their zero sentinels cannot choose epochs
   or produce meaningful RMSE.
4. Implement and verify fixed-budget production paths for each retained family,
   plus the full-probe-fitted blender and matching qualifying meta features.
   Rehearsals cover important families but do not prove the entire 79-column
   production pipeline is ready.
5. Run the frozen train+probe pipeline, validate prediction coverage/order,
   finiteness and [1,5] bounds, and preserve complete input/output provenance.
6. Report the result at the level the evidence supports. Qualifying labels are
   unavailable here, so a calibrated estimate cannot be called a verified
   hidden-test score or competition victory.

The inventory is a starting point for the column registry. Production readiness
is incomplete; no final retraining or qualifying-score claim is made.

## Recipe recovery checkpoint

`PR0-J6-knn64` replayed the documented J6 kNN command against the retained base
model and graph. Both predictions and support are byte-identical to the frozen
bank artifacts (maximum absolute difference zero). Evidence is in
`experiments/PR0-J6-knn64-verification.json`, with the command, input hashes and
source archive in the accompanying experiment records. This verifies inference
reconstruction; rebuilding the base model and graph on train+probe remains open.
`scripts/recover_knn_recipes.py` completed ten related fixed recipes in PR1:
seven matched exactly and three graph-256 J6 variants did not. The original
`ROUND4.md` base-model table records time decay 0.003 for that graph; filenames
omit it. PR2 replayed only those three variants with that documented decay.
All three predictions and the retained support column then matched exactly.
Both the failed initial reconstructions and corrected commands are preserved in
`experiments/PR1-knn-verification.json` and
`experiments/PR2-knn-decay-verification.json`. Output hashes were independently
rechecked. Together with PR0, eleven recipes now reproduce retained predictions
exactly, along with all four corresponding support columns used in the bank.

`scripts/recover_neural_recipes.py` recovered fixed-epoch command templates for
all 14 retained primary-worktree neural columns and all five recorded source
versions from git, with exact source SHA-256 matches. The registry is
`experiments/production-neural-recipes.json`; the source bytes are preserved in
`experiments/production-neural-sources.zip`. Eight columns also have historical
matching selected-output digests. The other six have current bank hash matches
and recorded epoch histories, without equivalent historical output digests.
Retained snapshot filenames take precedence over a later final training epoch.
Archive and member hashes were independently verified, and all five recipe and
inventory tests passed. These are recovered recipes, not verified retraining
results or a complete production pipeline.

## Production dataset prepared and verified

`scripts/production_dataset.py` created
`data/production/stage-c-v1/train-probe-qualifying.bin`: 100,480,507 labeled
train+probe ratings and 2,817,131 unlabeled qualifying queries for 480,189 users
and 17,770 movies. It recovers the original dense-user mapping by first
appearance in ascending raw movie files, preserves qualifying-file query order,
and writes both `dense-to-raw-user.npy` and `qualifying-order.npy`.

The labeled rows and qualifying queries exactly partition the original
103,297,638 implicit interactions. An independent inversion check locates all
1,408,395 original probe rows in production training, checks their labels, removes
them, and recovers all 99,072,112 original training items, dates and ratings.
User offsets, unchanged implicit arrays, and all output/source hashes also pass.
Evidence: `experiments/production-dataset-manifest.json` and
`experiments/production-dataset-verification.json`. Three fixture tests cover
mapping/order, label preservation, no overwrite, invalid dates, and missing or
duplicate queries. The development dataset remains unchanged.

Reproduce into a new output directory:

```sh
python scripts/production_dataset.py \
  --source data/processed/netflix.bin \
  --training-dir data/raw/extracted/download/training_set/training_set \
  --qualifying data/raw/extracted/download/qualifying.txt \
  --output-dir data/production/stage-c-v1
```

This is production-only data. Query ratings are zero sentinels, never labels;
existing development selection/scoring commands must not be used on them.
The full retained bank still lacks a complete production pipeline, and no final
model training has been launched.

## Fixed-budget neural adapter

`scripts/production_neural.py` supports the fourteen recovered primary-worktree
neural recipes. It verifies the production manifest and retained column hashes,
extracts the recipe's exact archived sources into an isolated directory, and
calls the archived fixed-epoch refit path directly. It bypasses the development
scorer, rejects selection data, early-selection settings and partial epochs,
and preserves raw column predictions for the final blender. There is no
qualifying RMSE in its report. The default only prepares a run; `--train`
executes the retained epoch budget.

All fourteen recipes passed preparation against the full production dataset:
`experiments/PN0-neural-preflight.json`. This confirms compatible imports,
arguments, configuration and verified data, not completed full-model training.
Two test methods cover all four distinct archived trainer versions plus archive
tampering. Each trainer ran two complete fixture epochs twice, with zero versus
poisoned query labels: trained parameters and predictions were identical.
Final-epoch checkpoint contents, query ordering and selection rejection pass.
The adapter writes predictions itself because the oldest archived trainer does
not expose the newer prediction-writing helper.

Example preparation, with a fresh output directory:

```sh
python scripts/production_neural.py \
  --column A2h-refit.refit.e12.f32 \
  --dataset-manifest data/production/stage-c-v1/manifest.json \
  --output-dir artifacts/production/A2h
```

Remaining work includes other retained model families and their dependencies,
production support features, the final blender, and full-budget training after
the development milestone and retained bank are frozen.
