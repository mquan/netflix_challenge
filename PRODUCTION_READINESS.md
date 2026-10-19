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
3. Construct train+probe labeled data and qualifying queries with a reversible
   mapping to original submission order. Raw qualifying and training files are
   present. Their original dense-user mapping must be respected. Query ratings
   are unknown; dummy labels cannot choose epochs or produce meaningful RMSE.
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
`scripts/recover_knn_recipes.py` is running ten related fixed recipes, accepting
only exact output hashes and recording mismatches without parameter searches.

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
