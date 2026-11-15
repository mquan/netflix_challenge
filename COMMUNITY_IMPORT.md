# Public predictor import pilot

Worktree: `/Users/quan/workspace/netflix_content`; branch:
`codex/content-features`. All downloads and outputs are under
`artifacts/community/`. The other session's persona run is untouched.

## Fixed candidates and protocol (before score)

Source: public `gtakacs/netflix_prize` prediction store, `preds_old/*.probe.npy`
from its original train/probe split. Fixed three candidates from different
families: `als8-8`, `cfnade_rmsew010_ep38`, and
`tsvdx5-400__nlpp__knn3x`. Download only their Probe arrays; do not read
the repository's released qualifying labels or use its qualifying scores.

Verify remote index MD5 and shape. Reconstruct the original raw-user to
processed-user mapping by replaying training-file order, then map each
original `probe.txt` pair to the processed Probe row. Require a bijection,
equal ratings from training files, and 1,408,395 finite values per column.

First report standalone Probe RMSE and a fixed one-weight five-user-fold
screen against R9 for each column. For a full-rating predictor, the screen
fits a weight on `(candidate - R9)` and predicts
`clip(R9 + weight * (candidate - R9), 1, 5)`. This is a diagnostic: R9 out-of-fold
predictions were trained under a shared probe selection history, and the
screen cannot establish independent test gain. The diagnostic gate for a
full 79-column R9 LightGBM/NN2 comparison is ≥0.0001 aggregate RMSE gain
and improvement in at least four of five folds. If multiple candidates pass,
test the largest diagnostic gain first without retuning. Final promotion
would require all five folds better in the matched full comparison.

## Stage-one results (2026-11-15)

All three remote `.npy` files match the public store's MD5 entries and contain
1,408,395 finite float32 predictions. Source ingest emits Probe rows in
numeric movie-ID order, whereas original `probe.txt` lists movie blocks
lexicographically. The converter reconstructs the source order and verifies a
bijection of all processed pairs plus equality of their training-file ratings.
The initial uncorrected ordering produced implausible standalone scores and
was discarded before the fixed blend comparison.

| Candidate | Standalone Probe RMSE | One-weight R9 gain | Better folds | Gate |
|---|---:|---:|---:|---|
| ALS | 0.96012 | 0.00001446 | 4/5 | Fail: small gain |
| CF-NADE | 0.89755 | 0.00000089 | 2/5 | Fail |
| Integrated time-factor/neighborhood | 0.88035 | **0.00013076** | **5/5** | **Pass** |

The integrated candidate's fixed-screen R9 score is 0.86673988 versus
0.86687064. Its user-cluster normal-approximation standard error for the RMSE
gain is about 0.00001488. This interval does not account for the source's
prior Probe-based model selection or our own search history. The full matched
79+1-column LightGBM/NN2 comparison has not run; the household-persona blend
currently uses three workers and most available RAM. The candidate is not
promoted. Detailed hashes and folds are in
`experiments/COMMUNITY-fixed-three-summary.json`.
