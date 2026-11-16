# Fresh upstream replay: completed and independently checked

The M3 Max completed all eight planned jobs on November 16, in approximately 4½ hours. Its final training commit is `db394c1125068de57b317f99b762bbb14c0b42f2`. The first Mac received all eight fresh ProbeX vectors, verified artifact/receipt hashes and exact label ordering, independently recomputed the scores, and completed the predeclared fresh-only blend experiment. A second rescore of the saved out-of-fold vectors agrees within `3e-10` RMSE.

## The residual chains help

All fits below use only freshly trained predictions. Released vectors are excluded from the feature matrix. Evaluation uses 1,128,164 ProbeX development rows, five fixed user folds and ridge lambda 1000; each scored user's labels are excluded from that fold's meta-model fit.

| Predeclared comparison | Base-only RMSE | With chains RMSE | Change |
| --- | ---: | ---: | ---: |
| Three model families together | 0.876411386 | **0.869549365** | **−0.006862021** |
| Temporal family | 0.879267790 | 0.876616850 | −0.002650940 |
| RBM family | 0.921271903 | 0.902656742 | −0.018615161 |
| DNN family | 0.912300425 | 0.893756045 | −0.018544381 |

Every comparison improved in all five folds. The combined eight-column blend's fold improvements range from 0.006346 to 0.007069 RMSE. This supports residual chains as a useful strategy in this small, freshly trained bank; it does not establish how much they contribute to the full 188-column upstream ensemble.

## Reproduction fidelity

- **Exact prediction matches:** `dnn-24`, `dnn-24__knn3`, `dnn-24__dnn-16p50`, and `dnn-24__dnn-16p50__knn3` match their released ProbeX arrays numerically, with zero prediction difference.
- **Close numerical match:** `tsvdx4-60` has prediction RMS difference `1.49e-6`; its fresh/released score difference is approximately `1.3e-9`.
- **Not exact:** `tsvdx4-60__knns` scores 0.879095602 versus released 0.878952653 (difference +0.000142949), with prediction RMS difference 0.017859. `rbmx2-40__knn3` scores 0.903717708 versus 0.903673503 (difference +0.000044205), with prediction RMS difference 0.040524. Their discrepancies are unresolved; similar aggregate scores do not imply matching predictions.
- **No released fixture:** `rbmx2-40` scores 0.921512211. The pinned store does not contain its base prediction, so numerical reproduction cannot be checked directly.

This is a successful partial source replay, not complete numerical reproduction of every tested output. The two neighborhood discrepancies need investigation before claiming full fidelity.

## What the score means

**The fresh-only result is 0.869549, not 0.856167.** The latter remains a development-fitted ridge reference using 188 downloaded upstream prediction columns. We have freshly trained eight selected columns, with a simple cross-fitted ridge; we have not regenerated the larger model bank or its learned multi-layer blends.

These are development comparisons. Published base recipes were historically selected using development performance, so meta-level cross-fitting does not make this untouched test evidence. No qualifying outcomes were used in fresh training or these blend fits.

The earlier 79-column local-bank experiment uses the original Probe split, rather than ProbeX. Its approximately 0.8668 scores cannot be directly ranked against this 0.8695 result without aligning evaluation rows and documenting the differing training sets.

## Existing local experiment also finished

The first Mac completed all six assigned stages and verified their frozen input/code hashes and stage receipts. Original-Probe development scores were:

| Arm | RMSE | Change from baseline |
| --- | ---: | ---: |
| Baseline | 0.866870642 | — |
| Snapshots | 0.866832317 | −0.000038325 |
| Calibration | 0.866920996 | +0.000050353 |

Snapshots gave a small improvement in four of five folds; calibration worsened the aggregate score. The second computer's earlier combined-feature and ENN experiments were negative. Keep these results separate from the ProbeX source replay.

Machine-readable evidence:

- [Fresh-only blend results](experiments/reproduction-coordination/fresh-only-blends.json)
- [Independent per-model checks](experiments/reproduction-coordination/model-checks.json)
- [Verified local experiment summary](experiments/reproduction-coordination/local-ensemble/comparison.json)
- [Final remote progress and native training times](https://github.com/mquan/netflix_challenge/blob/research/upstream-reproduction/UPSTREAM_PROGRESS.md)

Both bounded experiments and the automatic verification/blend work are complete. The coordinator exited successfully; it is not launching additional training. The next experiment should be chosen using these results, with the two fidelity discrepancies kept explicit.
