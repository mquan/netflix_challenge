# Matching tree/context NN: complete verified development result

Our best fully verified fresh-training development RMSE is now
**0.8600300601883686**, improving the preceding **0.8600637147575518** by
**0.0000336545691833**. The fixed equal mean combines the newly fitted
86-predictor tree with the preceding verified 86-predictor context NN.
No downloaded pretrained prediction features or qualifying outcomes enter
this experiment.

All five original user folds improve. All 15 saved trees replay bit-for-bit;
the preceding15 neural fits and their OOF prediction vector are reused
unchanged, with explicit originating receipts rather than new fitting claims.
The independent scorer verifies the fixed float32-family average and all
1,128,164 known-ProbeX rows from 462,858 users.

| Fold | Preceding fixed mean | Matching fixed mean |
| --- | ---: | ---: |
| 0 | 0.8593164235 | 0.8592639387 |
| 1 | 0.8575002797 | 0.8574717065 |
| 2 | 0.8610794494 | 0.8610541720 |
| 3 | 0.8617462498 | 0.8617380551 |
| 4 | 0.8606817791 | 0.8606281094 |

The matching tree alone scores 0.8607727186827031; the unchanged NN alone
scores 0.8600986905055020. Per-seed median tree budgets are 809/793/875, derived
from the five actual nested tree receipts. The existing NN budgets remain
19/20/20. Training configuration and user-fold assignments are unchanged.

The direct paired MSE change is -0.0000578890149506. Its conditional user-cluster
95% interval is [-0.0001021175180971,-0.0000136605118040]. This interval conditions
on the historically selected bank and fitted models; it has no selection
correction and is not independent Test evidence.

Evidence under `experiments/upstream-cf12-tree/`:

- `evaluation.json`: all arms, fold scores and final vector hashes.
- `tree-replay.json`: all five saved-tree replays pass bit-for-bit.
- `independent-verification.json`: compensated full-row scoring and fixed mean.
- `comparison-previous.json`: row identities, unchanged neural vector, all
  tree/neural checkpoint hashes, independent paired comparison and interval.
- `tree/0.json` through `tree/4.json`: actual nested model receipts and budgets.

The complete development gate in
[CFNADE12_MATCHING_PRODUCTION_PLAN.md](CFNADE12_MATCHING_PRODUCTION_PLAN.md)
now passes. The production adapter has been implemented, tested and queued;
its final fits have not started. The existing first85 production queue
still waits for the temporal model's complete25-epoch paired outputs. AEX64
has completed its first20-epoch phase and started the qualifying phase.

The goal remains independently verified RMSE below 0.85725 on both
community-recovered qualifying groups. Those groups are not official or
untouched Netflix Test evidence. No qualifying blend score is available yet.
