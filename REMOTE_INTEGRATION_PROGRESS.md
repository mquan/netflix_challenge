# Remote bknbrx integration: complete and verified

Branch: `research/fresh-remote-integration`. Verified 2026-11-18T04:24:51.083822+00:00.

**New fresh-only ProbeX best: 0.8636584760**, improving the prior fixed average by 0.0008132523. All five matched folds improve.

| Arm | Before remote addition | With remote addition | Change |
| --- | ---: | ---: | ---: |
| control | 0.8662864616 | 0.8655052915 | -0.0007811701 |
| tree | 0.8645867241 | 0.8637856935 | -0.0008010306 |
| nn | 0.8646594115 | 0.8638296280 | -0.0008297835 |
| fixed-half-tree-nn | 0.8644717283 | 0.8636584760 | -0.0008132523 |

| User fold | Previous fixed average | New fixed average | Change |
| --- | ---: | ---: | ---: |
| 0 | 0.8637478062 | 0.8629253578 | -0.0008224484 |
| 1 | 0.8619351625 | 0.8611075014 | -0.0008276611 |
| 2 | 0.8654885621 | 0.8647548091 | -0.0007337529 |
| 3 | 0.8659438050 | 0.8651129045 | -0.0008309005 |
| 4 | 0.8652493809 | 0.8643984482 | -0.0008509327 |

## What was added and checked

- Remote bknbrx-100 completed in about nine minutes. Terminal and epoch 5 duplicate; terminal plus epoch 1 add exactly two unique rating columns. CF-NADE-96 supplies a resource profile, not model inputs.
- All three exported arrays passed NPZ/content/receipt-binding checks and ratings/user/item/date alignment against native local data. Sender verified all 13 native output hashes; fulltrain/qual files remain on the sender.
- The original 68 nonlinear feature columns and all old ridge design columns were independently verified bit for bit unchanged. Only two new rating features were appended. Context interactions remain restricted to the original fourteen predictions.
- The fourteen, snapshot/context and latest ridge controls reproduced within 1e-9 before refitting. Seeds, hyperparameters, user folds, nested residual fitting, stopping and clipping stayed fixed.
- All 43 tests pass. Each family replays all 1,128,164 saved predictions bit for bit. Independent compensated-sum scoring agrees within 1e-12 RMSE; the 50/50 average is exact.
- Matched average MSE change has a user-cluster normal 95% interval [-0.0015287981, -0.0012820137], conditional on the fitted bank/predictions; historical model selection is not adjusted.
- Parallel training took about 23.4 minutes, two CPU threads per family. Tree stages sum to 17.2 minutes; neural to 23.4. All owned training and sleep-prevention processes exited.
- Saved models, ridge weights, scalers and cached data remain under ignored `artifacts/upstream-remote-integration/`. All four compressed ProbeX vectors, logs, fold receipts and verification evidence are tracked under `experiments/upstream-remote-integration/`.

## Next remote assignment

The larger plan is pushed to `research/remote-temporal-push`, commit `e3daec1`, in `REMOTE_TEMPORAL_PLAN.md`. It prioritizes tsvdx5-150o, tsvdx5-300, and measured-feasible tsvdx4-4800lm with compatible neighborhood chains. Fallbacks are the small cfnade-12 preset (profile its actual 40-hidden/rank-12 settings first) and bknbrx-50. It retains the original 12-hour clock, uses complete native receipts and excludes full-ProbeX NLPP parameter search/published epoch coefficients.

These are fresh-training ProbeX development results, not official or untouched qualifying scores. No qualifying outcomes or downloaded prediction vectors entered the fits. The next fresh models must be evaluated against this stronger matched control; gains diagnosed against the earlier ridge are not assumed additive.
