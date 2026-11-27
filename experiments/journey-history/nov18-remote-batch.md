# Full remote batch: complete and independently verified

Branch: `research/fresh-remote-batch`. Verified 2026-11-18T15:55:52.842432+00:00.

**New fresh-only ProbeX best: 0.8621943362**, improving the retained 0.8636584760 average by 0.0014641399. Every matched user fold improves.

| Arm | Previous 64-column bank | New 83-column bank | Change |
| --- | ---: | ---: | ---: |
| control | 0.8655052915 | 0.8637056490 | -0.0017996426 |
| tree | 0.8637856935 | 0.8623975890 | -0.0013881044 |
| nn | 0.8638296280 | 0.8622766094 | -0.0015530186 |
| fixed-half-tree-nn | 0.8636584760 | 0.8621943362 | -0.0014641399 |

| User fold | Previous fixed average | New fixed average | Change |
| --- | ---: | ---: | ---: |
| 0 | 0.8629253578 | 0.8614679246 | -0.0014574332 |
| 1 | 0.8611075014 | 0.8594693947 | -0.0016381068 |
| 2 | 0.8647548091 | 0.8630706889 | -0.0016841202 |
| 3 | 0.8651129045 | 0.8638361085 | -0.0012767960 |
| 4 | 0.8643984482 | 0.8631317686 | -0.0012666796 |

## Remote result audit

- Independently checked all 25 compressed exports, all sixteen pinned job commands/receipts, transferred native.out hashes, raw standalone scores and ratings/user/item/date alignment.
- There are 24 unique vectors; two bknbrx-100 vectors were already integrated and were verified rather than appended again. Nineteen new vectors form the primary bank. Three extra DNN parents are separate predeclared ridge-only sensitivity arms.
- Eighteen available released fixtures were independently checked: eight bit-identical matches; six ALS/temporal matches within 7e-7 RMS; and four neighborhood differences consistent with the remote Hogwild reports. Seven exports lack a separate released fixture. Reference arrays were read only after freezing fresh prepared features and never became fit inputs.
- All source arrays, NPZ/content/receipt hashes and source pin are retained in remote-verification.json. Native fulltrain/qual files remain on the originating machine; sender hash verification is distinguished from receiving-side checks.

## Fixed family screens

| Ridge arm | Added columns | RMSE | Change from retained ridge |
| --- | ---: | ---: | ---: |
| control | 0 | 0.8655052917 | +0.0000000000 |
| temporal | 4 | 0.8645096694 | -0.0009956223 |
| neighborhood | 5 | 0.8654334354 | -0.0000718564 |
| matrix_factorization | 2 | 0.8652674055 | -0.0002378862 |
| als | 2 | 0.8653417827 | -0.0001635090 |
| dnn | 6 | 0.8650799876 | -0.0004253042 |
| main | 19 | 0.8637056491 | -0.0017996427 |
| intermediates | 3 | 0.8654516524 | -0.0000536393 |
| extended | 22 | 0.8636936148 | -0.0018116769 |

Every noncontrol ridge arm improves all five folds. The main nineteen-column combination improves ridge by 0.0017996427; adding the three intermediate parents beyond that yields only 0.0000120343 in the separate ridge sensitivity. No outer-score-based selection changed the primary nonlinear bank.

## Verification and execution

- All 49 checks pass (45 upstream and four supervisor), including label poisoning through actual stopping/refit backends, transfer corruption checks, legacy-report regression coverage and intercept-preserving family comparisons.
- All 1,128,164 predictions in each nonlinear family replay bit for bit from saved models/ridge weights/scalers. Independent compensated-sum scores agree within 1e-12 RMSE. The fixed equal average is exact.
- All nine ridge screen vectors are hash-checked and independently rescored; float32 scores agree within 1e-8 of the reported double-prediction values.
- The original nonlinear features, original ridge columns and retained ridge predictions are independently verified exactly unchanged. Only the nineteen predefined fresh rating features were added.
- Matched-average MSE change has a conditional user-cluster normal 95% interval [-0.0027098456, -0.0023439342]. It does not adjust for historical base-recipe selection.
- Parallel training took about 24.1 minutes. Tree stages sum to 18.0 minutes, neural to 24.1; two CPU threads per family. All owned training and sleep-prevention processes exited.
- The first pretraining fixture audit stopped on a legacy ep01/ep05 report-key schema. The failed preparation, code snapshot and cache were preserved, the explicit report adapter was regression-tested, and the unchanged scientific recipe was re-frozen before any nonlinear fit.
- Models/caches remain under ignored artifacts/upstream-remote-batch/. All four final ProbeX vectors are compressed in experiments/upstream-remote-batch/predictions.npz; completion.json binds them, fold receipts, independent checks, audits and tests.

## Scope and remaining work

This is a fresh-training ProbeX development result using pinned upstream implementations and locally fitted nonlinear blends. Downloaded prediction vectors, pretrained weights, published epoch coefficients and qualifying outcomes did not enter these fits. Historical base-recipe selection remains. It is not a verified 10% qualifying result, and original-Probe R9 scores use a different population.

All remote results from f130172 are now accounted for. Larger 300/400/4800-factor temporal jobs and CF-NADE were not trained in this batch. The remote handoff incorrectly generalized the 32-GiB host memory rejection to the 96-GiB host: source-derived lower bounds for 300/400/4800lm are about 49.7/65.0/35.3 GB, before overhead, so actual feasibility still requires measurement. The original remote window has ended; do not launch new remote work without an extended availability window.

The next direct target check requires a separately frozen production blend/refit and the corresponding fulltrain/qual prediction bank, whose vectors are currently distributed across the two machines. Do not infer a qualifying score by subtracting the old original-Probe offset from this ProbeX result.
