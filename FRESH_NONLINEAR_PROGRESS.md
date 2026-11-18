# Fresh-only nonlinear blend: complete and independently verified

Branch: `research/fresh-nonlinear-blend`. Completed 2026-11-18 03:34 UTC (November 17 Pacific).

**New best in this fresh-only ProbeX line: 0.8644717283**, a 0.0018147335 RMSE improvement over the frozen 0.8662864618 control. The fixed equal average improves all five user folds.

| Arm | Five-fold RMSE | Change from control |
| --- | ---: | ---: |
| control | 0.8662864616 | -0.0000000002 |
| tree | 0.8645867241 | -0.0016997378 |
| nn | 0.8646594115 | -0.0016270503 |
| fixed-half-tree-nn | 0.8644717283 | -0.0018147335 |

The original double-precision control reproduced at 0.8662864618486446; its saved float32 vector scores 0.8662864616133523. The tiny difference is rounding, not a model change.

| User fold | Control | Fixed average | Change |
| --- | ---: | ---: | ---: |
| 0 | 0.8657325554 | 0.8637478062 | -0.0019847496 |
| 1 | 0.8637408905 | 0.8619351625 | -0.0018057282 |
| 2 | 0.8672154061 | 0.8654885621 | -0.0017268442 |
| 3 | 0.8676331480 | 0.8659438050 | -0.0016893430 |
| 4 | 0.8671154204 | 0.8652493809 | -0.0018660398 |

## Verification and retained artifacts

- All 35 upstream checks pass, including actual-backend label poisoning through stopping/refitting, scaler isolation, corrupted-output rejection and verifier checks.
- All 1,128,164 predictions in each family replay bit for bit from saved per-fold/seed models, ridge weights and scalers.
- Independent compensated-sum scoring agrees within 1e-12 RMSE. The fixed 50/50 average is bitwise verified.
- The average has a paired user-cluster 95% normal interval for MSE change of [-0.00330774, -0.00297399], conditional on the bank and fitted predictions. This does not adjust for historical model/recipe selection.
- Training took about 22.5 minutes wall time for the two parallel workers; tree stages sum to 17.4 minutes and neural stages to 22.5 minutes. Each family used two CPU threads. All training and sleep-prevention processes exited.
- `experiments/upstream-nonlinear/predictions.npz` contains all four float32 ProbeX vectors, with archive/content hashes in `completion.json`. It contains no labels or qualifying predictions.
- `evaluation.json`, `independent-verification.json`, `tree-replay.json`, `nn-replay.json`, family logs and all ten fold receipts preserve results. Model files and cached data remain under ignored `artifacts/upstream-nonlinear/`.

## Scope and next step

All 62 input prediction columns were freshly trained from pinned upstream source; no released prediction vectors, pretrained weights, released epoch coefficients or recovered qualifying outcomes enter these fits. This is development evidence on ProbeX, with historical base-recipe selection exposure. It is not proof of the 10% qualifying target and cannot be directly compared with original-Probe R9 results.

Keep both the frozen ridge control and this verified average. When the other laptop returns its completed bknbrx/CF-NADE outputs, verify provenance and row alignment, measure their fixed additions against the ridge control, then run a separately frozen nonlinear refit on the expanded fresh bank. Do not reuse outer-label-dependent global residuals or fit final weights on globally cross-fitted predictions.
