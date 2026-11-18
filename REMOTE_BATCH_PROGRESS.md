# Full remote-batch evaluation: running

Branch: `research/fresh-remote-batch`. Updated 2026-11-18T15:35:03.680778+00:00.

All 25 exports / 16 native receipts were checked, including pinned commands, native.out hashes and native ProbeX alignment. There are 24 unique vectors, two already integrated and 22 new. Eighteen released diagnostic fixtures were independently checked: eight bit-identical matches, six ALS/temporal matches within 7e-7 RMS, and four Hogwild differences agreeing with the remote reports. Seven exports have no separate released fixture. Released vectors are not model features.

The first diagnostic audit stopped on a legacy report-key format; its pretraining state is preserved under `experiments/upstream-remote-batch-pretraining-audit-format/`. The report adapter was fixed and regression-tested before fitting. No model/feature/recipe choice changed. All 49 checks pass.

| Fixed ridge arm | Added columns | RMSE | Change from 64-column ridge |
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

The primary nonlinear refit uses all nineteen predefined additions (83 prediction columns total). The three extra DNN intermediate parents are separate ridge-only sensitivity arms, fixed before scores. The retained nonlinear control is 0.8636584760.

| Family | User fold | New RMSE | Change from previous same family |
| --- | ---: | ---: | ---: |
| tree | 0 | 0.8615712526 | -0.0014426604 |
| nn | 0 | 0.8616515640 | -0.0014706806 |

Two CPU workers use two threads each. No choices change after partial results. Final five-fold comparison, saved-model replay and independent scoring remain required. This is fresh-only ProbeX development, not an official/untouched qualifying score.
