# First qualifying result from our own training

Independently verified November 19, 2026, 06:49 UTC. **Target missed; goal active.**

| Source group | Rows | RMSE | Improvement over 0.9525 |
| --- | ---: | ---: | ---: |
| `is_test=0` | 1,408,789 | 0.8619187880311255 | 9.5098385269% |
| `is_test=1` | 1,408,342 | 0.8628067765436453 | 9.4166113865% |

Both must score strictly below 0.85725. The worse group remains 0.0055567765
above that mark. This is an adaptive audit of community-recovered outcomes.
Published Quiz/Test counts reverse the source group labels; neither group is
claimed as an official or untouched Netflix Test measurement.

All 85 predictors were trained by us. The recovered temporal vectors reproduce
their original native NPY hashes exactly. The final ridge/scalers and three
seed models per family were fitted only to known-ProbeX rows. Tree budgets
were 698/737/860; neural budgets 25/31/25, independently recomputed from actual
development receipts and their saved model hashes. No downloaded pretrained
prediction features or qualifying-label fitting/selection were used.

The original producer saved each full family, replayed its predictions,
formed the fixed equal float32-family mean, and froze the final vector before
recovering qualifying labels. A separate checker verified 29 own source jobs,
275 native outputs, the complete paired bank and actual development model
artifacts. It independently replayed both final families in separate processes,
verified the exact fixed mean, required the pinned recovered-label Git blob,
and recomputed both group scores with compensated summation. Its final report
binds 333 evidence files and explicitly sets `goal_achieved=false`.

The initial separate checker aborted because importing LightGBM and PyTorch
together loaded conflicting OpenMP runtimes. The corrected checker uses isolated
family processes, as the original numerical runner does. Three targeted tests
pass, including real trained-model replay and retagged prediction substitution
rejection. The original production models and frozen vector were preserved.

Frozen vector SHA256:
`b93a84fed0078a7ccb9d71e26dd06d56773999c0e602a0d6e1fb698a35d60c54`.
The complete checker report is
`experiments/upstream-first85-completion-audit/verification.json`; original
producer reports are under `experiments/upstream-phase-production/`.

The existing fixed continuation is running: all six additional native models
are ready for the 91-predictor development study, and the original 1,000-factor
source model has started. CFNADE48 continues remotely and CFNADE96 locally at
their full 95-epoch capacities. The matching86 qualifying queue has a guarded
restart after its original phase bank finishes. Known-ProbeX development,
not qualifying residuals, selects whether measured candidates proceed.
