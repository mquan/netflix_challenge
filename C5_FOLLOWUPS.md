# Follow-up screens on compact C5

C5's 0.8759027924 score uses the original Netflix probe and five held-out
user folds. These screens keep its saved prediction and input bank fixed.
All are development comparisons because predictor checkpoints and research
directions have already been selected using probe labels.

| Candidate | Standalone or CV RMSE | Combination with C5 | Decision |
| --- | ---: | ---: | --- |
| GP on J2 factors, noise 0.1 | 0.8911784 | 0.8758998, one cross-fitted weight | Redundant with the J3 GP |
| GP on J2 factors, noise 1 | 0.8907672 | 0.8759026, one cross-fitted weight | Redundant |
| GP on J3 implicit factors, noise 0.1 | 0.8949654 | 0.8758791, one cross-fitted weight | Too small for a full tree refit |
| GP on J3 implicit factors, noise 1 | 0.8950847 | 0.8758952, one cross-fitted weight | Too small |
| Residual item kNN on J3 | 0.8950598 | 0.8759029, one cross-fitted weight | No incremental signal |
| Residual item kNN on temporal baseline | 0.9173705 | 0.8759014, one cross-fitted weight | Too small |
| Nested 17-column neural blend | 0.8777674 CV | 0.8763285, fixed 50/50 average | Worse than C5 |

The neural blend fitted each outer fold's ridge, normalizer and network using
other users, with its epoch chosen on an inner user-disjoint split. Its fixed
50/50 average with the C5 tree preserves held-out-fold exclusion and requires
no second-stage weight fit. The complete neural fold and combination reports
are `experiments/NN-C5-report.json` and `experiments/NN-C5-combination.json`.
The GP, kNN and one-weight screens have their source manifests and per-fold
weight reports under `experiments/`.

These cheap variants do not explain the remaining gap to R5 or 0.855. The
conditioned RBM is being evaluated at a fixed later checkpoint. Beyond that,
the next high-upside work is an independently trained predictor family or a
larger rebuilt ensemble bank.
