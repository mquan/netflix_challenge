# Remote branch consolidation (2026-10-11)

All fetched `origin/*` research branches were merged into `main`. Each original
branch remains available. Scores below are original-Probe development scores;
they are not hidden Quiz or Test scores. `ROUND8.md`, `ROUND9.md`, `ROUND10.md`,
and the cited experiment manifests contain run details and fold results.

| Remote branch | Method or contribution | Recorded result |
| --- | --- | --- |
| `moonshot-local-kernels` | Local Gaussian-process and residual-tree work | Ancestor already present on `main`; C5 0.8759027924 and later results in `RESULTS.md` |
| `rbm-and-blend-improvements` | Conditional RBM and ensemble improvements | Ancestor already present; B4 0.8866646513 in `RESULTS.md` |
| `score-iteration-2` | Joint models, residual kNN, AutoRec, LightGBM | Ancestor already present; B6 0.8804020935 in `RESULTS.md` |
| `round4-combined` | Expanded model bank, round 8 sequence and Bayes work, round 10 B5 and kNN features | R8-A2h-NN2 0.8670984524; R10-B5-NN2 0.8668212 failed the all-fold promotion rule; R10-KNNF has no final blend score |
| `round9-breakthrough` | K1 latent and bias features, Bayesian frequency pilot, retrieval screen | R9-LAT-A2h 0.8668706425, promoted after all five folds improved |
| `paper-implementations` | BAR paper port, controlled reference experiments, push checks | BAR2 Probe RMSE 0.9741851128; did not improve the ensemble |
| `bet-a1-item-twin` | Item-side masked-history model and tests | Implementation only; no completed Probe score on this branch |
| `bet-a2-sequence` | Masked date-ordered sequence encoder and tests | Implementation only on this branch; later A2h ensemble result above |
| `bet-b-bayes-timesvdpp` | Bayesian timeSVD++ implementation and tests | Implementation only on this branch; later B5 screen above |
| `bet-c-user-knn` | User-user residual kNN implementation and tests | Implementation only; no completed Probe score on this branch |

Merge resolution kept the newer item-frequency extension of the Bayesian
sampler and the retrieval-enabled sequence implementation, which include the
earlier branch functionality. The two branches' conditional RBM code used
different checkpoint layouts with the same format magic. The `main` RBM
implementation and its separate visible-context loader were retained; the
paper branch's alternate implementation remains in that branch's history.

The highest mean score in this table is not automatically the promoted score.
R10-B5-NN2 improves the mean by about 0.00005 but worsens folds 0 and 1.
R9-LAT-A2h remains the strongest result under the recorded all-five-fold rule.
