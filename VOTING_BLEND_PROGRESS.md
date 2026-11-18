# Fresh voting-feature blend: verified development improvement

The fixed tree/NN average achieved **0.8604038556527669 ProbeX RMSE**, versus the previous verified fresh-only best **0.8621943361714147**. The improvement is **0.00179048051864783** and appears in all five user folds. This is adaptive development evidence; it does not establish the qualifying target of 0.85725.

The model uses 85 freshly trained rating predictors and all 130 newly reproduced published voting features. Tree RMSE is 0.8609283193634096; neural RMSE is 0.8604849991345764. The equal tree/NN average was fixed before scoring. All ten fold receipts bind saved artifacts, both model families replayed bitwise, the equal average matched bitwise, and independent compensated summation agreed within 1e-12 over 1,128,164 rows from 462,858 users.

Against the preceding fixed average, paired MSE changes by -0.003084278503919366; its conditional user-cluster normal 95% interval is [-0.0032561739743863197, -0.0029123830334524124]. This interval conditions on historically selected banks and fitted predictions and has no selection correction.

Evidence: `experiments/upstream-voting-blend/evaluation.json`, `independent-verification.json`, `comparison-previous.json`, both family replay reports, and the five fold receipts per family. No qualifying outcomes were accessed for this experiment.

The existing paired-production watcher will refit on locally paired ProbeX/qual outputs once all 85 columns are available, using the predeclared per-seed median fold-budget rule. Current coverage is 69/85. The original-capacity temporal model and missing DNN chains remain dependencies. Two independent canonical CFNADE-96 source phases have now started with the exact verified optimization/checkpoint patch and original 320-hidden/rank-96/95-epoch preset; these are a later fresh extension, separate from the fixed 85-column production recipe.
