# Fresh reproduction: published voting features

Use the complete 130-feature `vfeat1` source recipe, freshly executed for both trainx/ProbeX and fulltrain/qual. The pinned implementation fits its statistics and latent factors on training ratings and exposes query rows through `MaskedDataset`. Qualifying outcomes remain zero.

Append all 130 features to the existing 83-predictor blender's nonlinear inputs, plus the freshly completed `mf-61` and `mf-61__knn3` predictions. Both new rating predictors also enter ridge. This is a fixed bundle; no feature or family subset is selected using outer scoring labels. The result has 85 prediction columns, 161 ridge design columns and 221 nonlinear inputs before the fold-specific ridge feature.

Keep the existing nested user folds, inner-only scaling and stopping, fixed three seeds, tree/NN recipes, and equal average. Run tree and NN sequentially to leave RAM for ongoing native training. Replay each family in a separate process, independently score saved predictions, and verify the fixed average. Published prediction vectors are excluded. This development result does not establish a qualifying score.

The prepared ridge baseline is 0.8636297008006221 versus 0.8637056490665563 before the two MF61 columns. The previous best fixed tree/NN average is 0.8621943361714148. The experiment manifest was frozen before nonlinear fitting at `experiments/upstream-voting-blend/manifest.json`. Artifacts live under `artifacts/upstream-voting-blend/`. The full native bank and original-capacity temporal reproduction continue in parallel; this experiment measures the newly available published feature block while those jobs run.
