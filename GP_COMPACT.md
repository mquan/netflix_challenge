# Local GP added to the reproducible compact bank

The compact 16-column C4 ensemble scored 0.8783242000 development CV on the
original Netflix probe. A persistent plus 30-day local Gaussian-process
residual predictor was regenerated from the retained 128-factor J3 joint
checkpoint. It uses the 32 most similar movies in each user's training history
and solves the local kernel regression with noise 0.1. Probe ratings are never
inputs to this predictor. The run and four output hashes are in
`experiments/GP-C4-temporal-full.json` and
`experiments/GP-C4-temporal-outputs.json`.

The GP scores 0.8922935916 alone. Adding its noise-0.1 prediction as a
seventeenth column to the *same* C4 residual-tree blend gives:

| Fold | C4 RMSE | C5 RMSE | Improvement |
| ---: | ---: | ---: | ---: |
| 0 | 0.8794196 | 0.8768209 | 0.0025988 |
| 1 | 0.8778140 | 0.8754441 | 0.0023699 |
| 2 | 0.8751791 | 0.8726245 | 0.0025546 |
| 3 | 0.8796260 | 0.8772637 | 0.0023623 |
| 4 | 0.8795929 | 0.8773721 | 0.0022207 |
| **All** | **0.8783242** | **0.8759028** | **0.0024214** |

All sixteen C4 input hashes match exactly in the C5 blender report at
`experiments/GP-C5-blend.json`. Independent float32 scoring gives
0.8759027924 RMSE, matching the blender's 0.8759027924. The saved C5 column
has SHA-256 `3e3c8e83313d18efae133de7e2e54652eb676a5e361e8c428910fa077791f562`.
This was the best reproducible local result at the C5 checkpoint, 0.0010934 above historical R5's
0.8748094089. The R5 input bank is missing, so that benchmark cannot be
refit locally. Both scores are development estimates; repeated original-probe
research choices and base-model epoch selection are not removed by the five
user folds used for the tree blender.

The next test is a GP on the different J2 factor geometry. It should be judged
by incremental C5 gain under the same folds, then by the full tree blender if
the cheap cross-fitted weight screen shows distinct information.

## J2 geometry screen

The same full-data GP was run on the retained J2 128-factor model. Its noise
0.1 and 1 variants score 0.8911784 and 0.8907672 alone, respectively. Despite
stronger standalone scores than the J3 GP, a cross-fitted linear weight on C5
improves by only 0.0000030 and 0.0000002. The near-zero weights are consistent
across all five user folds. This is insufficient incremental signal to justify
another full tree blend. The source run and screens are recorded in
`experiments/GP-J2-temporal-full.json` and
`experiments/GP-J2-C5-screen-0.json` / `GP-J2-C5-screen-1.json`.

## Implicit-factor geometry screen

Replacing J3's explicit movie-factor kernel with its implicit-feedback movie
vectors gives 0.8949654 standalone RMSE at noise 0.1. Its matched C5 linear
screen improves by only 0.0000237; noise 1 improves by 0.0000076. Full-data
run and output hashes are in `experiments/GP-J3-implicit-full.json` and
`experiments/GP-implicit-outputs.json`. The full tree blend is deferred because
neither alternative shows enough new signal. The next effort should return to
an independently trained predictor family instead of repeatedly changing this
GP's factor geometry.

## GP support features

The GP now also exports three label-free query features: sum of selected
neighbor covariances, strongest covariance, and effective neighbor count
`(sum weights)^2 / sum(weights^2)`. On the full data, its regenerated
prediction column has the **exact same SHA-256** as C5's GP column, so this
experiment isolates the support features. Numeric and probe-label-isolation
tests pass.

C6 refits C5's same seventeen raw prediction columns and tree settings, adding
only those three support features. The original seventeen input hashes match
C5. C6 scores **0.8755711017** development CV, improving C5 by
**0.0003316907**. All five folds improve (0.00028–0.00044). Independent
float32 scoring gives 0.8755711017. The saved C6 column has SHA-256
`fd30e73c93e0c6033709f3e1d703c81b078ea85c7f3c897cee91b6328c4ff243`.
The full comparison and support-output hashes are archived in
`experiments/GP-C6-support-blend.json` and
`experiments/GP-C5-support-outputs.json`. C6 was the strongest reproducible
local result at this checkpoint, 0.0007616932 above historical R5. The remaining distance to
0.855 on this development split is 0.0205711017.

## kNN graph support

Residual kNN predictions on J3 and on the temporal baseline add essentially
no signal to C6 in the quick screens. Their support files, however, are
identical because the support is the sum of available graph-edge weights and
does not depend on the residual base. Adding this *one unique* support feature
to C6's unchanged seventeen raw predictions gives **0.8754457423** (C7a),
improving all five folds by 0.00009–0.00018. Independent float32 scoring gives
0.8754457425; SHA-256 is
`8d36d645e2b4e7d53b7513526c55014ff84c8e90b646191d16aa45404ada1e7b`.

The initial C7 fit supplied both identical support files. That effectively
gave the graph-support feature two slots when LightGBM sampled features. It
scored **0.8754260110**, 0.0000197313 below C7a, and was the lowest
observed reproducible development score at that checkpoint. Independent float32 scoring gives
0.8754260111; SHA-256 is
`b92c3c007a28704c736e6b6e37cd707ed7cc6dfa5790966893f27acafe966eb9`.
The two input columns have the same SHA-256
`6ff8f1e1444a086f884cc20efdd8dc451a56e447b8a3b62be410014787987b74`.
Reports are `experiments/KNN-C7-support-blend.json` and
`experiments/KNN-C7a-support-unique-blend.json`. The C7 gain over C6 is
0.0001450907, putting it 0.0006166021 above historical R5. Both variants
remain original-probe development results; the tiny C7-versus-C7a difference
should not be treated as a generalization result.

The subsequent C8 experiment replaced C7a's conditioned RBM prediction with
its fixed epoch-45 continuation, holding the other input hashes fixed. It
scored **0.8752712189**; see [RBM_CONTEXT.md](RBM_CONTEXT.md) and
`experiments/RBM-C8-45e-blend.json`.

## Seven-day temporal kernel screen

A seven-day GP kernel on the same J3 factors, neighbor count and persistent
weight scores 0.8923871307 alone for noise variant 0. Its one-weight
cross-fitted combination with C8 scores 0.8752742258, worse than C8's
0.8752712190 independently scored column. Variant 1 also worsens it to
0.8752737795. The three seven-day support columns correlate 0.991–0.997
with their 30-day counterparts. This does not justify a full tree refit.
The run and screens are `experiments/GP-C8-7day-full-v2.json` and
`experiments/GP-C8-7day-screen-{0,1}.json`. The first launch failed at output
writing because the output directory did not exist; its failure is recorded
in `experiments/GP-C8-7day-full.json`.

The fixed epoch-60 conditioned RBM continuation subsequently improved the
matched C8 bank to C9 at **0.8750626363**; see
[RBM_CONTEXT.md](RBM_CONTEXT.md).
