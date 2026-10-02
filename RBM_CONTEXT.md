# Visible-context RBM experiment

The best historical probe development result remains R5 at 0.8748094089 RMSE.
Its input bank is absent, so the reproducible C1 ensemble at 0.8792984371 is
the screening reference. Neither score is an untouched test result.

BellKor's 2009 [section VI](https://www.asc.ohio-state.edu/statistics/dmsl/GrandPrize2009_BPC_BellKor.pdf)
adds rating-specific user, user-day, and item-frequency terms to the RBM's
visible logits. `src/rbm.hpp` now supports these terms with `--visible-context`:
bit 0 is user, bit 1 is user-day, and bit 2 is item-frequency. The old mode is
0; all three terms are 7. Frequency comes from all known interaction dates,
while only training ratings enter contrastive-divergence updates. The model
file records the mode and checks the dataset fingerprint on reload. The optional
`--fixed-epoch` saves the declared final epoch rather than selecting with probe
ratings. The frequency-dependent hidden weights described in the paper are not
implemented; the paper says they add little once frequency biases are present.

## Ten-epoch controlled pilot

Both runs use 100 hidden units, the R2 learning settings, seed 2026, eight
threads, a fixed epoch-10 checkpoint, and the same complete Netflix dataset.
The manifests include source snapshots and hashes:
`experiments/RBM-C0-control-10e.json` and
`experiments/RBM-C7-visible-10e.json`.

| Predictor | Standalone probe RMSE | C1 plus one cross-fitted linear weight |
| --- | ---: | ---: |
| C1 alone | — | 0.8792984371 |
| RBM-C0 (original visible logits) | 0.9297121017 | 0.8792984865 |
| RBM-C7 (user, day, frequency logits) | 0.9243300870 | 0.8792827886 |

Conditioning improves standalone RMSE by 0.0053820 at this checkpoint. Its
incremental C1 gain is only 0.00001565, below the project's 0.001 threshold
for costly model expansion. This screening blend adds one weight to C1's saved
out-of-fold prediction in each held-out user fold; it is not a full tree refit
and does not establish a gain over R5. Base-model design still used probe data.
The ten-epoch RBMs are undertrained relative to the earlier 60-epoch RBM.

Next: compare a longer fixed-epoch C7 run and its matched C0 continuation.
Promote a candidate only if it adds material ensemble value, then test it
against the restored R5 bank. If the gain stays small, move to a different
model family rather than tuning this RBM indefinitely.

## Epoch-30 continuation and full C1 blend

C7 was warm-started from its fixed epoch-10 checkpoint and trained through a
predeclared epoch 30. Momentum and user-shuffle state restart at epoch 11, so
this is not an exact uninterrupted 30-epoch run. The fixed final checkpoint
scores **0.9080993293** standalone probe RMSE. A single cross-fitted weight
on C1 scores 0.8790235034, improving C1 by 0.0002749338.

The decisive comparison refits C1's residual-tree blender with exactly the
same settings and fourteen original prediction columns, adding only C7's
epoch-30 column. The original fourteen input hashes match C1's report. The
saved out-of-fold prediction scores 0.8785454267 on independent reload,
agreeing with the blender's 0.8785454268 report.

| Fold | C1 RMSE | C1 + C7 RMSE | Improvement |
| ---: | ---: | ---: | ---: |
| 0 | 0.8803157 | 0.8795530 | 0.0007627 |
| 1 | 0.8789251 | 0.8780670 | 0.0008581 |
| 2 | 0.8760476 | 0.8754412 | 0.0006064 |
| 3 | 0.8806375 | 0.8798074 | 0.0008301 |
| 4 | 0.8805773 | 0.8798694 | 0.0007079 |
| **All** | **0.8792984** | **0.8785454** | **0.0007530** |

This was the strongest reproducible local ensemble at the RBM comparison. It is still 0.0037360
behind historical R5 and 0.0235454 above the requested 0.855 target. The
incremental gain is below the 0.001 per costly family continuation gate, so
the next major effort should restore the R5 bank and test a different model
family. The archived C7 training manifest is
`experiments/RBM-C7-visible-30e.json`; the complete blender report is
`experiments/RBM-C7-C3-blend.json`. The saved C3 out-of-fold column has SHA-256
`1823629fc669ef76d9d0e80101a546db20ba5c56866f72c04852b4016e8fcb16`.
All comparisons are development estimates selected over the original probe,
not independent evidence for a 0.855 private-test score.

## Distinct neural addition

The saved epoch-13 DeepRecommender prediction (SHA-256
`178aae35be956f8369d05c885bab006e185d257481dabffe8abe026a8a276085`)
was added as a sixteenth column to the same C3 tree blend. Its training had
stopped before completion, so this is a frozen exploratory snapshot. C3's
fifteen input hashes match exactly. The resulting C4 scores **0.8783242000**
development CV, a further **0.0002212268** gain. Each fold improved by
0.00013–0.00028 RMSE. Independently reading its saved out-of-fold predictions
gives 0.8783241999 RMSE; their SHA-256 is
`fff685926bee5f0fb7b59eed0a5258bf00a9d9d4ec327b70e5ceb2082c3297d9`.
The full result is in `experiments/RBM-C7-DR1-C4-blend.json`. C4 was the
best reproducible local result at that checkpoint.

## Fixed epoch-45 continuation in the clean C7a bank

The conditioned RBM continued from epoch 30 to a fixed epoch 45; its standalone
RMSE improved from 0.9080993293 to 0.9054782797. Replacing only the RBM column
in the clean C7a ensemble gives C8 at **0.8752712189** development CV, a
0.0001745234 gain over C7a. Four of five user folds improve; the remaining
fold worsens by 0.00001396. Independent float32 scoring gives 0.8752712190.
The saved out-of-fold prediction SHA-256 is
`cd839161fc771790813be45ddca86fbf6731bb9d3b58bc29a3f8a6d243af72e7`.
All other input hashes match C7a. The warm start resets momentum and shuffle
state, as recorded in `experiments/RBM-C7-visible-45e.json`. The blend report
is `experiments/RBM-C8-45e-blend.json`. This is the best reproducible local
development result, 0.0004618100 above historical R5 and 0.0202712189 above
the requested 0.855. Model selection used probe feedback.

## Fixed epoch-60 continuation

Continuing the same RBM from epoch 45 to a fixed epoch 60 improves standalone
probe RMSE from 0.9054782797 to 0.9039561836. Replacing only the epoch-45
RBM column in C8 gives C9 at **0.8750626363** development CV, a
0.0002085827 gain. All five held-out user folds improve by
0.00012396–0.00032435. Independent float32 scoring gives 0.8750626363;
the saved out-of-fold SHA-256 is
`4404cfb596eb87c2fdb426c84cc292d243888494953e73785e63ef65a3bad435`.
All other input hashes match C8. The training manifest is
`experiments/RBM-C7-visible-60e.json`; the matched blender report is
`experiments/RBM-C9-60e-blend.json`. C9 is now the best reproducible local
development result, 0.0002532274 above historical R5 and 0.0200626363
above the 0.855 target. Repeated probe-guided choices limit interpretation.
