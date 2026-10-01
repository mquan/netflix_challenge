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
