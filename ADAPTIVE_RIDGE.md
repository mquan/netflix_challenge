# Per-user adaptive factor correction pilot

The frozen J3 joint model supplies 24 item factors and a baseline prediction.
For each sampled user, `src/moonshot_adaptive.cpp` fits a regularized ridge
correction to that user's **training residuals only**. The pilot samples users
with a fixed modulo-16 rule, giving 87,737 probe rows. Probe labels are used
only to report the resulting development score. The source snapshot, input
hashes, command and outputs are in `experiments/ADAPT-24-m16-pilot.json`.

| Ridge penalty | Standalone sampled RMSE | One-weight C9 gain on full probe |
| ---: | ---: | ---: |
| 1 | 0.9683770 | −0.00000052 |
| 10 | 0.9120018 | +0.00000107 |
| 100 | 0.8973808 | +0.00000129 |

The frozen J3 baseline scores 0.8972676 on sampled rows. At penalty 100, the
cross-fitted C9 combination lowers sampled-row RMSE from 0.8754418 to
0.8754211, but changes full-probe RMSE by only 0.00000129 because 1/16 of
users are covered. Fold-specific weights are positive, 0.102–0.165. The
screen reports are `experiments/ADAPT-24-C9-screen-{0,1,2}.json`, produced by
`scripts/screen_correction.py` using five held-out user folds.

This pilot does not justify a full user run: the correction is weak and
highly regularized, while the existing local GP already adapts to each user's
history. It does not move the best reproducible development result beyond
C9's 0.8750626363. A later test would need a different signal, such as a
training-only selected temporal weighting or a much stronger latent basis.
