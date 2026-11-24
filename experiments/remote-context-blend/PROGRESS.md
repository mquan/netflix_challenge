# Remote context-blend progress

| UTC | Phase | State |
|---|---|---|
| 2026-11-24T06:52:23Z | launch | `s3-tx300r__knn3` (control) and `s3-tx300r__knn3d` (a=0.5, τ=30) running in tmux `context-blend`, 6 Rayon threads each, ~3 GB RSS each; both phases (train_3→probe_3, fulltrain→qual) |

Next: RBM400 LR 0.015 native repackage; bank inventory (`bank.json`); blend code + synthetic tests.
| 2026-11-24T06:56:22Z | rbm400pf-lr015 | Full native repackage done: 2 parts / 1,181,260,377 B / 56 members (31 query arrays, train/fulltrain/ifeat side arrays, trace, logs, row map, 16 canonical probex+qual pairs); sources/data/binary verified against weekend plan (only `.gitignore` absent from r11 clone). Uploaded to draft release `remote-context-blend-20261124`; part01 download round-trip SHA verified. |
| 2026-11-24T06:58:14Z | knn3d | Both chains complete, exit 0, both phases (~4 min each at 6 threads, ~3 GB). Standalone probe_3 RMSE: TX300r 0.87834 → knn3 0.87607, knn3d 0.87596. Native outputs hashed in `knn3d/outputs.sha256`. |
| 2026-11-24T06:58:14Z | blend | Amendment before any comparison: inner λ_ctx split seed = 20261116+1+outer (final all-known fit: 20261116+99); inner folds reuse the outer-training cutoffs (cutoffs use no labels). Code `scripts/rcb_blend.py`, synthetic tests pass. |
