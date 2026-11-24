# Remote context-blend progress

| UTC | Phase | State |
|---|---|---|
| 2026-11-24T06:52:23Z | launch | `s3-tx300r__knn3` (control) and `s3-tx300r__knn3d` (a=0.5, τ=30) running in tmux `context-blend`, 6 Rayon threads each, ~3 GB RSS each; both phases (train_3→probe_3, fulltrain→qual) |

Next: RBM400 LR 0.015 native repackage; bank inventory (`bank.json`); blend code + synthetic tests.
| 2026-11-24T06:56:22Z | rbm400pf-lr015 | Full native repackage done: 2 parts / 1,181,260,377 B / 56 members (31 query arrays, train/fulltrain/ifeat side arrays, trace, logs, row map, 16 canonical probex+qual pairs); sources/data/binary verified against weekend plan (only `.gitignore` absent from r11 clone). Uploaded to draft release `remote-context-blend-20261124`; part01 download round-trip SHA verified. |
| 2026-11-24T06:58:14Z | knn3d | Both chains complete, exit 0, both phases (~4 min each at 6 threads, ~3 GB). Standalone probe_3 RMSE: TX300r 0.87834 → knn3 0.87607, knn3d 0.87596. Native outputs hashed in `knn3d/outputs.sha256`. |
| 2026-11-24T06:58:14Z | blend | Amendment before any comparison: inner λ_ctx split seed = 20261116+1+outer (final all-known fit: 20261116+99); inner folds reuse the outer-training cutoffs (cutoffs use no labels). Code `scripts/rcb_blend.py`, synthetic tests pass. |
| 2026-11-24T06:59:42Z | blend v1 | Done (28 s). Global 0.863340, context 0.862646 (−0.00069, 5/5 folds) → below 0.001 bar, **global primary**. knn3d incremental −0.000096 (knn3 −0.000044). λ_ctx hit grid floor 1e4 everywhere. Frozen qual vectors + knn3/knn3d native packages uploaded. See RESULTS_V1.md. Recovered audit: pending on coordinator. |
| 2026-11-24T07:00:55Z | blend v2 | Context v2 0.862355 vs global 0.863340 (−0.000985, 5/5) — misses 0.001 bar by 0.000015; global stays primary. Uploaded `rcb-blend-v2.tar.gz`. Owner decision pending. |
