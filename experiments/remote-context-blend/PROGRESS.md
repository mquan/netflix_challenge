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
| 2026-11-24T16:10:41Z | decision | Owner admitted **context v2 as primary** (global = control); asked to rerun on coordinator own17 bank, and to run more knn3d chains + fresh latent CatBoost. `primary.json` written. |
| 2026-11-24T16:11:14Z | own17 bank | Coordinator not reachable from this machine; arrays only exist there. Request pushed: `REQUEST_OWN17_BANK.md`. Waiting on upload. |
| 2026-11-24T16:12:02Z | knn3d wave | Declared + launched 22 chain jobs (11 bases × knn3/knn3d) in tmux `context-blend:wave`. |
| 2026-11-24T16:15:33Z | catboost | Env ready (`artifacts/venv-cb`, CatBoost 1.2.10). Throughput pilot: 0.19 s/iter @2 thr, 0.12 @8 thr, ~1 GB RSS, thread-invariant output. Full-context closure is not on this machine; added to the coordinator request. Local tree-residual arm on bank v2 will be declared next. |
| 2026-11-24T17:12:25Z | knn3d wave | 22 jobs finished, 0 non-zero exits; logs + output hashes in knn3d/wave/ |
| 2026-11-24T17:12:33Z | bank v2 | built: bank 65 columns |
| 2026-11-24T17:13:28Z | blend bank2 | {"pooled": {"global": 0.8631551888911532, "global-no-new": 0.8631608569382332, "global-plus-knn3": 0.8631587406826176, "global-plus-knn3d": 0.8631540712961634, "context": 0.8621969850303761}, "primary": "global", "fold_gain": [0.001156432444465616, 0.0009336142341882514, 0.0008149985122022363, 0.000 |
| 2026-11-24T17:29:53Z | tree units | 15/18 units done |
| 2026-11-24T17:29:56Z | tree combine | {"pooled": {"tree": 0.8617917848309753, "glob": 0.8631551887463838, "context": 0.8621969850388445}, "tree_vs_context_gain": 0.0004052002078691608, "tree_replaces_primary": false} |
| 2026-11-24T17:32:51Z | wave pkg + results | Wave packaged: `rcb-knn3d-wave.tar.gz` uploaded (sha 4784440b…). Tree arm 0.861792 vs context 0.862197 on bank v2 (−0.000405, 5/5) — below 0.001 bar, context v2 stays primary. Final tree units crashed (qual baseline bug, fixed f66e542); rerunning only those 3. `RESULTS_V2.md`. |
| 2026-11-24T17:37:01Z | tree final | 3 final units rerun OK (785/1093/991 it). qual-tree sha cf5404f5…. Uploaded `rcb-bank2-blend-tree.tar.gz` (sha d99cf925…). primary.json: context v2 (bank v1) unchanged; bank-v2 context/global/tree listed as adaptive follow-ups. Waiting on coordinator own17 bank + latent closure. |
| 2026-11-24T19:32:46Z | own17 | Coordinator own17 assets verified (17/17 column hashes, label hash, row order). Declared rerun (non-nested caveat). Banks own17 (17) and own17x (82) built; launching v2 blends. |
| 2026-11-24T19:34:58Z | own17 blends | own17: global 0.858675, context +0.000016 (0/5). own17x (82 cols): global 0.858340 (−0.000335 vs own17), context ±0. Global primary on both (non-nested CV). Uploaded `rcb-own17-blends.tar.gz`. Recommend audit own17x-global vs own17-global. |
| 2026-11-24T19:56:59Z | lanes launched | Coordinator v2 note read. **Lane 1**: CF48 checkpoint export (replay byte-exact) → 5 jobs (3 stats + knn3/knn3d) running, then knn-25c/25m. First result: cfnade-48__knn3 probex 0.88271 vs base 0.89496. CF96 inputs missing (lane1/README.md). **Lane 2**: split_3 seed 20261124 generated + verified; TX300 arms B (split-3, seed 20261124, both phases) and C (split-3s, known phase) running. **Lane 3**: already complete (tree arm on bank v2, 0.861792). |
