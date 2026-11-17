# Fresh snapshots and context-weighted blending

The saved-output experiment is complete. The original fourteen-model control reproduced at 0.8677119362283469 (previous 0.8677119362283443). The best new arm scores **0.8663200739003126**, improving by **0.0013918623280342**, with improvements in all five fixed user folds.

| Arm | Prediction columns | Ridge design columns including intercept | User-fold development RMSE |
| --- | ---: | ---: | ---: |
| Fourteen control | 14 | 15 | 0.867711936 |
| Plus snapshots | 56 | 57 | 0.867361831 |
| Plus components | 19 | 20 | 0.867571837 |
| Plus snapshots and components | 61 | 62 | 0.867287793 |
| Fourteen with context weights | 14 | 90 | 0.866670632 |
| Snapshots/components with context weights | 61 | 137 | **0.866320074** |

The 45 saved epoch predictions and five component outputs were checked against their native parent execution receipts and frozen with content hashes. Three terminal epochs are exact duplicates of their parent prediction vectors and were removed. The result has fourteen original columns, 42 unique snapshots, and five components. Rating predictions clip to [0,6] at input; component outputs remain unclipped. All outputs clip to [1.05,4.95]. The original fourteen remain unchanged; the corrected neighborhood-support diagnostic is excluded.

Context uses five fixed transformations: log user count, log movie count, query date, query date minus the user's mean known training date, and query date minus the movie's mean known training date. Counts and date means are computed from trainx identities/dates only. ProbeX supplies query identities/dates only. Neither development outcomes nor qualifying outcomes enter context construction. The manifest records the exact six metadata input hashes and normalization constants.

The context design adds five context main effects and 70 interactions between those contexts and the fourteen original rating predictions, centered at a fixed 3.6. Snapshot/component weights are global in this first bounded experiment. This is a single learned regression layer, with lambda 1000 and an unpenalized intercept. Each outer user fold's labels are excluded from that fold's coefficient fit; there is no target-derived feature normalization or intermediate learned layer. No published epoch blend coefficients or released prediction features enter the fit.

Context improves the combined bank by 0.0009677191846978, in all five folds. The combined-context fold changes versus fourteen are -0.0014709373, -0.0014262430, -0.0014406452, -0.0013473243, and -0.0012744150. These are development comparisons on historically selected base recipes, not untouched test results. They remain above the released 137-column base-bank diagnostic (0.860485311) and full released stack diagnostic (0.856323241). The latter uses prelearned released columns and is not an end-to-end held-out result.

Reproduce with:

```sh
PYTHONPATH=scripts .venv/bin/python scripts/upstream_snapshot_blend.py
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
```

Evidence: `experiments/upstream-snapshot-blend/manifest.json` freezes the input bank, receipt hashes, duplicate map, clipping semantics, metadata inputs and recipe. `evaluation.json` records all six scores, paired fold deltas, and out-of-fold prediction file hashes. Prediction vectors are retained under ignored `artifacts/upstream-snapshot-blend/`.

The implementation suite passes 24 tests. New checks compare the streaming ridge to the original control, poison scoring-fold labels to check isolation even with context interactions, test component clipping and metadata aggregation, and reject changed native saved outputs. A repeat full evaluation checked the frozen manifest before fitting and reproduced every score exactly. It took 11.70 seconds with 2.86 GB maximum resident memory (macOS reports a 1.69 GB peak memory footprint); no base training jobs were launched. `verification.json` records execution measurements and implementation/test hashes.

Next: screen missing released predictors against this new fresh control, label that screen diagnostic, and resolve the complete native parent dependency bundles before selecting additional training. GBM/MLP experiments remain optional; any multi-layer evaluation must isolate the final scoring fold through every learned layer. Do not rerun completed base queues or start the full 201-job graph.
