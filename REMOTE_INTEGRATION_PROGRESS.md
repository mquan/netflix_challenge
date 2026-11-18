# Remote bknbrx integration progress

Branch: `research/fresh-remote-integration`. Updated 2026-11-18T04:07:53.895875+00:00.

Transfer provenance and native row alignment verified; three exported vectors contain two unique predictions. The fixed ridge addition improves all five folds: **0.8662864618 → 0.8655052917** (gain 0.0007811701).

The unchanged three-seed nonlinear recipes are being refitted on the expanded 64-column bank. These are partial results, not the final five-fold comparison.

| Family | User fold | New RMSE | Change from same family before remote addition |
| --- | ---: | ---: | ---: |
| tree | 0 | 0.8630139129 | -0.0008202076 |
| tree | 1 | 0.8612726397 | -0.0008333178 |
| nn | 0 | 0.8631222446 | -0.0008346714 |

All 43 checks pass (39 upstream and four remote supervisor). Both workers use two CPU threads. Input, code, package and source-manifest hashes are frozen. No hyperparameter or feature changes follow these results.

The more ambitious remote assignment is separately published on `research/remote-temporal-push` in `REMOTE_TEMPORAL_PLAN.md`. It starts ordinal temporal training, then profiles larger temporal parents and compatible chains, with a cheaper fallback. NLPP full-ProbeX parameter search and published epoch coefficients are excluded.

After both five-fold runs, require saved-model replay and independent scoring, then publish all final predictions and matched fold deltas. This is fresh-only ProbeX development evidence, not a verified qualifying score.
