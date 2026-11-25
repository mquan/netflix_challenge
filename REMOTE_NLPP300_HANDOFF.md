# Remote handoff after the 4800lm package

Prepare the pinned upstream `tsvdx5-300__nlpp__knn3` chain while 4800lm trains. Start it after both 4800lm phases and the planned `knnf` and `knns` runs complete and release memory. Return the 4800lm base and epoch predictions as soon as both base phases are verified. [4800lm handoff](REMOTE_4800LM_HANDOFF.md) · [Remote progress](experiments/remote-context-blend/PROGRESS.md)

## Execution order

Use upstream commit `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c` and the canonical new split. This package uses the `tsvdx5` integrated temporal model on `trainx → probex`, followed by its full-data phase on `fulltrain → qual`. The completed `s3-tx300r` experiment used a different model and split. Keep the new work in an isolated tree with verified fresh training inputs and zero qualifying targets. [Pinned recipe](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/src/bin/tsvdx5-new.rs) · [Pinned pipeline](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/pipeline-new.toml)

| Order | Published job | Required return |
| --- | --- | --- |
| 1 | `tsvdx5-300` | Both complete 23-epoch phases, paired query vectors, and parent artifacts required by NLPP |
| 2 | `tsvdx5-300__nlpp_paropt` | Trainx/ProbeX fit log and the fitted `tsvdx5-300__nlpp_regs.npy` |
| 3 | `tsvdx5-300__nlpp` | Paired query predictions and both training-row residual inputs for KNN3 |
| 4 | `tsvdx5-300__nlpp__knn3` | Paired final predictions and both phase logs |

Pin the build and data hashes, check memory and disk before launch, and record measured peak memory and epoch cadence before quoting an ETA. **49.74 GB is a source-derived lower bound** for the base, with peak memory still unmeasured. Run one heavy stage at a time. Preserve partial output and logs if the execution window closes. Return stage outputs as they finish, with row-order evidence, file hashes, shapes, and timestamps. Use the original capacities and budgets throughout. [Dependency and allocation evidence](MISSING_PREDICTOR_PROGRESS.md)

## How to measure it

The released endpoint gained **0.001452 ProbeX RMSE** in the early smaller-bank screen. The 400-factor endpoint gained **0.001600**, with a higher **64.96 GB** memory floor. The 300-factor package is first because it tests nearly the same historical signal at lower allocation cost. Neither screen predicts the gain against today's bank. [Released-vector screen](MISSING_PREDICTOR_PROGRESS.md)

Keep a fixed current-bank control and report the effect of adding the base, then NLPP, then KNN3 on the same ProbeX rows. The exact upstream parameter optimizer sees all ProbeX labels. Treat downstream ProbeX comparisons as development diagnostics even when the final blend uses user folds. Freeze the paired predictions and report that provenance so the coordinator can check transfer on both recovered qualifying groups with weights fitted on known ratings.

Local follow-up for context. The second-split TX300 addition had a small qualifying diagnostic gain and missed the fold selection rule. The separate rank-1000 `knnf` run finished here in 44 seconds, peaked at 4.1 GiB RSS, and added no gain over its parent in the qualifying blend. The current accepted fresh-trained qualifying control remains **0.85956670 / 0.86059917**.
