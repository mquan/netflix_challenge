# Remote handoff after TX300

## While TX300 is running

Keep the current TX300 arms on their declared schedules. Prepare the next job from pinned upstream commit `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. Resolve its native dependencies and estimate the storage needed for both phases, including train-row predictions, item factors, and snapshots. Confirm that the coordinator can retrieve the completed RBM800 return and the existing residual-wave package. Send the existing asset names, sizes, and SHA-256 values if needed. CF96 and the tree package have already been received and audited on the coordinator. [Current remote progress](experiments/remote-context-blend/PROGRESS.md) · [Residual wave package](experiments/remote-context-blend/RESULTS_V2.md)

## Next heavy job

Run upstream's original `tsvdx4-4800lm` on the new split. Use the published low-memory preset, with 4,800 factors and 27 epochs in each phase. Start after the TX300 training load clears and the machine has headroom. The source-derived 35.35 GB figure is an allocation lower bound. Record actual memory use, disk use, and epoch time early in the original run and use them to update the two-phase ETA. [Upstream preset](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/src/bin/tsvdx4-new.rs) · [Dependency and cost audit](MISSING_PREDICTOR_PROGRESS.md)

- Run the full `trainx → probex` and `fulltrain → qual` phases. Preserve the terminal predictions and the published snapshots at epochs 3, 13, 21, and 24, plus the train-row predictions and item factors needed by the downstream jobs.
- Reuse the completed parent for upstream's `tsvdx4-4800lm__knnf` and `tsvdx4-4800lm__knns` in both phases. Consider `__knn-d` and its statistics dependency after the primary package is complete and measured.
- Export paired ProbeX and qualifying predictions with row order, shapes, hashes, recipe and binary provenance, resource traces, and complete logs. Publish the base and snapshots as soon as both phases are verified so the coordinator can start integrating while the residual jobs run. Preserve partial checkpoints and logs if the window closes.
- Compare the fixed native bank, bank plus base and snapshots, and bank plus the full residual package on the same five user folds. Report pooled and per-fold deltas. Use only freshly generated predictions for fitting. The coordinator will handle the recovered-label audit of frozen candidates. [Upstream pipeline](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/pipeline-new.toml)

## Why this package comes next

The released 4800lm package improved the early matched development bank by 0.00164 RMSE. That is a historical screening result on a smaller bank. The existing 22-job residual wave gained 0.000185 on the native ridge control. This makes a new high-capacity parent with its own residual outputs a better next experiment than another broad sweep over the same parents. [Missing-predictor screen](MISSING_PREDICTOR_PROGRESS.md) · [Remote wave results](experiments/remote-context-blend/RESULTS_V2.md)

The latest coordinator audit of Own17x + CF96 + separate context and tree corrections scored 0.85956670 / 0.86059917. The harder group is 0.00334917 above 0.85725. That result improved on the previous best by 0.00007107 on the harder group. The public-bank comparison remains useful for choosing experiments, while the fresh-training target is measured on both recovered groups with known-only fitting.
