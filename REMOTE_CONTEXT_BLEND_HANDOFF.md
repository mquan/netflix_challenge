# Remote temporal residual and context-dependent blend

## Objective

Run a separate full-data research workflow on the remote Mac. Train a new temporal residual predictor and test whether a context-dependent blend improves on a global blend using the same predictors. The six-day horizon is a planning reference. Keep running useful work and keep the owner informed so they can intervene on timing.

The coordinator's latest independently verified primary is own17 simplex, with RMSE **0.86032634** on 1,408,789 recovered rows and **0.86132655** on 1,408,342. The goal is an independently verified primary **strictly below 0.85725 on each complete group**. Treat those scores as the fixed audit benchmark. Use only known rows to develop and select models and weights. The coordinator holds the recovered labels for the final audit.

## Start here

Use a separate worktree, execution branch, artifact directory, and owner for this study. Start from the existing remote checkout without switching or cleaning it:

```sh
git fetch origin research/remote-context-blend-handoff
git worktree add -b research/remote-context-blend-execution \
  ../netflix_remote_context_blend origin/research/remote-context-blend-handoff
```

Inventory live processes, hardware, runtime versions, and owned inputs on arrival. The last remote report described the 96 GiB Mac and showed the weekend training jobs finished. That is a historical snapshot. Verify the current machine and files before starting. Preserve existing jobs, frozen references, models, data, and prior experiment directories on both machines. Keep the local paused latent worker under its existing owner.

Use as much of the laptop as produces useful throughput. Tune thread counts and concurrency to the actual CPU, GPU, memory, swap, storage, and thermal behavior. Use CPU and GPU together where the implementations support it and the measured work benefits. Keep the machine observable and interruptible by the owner. Choose the host settings from measurements on this machine. If a setting affects the numerical recipe, record and freeze it as part of the versioned experiment before the fit. Record actual resource use and use the owner’s timing direction as it arrives.

## Research instructions

1. **Produce a new temporal residual predictor.** Start from the existing native temporal implementation. Test a temporal correction alongside the existing TX300 trajectory, and predeclare what the correction adds, such as time-dependent residual neighborhoods. Use the original full training populations for both the known and qualifying fits. Pin the actual commands, settings, initializations and full training budget before launch. Return the full native outputs and a matched pair in canonical row order. Use the completed TX300 as a reference rather than running the same recipe again.
2. **Test the value of context in the blend.** Fit a global regularized blend and a context-dependent counterpart on the same admitted prediction bank. Start with user activity, movie popularity and rating-date recency computed using information permitted in the corresponding training split. Shrink context-specific adjustments toward the global weights. Freeze the candidates, features, penalties, tie-break and fold rule before comparing scores. The proposed rule gives the context-dependent blend the primary slot only if it improves pooled known RMSE by at least 0.001 and improves at least four of five outer folds. Record any owner-directed change before the comparison.
3. **Keep the known comparison genuinely held out.** Use the original five user folds. Fit blend weights, preprocessing, context cutoffs and any tuning inside the appropriate training folds. For each outer comparison, use predictor features produced without that outer fold's outcomes. Use complete native holdout predictors that excluded the whole known population, or regenerate the required fold-specific inputs. An existing OOF vector alone does not prove that its other rows' models excluded the current outer fold. Record pooled and per-fold results on identical rows, plus the new temporal predictor's incremental effect under the same blend.
4. **Use additional capacity for complementary work.** Repackage the completed RBM400 LR 0.015 rerun if the original native files are still on the remote Mac. Its current compact export contains all 16 finite pairs. Return the full native arrays and lineage so the coordinator can independently admit it as the changed recipe. The original LR 0.02 run remains a partial failure. A fresh full latent CatBoost arm is another useful comparison if its input inventory and measured throughput support it. Preserve all five folds and three seed slots. The current source closure spans about 110 GiB across logical paths, so check existing hashes and transfer only missing inputs.
5. **Freeze one primary and hand it back for audit.** Once the known comparison selects a primary, freeze its configuration, coefficients, provenance, and predictions for all 2,817,131 qualifying rows. Preserve the matched global control and the corresponding known vectors. Return the known inputs and fold receipts so the coordinator can independently replay model selection and fit the final calibration locally using known labels. The coordinator verifies the full source closure, reproduces the frozen predictions, and then scores both complete recovered groups. If the coordinator audit lane is occupied, return the frozen artifacts and record that the recovered audit is pending.

## Return and report

- Keep a run manifest with the exact commit, source and binary hashes, inputs, split and row-map hashes, commands, environments, seeds, phase receipts, and model inventory. Record failed, partial, and complete jobs explicitly so the coordinator can tell which artifacts can be admitted.
- For each completed producer, package every expected finite native prediction array and side output required by its verifier, plus compact canonical pairs and training logs. Bind each member and archive part by byte count and SHA-256. Confirm that the receiving side can sustain and verify a transfer with an early archive part. Return completed artifacts as they become available.
- Keep raw ratings, models, predictions, and full logs in the approved private artifact channel. Commit code, instructions and shareable progress receipts to the execution branch in small verified batches. Give the owner the live phase, completed work, observed throughput, next decision, and any intervention needed. Preserve checkpoints and receipts when the owner changes priorities or timing.
- Use synthetic zero qualifying targets in all producer inputs that expect target fields. Keep recovered outcomes out of the remote study, selection, and all preaudit work. Report the two recovered scores as independent adaptive audits once the coordinator has verified them.

## References in this repo

- [Original remote producers and preparation](REMOTE_WEEKEND_HANDOFF.md)
- [Temporal source and data background](OWN352_TEMPORAL_PLAN.md)
- [Full-context CatBoost recipe and verification](OWN_FULL_CONTEXT_CATBOOST_FOLLOWUP_PLAN.md)
- [Complete archive-return evidence](REMOTE_WEEKEND_OWNER_OVERRIDE_RETURN_PLAN.md)
- [Published weekend completion and uploaded run](https://github.com/mquan/netflix_challenge/commit/751f6d5)
