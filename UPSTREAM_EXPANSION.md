# Next bounded run: mismatch diagnostics and six new models

**Execution update:** the user moved all remaining work to the first Mac because the other computer is busy. `experiments/upstream-expansion/execution.json` supersedes the original machine assignment below. Do not start the M3 launcher. The scientific model set, folds and comparisons remain fixed.

Local continuation command:

```sh
UPSTREAM_REPRO_THREADS=4 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_expansion.py --machine m1 --take-over-m3 --publish
```

The completed wider DNN is verified and reused. The first new job is `tsvdx4-400lm`, followed by its factor-neighbor chain, the 150-unit RBM and its chain, then the attention chain. The latter automatically trains its missing `dnn-24` parent locally. Results are exported under `experiments/upstream-expansion/m1/`, with the actual execution host recorded. Afterward the fixed fourteen-model evaluation runs locally without waiting for M3. Existing diagnostics are retained.

The user approved starting the mismatch investigation and a larger upstream model batch. The first eight-model replay is complete. Preserve it; none of its predictions, source recipes or receipts are overwritten by this continuation.

## Fixed assignment

| Machine | Work |
| --- | --- |
| M1 Pro / verification branch | Fresh local `tsvdx4-60` replay for independent diagnosis; support-provenance diagnostic; train `dnn-64`; verify both machines' new outputs and run the fixed fourteen-column comparisons |
| M3 Max / reproduction branch | Reuse existing `tsvdx4-60` saved train predictions for the support diagnostic; then `dnn-24__attn-32`, `tsvdx4-400lm`, `tsvdx4-400lm__knnf`, `rbmx2-150`, `rbmx2-150__knns`, sequentially |

The new six jobs all resolve through the pinned author's pipeline at `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. The attention job reuses the already-trained `dnn-24` parent; existing receipts are verified. The temporal and RBM base recipes may lack released fixtures, so their scores are reported without inventing a reference comparison. Downloaded fixtures are never model inputs.

This is a bounded capacity/diversity extension, not the full 201-job replay. The larger models can take substantially longer than the first small-model batch; there is no measured whole-batch ETA yet. Keep native logs and receipts for a defensible estimate after the first new job.

## Start on the M3 Max

Use the **same upstream-reproduction worktree and venv** that completed the first eight jobs. Confirm the previous queue has exited before launching. Do not create a second fresh-training tree or rerun dataset setup.

```sh
git pull --ff-only origin research/upstream-reproduction
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
.venv/bin/python scripts/upstream_expansion.py --machine m3 --plan-only
tmux new-session -d -s upstream-expansion 'env UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python -u scripts/upstream_expansion.py --machine m3 --publish > upstream-expansion.log 2>&1'
```

The driver prevents duplicate expansion workers, keeps macOS awake, uses the existing receipt/hash guards, preserves partial failures, and publishes stage transitions plus a half-hour heartbeat. It never force-pushes or merges another branch. Resume with the same command after inspecting a failure; unreceipted outputs remain protected.

Results and compressed **ProbeX-only** vectors are committed to `experiments/upstream-expansion/m3/` on `research/upstream-reproduction`. Progress is `EXPANSION_M3.md`. Qualifying outcomes remain zero sentinels; qualifying/full-training/latent artifacts stay on the machine.

The first Mac's driver is on `research/upstream-verification`:

```sh
UPSTREAM_REPRO_THREADS=4 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_expansion.py --machine m1 --publish
```

It waits for its already-running diagnostic base replay rather than duplicating it. After the support diagnostic it trains the wider DNN, then fetches M3 expansion artifacts and evaluates once all six new outputs are present.

## Mismatch hypotheses, not fixes

The released `tsvdx4-60__knns.cfg` agrees on k/shrinkage/scaling/temporal decay but omits support provenance. The current pinned dispatcher computes support from training plus prediction rows; the model's default points to a precomputed full-training support matrix. The diagnostic holds the learned residual base fixed and compares current support against support that also includes qualifying query identities. **Qualifying outcomes must be zeros.** The helper asserts this and writes separate diagnostic files under the fresh checkout's `investigation/` directory. It does not patch the tracked upstream models or replace the original chained output.

A prediction-level match after changing only support provenance would support the hypothesis; a lower score alone would not. Diagnostic variants are not expansion features and are not selected for the final bank based on their scores.

The released RBM chain config targets `1.0*rbmx2-b40`; the current dispatcher uses `1.0*rbmx2-40`. Neither historical base vector/config is available in the pinned store. This is evidence of an identity/version discrepancy, not proof that the rename changed training. The RBM mismatch remains unresolved until its missing parent or equivalent historical recipe can be established. Do not patch the generic KNN3 implementation merely because that output differs: all DNN KNN3 chains already matched exactly.

The published config snapshots and checksum evidence are recorded separately in `experiments/upstream-expansion/investigation/`.

## Fixed evaluation

The original eight fresh vectors remain pinned by their previously verified artifact hashes. The six expansion outputs are transferred with dtype/shape, native receipt and content hashes, and independently rescored against the exact same ProbeX labels/user identities.

Predeclared arms are the original eight, plus temporal additions, plus RBM additions, plus neural additions, and all fourteen. Reuse the same five user folds (seed 20261116), ridge lambda 1000, unpenalized intercept, input clip `[0,6]` and output clip `[1.05,4.95]`. The original-eight control must reproduce its prior 0.869549365 score before accepting any new comparison. No tuning or downloaded predictors enter these fits.

This remains development evidence on historically development-selected base recipes, not an untouched or official test score. Scores, paired fold deltas and prediction hashes will appear in `experiments/upstream-expansion/evaluation.json` on the verification branch. The M1 driver exits once that comparison completes.
