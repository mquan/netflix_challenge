# Next assignment: start the bounded upstream expansion

The first eight-job source replay is complete and independently verified. The user has now explicitly approved investigating the two neighborhood mismatches and training the next larger batch.

**M3 Max:** fetch/pull `research/upstream-reproduction` in the existing upstream worktree and follow **UPSTREAM_EXPANSION.md**. Confirm the old queue exited, then launch:

```sh
tmux new-session -d -s upstream-expansion 'env UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python -u scripts/upstream_expansion.py --machine m3 --publish > upstream-expansion.log 2>&1'
```

The driver reuses the original native training artifacts for a support-provenance diagnostic, then trains `dnn-24__attn-32`, `tsvdx4-400lm`, `tsvdx4-400lm__knnf`, `rbmx2-150`, and `rbmx2-150__knns`, sequentially. It publishes verified fresh vectors, native receipts/logs and per-stage progress. No setup rerun, predictor-bank download into the training directory, or retraining of the first eight models is needed.

**First Mac:** a local replay is already training for independent diagnosis. Its durable continuation driver then runs the support diagnostic and `dnn-64`, fetches the M3 expansion outputs, and performs the fixed original-eight versus fourteen-column comparisons. It owns `research/upstream-verification`.

Preserve all previous outputs, including negative results and unresolved discrepancies. Do not force-push, delete partial predictions, change the pinned model code, or launch the full 201-job pipeline. Publish failures for investigation. The published RBM parent identity differs from the pinned recipe; that is recorded as unresolved, not silently patched.

Full instructions: https://github.com/mquan/netflix_challenge/blob/research/upstream-reproduction/UPSTREAM_EXPANSION.md
