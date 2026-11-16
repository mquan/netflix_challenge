# Upstream fresh-training progress (M3 Max, 96 GB)

Auto-updated by `scripts/upstream_repro_queue.py`. Last update: 2026-11-16 01:16 PST.

- Queue state: **running** — running `tsvdx4-60__knns`
- Threads: RAYON 8, BLAS 1

ProbeX development diagnostics only: fresh training, released vectors imported for comparison.

| Job | State | Train h | Fresh RMSE | Released RMSE | Fresh − released | Pred RMS diff | Max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|
| tsvdx4-60 | done | 0.96 | 0.879210 | 0.879210 | +0.000000 | 0.000001 | 0.0000 |
| tsvdx4-60__knns | running |  |  |  |  |  |  |
| rbmx2-40 | queued |  |  |  |  |  |  |
| rbmx2-40__knn3 | queued |  |  |  |  |  |  |
| dnn-24 | queued |  |  |  |  |  |  |
| dnn-24__dnn-16p50 | queued |  |  |  |  |  |  |
| dnn-24__dnn-16p50__knn3 | queued |  |  |  |  |  |  |
| dnn-24__knn3 | queued |  |  |  |  |  |  |

## Residual-chain gains (RMSE reduction from parent to child)

| Parent → child | Fresh gain | Released gain |
|---|---:|---:|
| tsvdx4-60 → tsvdx4-60__knns | pending | pending |
| rbmx2-40 → rbmx2-40__knn3 | pending | pending |
| dnn-24 → dnn-24__knn3 | pending | pending |
| dnn-24 → dnn-24__dnn-16p50 | pending | pending |
| dnn-24__dnn-16p50 → dnn-24__dnn-16p50__knn3 | pending | pending |
