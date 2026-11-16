# Upstream fresh-training progress (M3 Max, 96 GB)

Auto-updated by `scripts/upstream_repro_queue.py`. Last update: 2026-11-16 02:22 PST.

- Queue state: **running**
- Threads: RAYON 8, BLAS 1

ProbeX development diagnostics only: fresh training, released vectors imported for comparison.

| Job | State | Train h | Fresh RMSE | Released RMSE | Fresh − released | Pred RMS diff | Max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|
| tsvdx4-60 | done | 0.96 | 0.879210 | 0.879210 | +0.000000 | 0.000001 | 0.0000 |
| tsvdx4-60__knns | done | 0.07 | 0.879096 | 0.878953 | +0.000143 | 0.017859 | 0.4994 |
| rbmx2-40 | done | 0.96 | 0.921512 | nan | +nan | nan | nan |
| rbmx2-40__knn3 | done | 0.05 | 0.903718 | 0.903674 | +0.000044 | 0.040524 | 3.7925 |
| dnn-24 | queued |  |  |  |  |  |  |
| dnn-24__dnn-16p50 | queued |  |  |  |  |  |  |
| dnn-24__dnn-16p50__knn3 | queued |  |  |  |  |  |  |
| dnn-24__knn3 | queued |  |  |  |  |  |  |

## Residual-chain gains (RMSE reduction from parent to child)

| Parent → child | Fresh gain | Released gain |
|---|---:|---:|
| tsvdx4-60 → tsvdx4-60__knns | 0.000115 | 0.000258 |
| rbmx2-40 → rbmx2-40__knn3 | 0.017795 | nan |
| dnn-24 → dnn-24__knn3 | pending | pending |
| dnn-24 → dnn-24__dnn-16p50 | pending | pending |
| dnn-24__dnn-16p50 → dnn-24__dnn-16p50__knn3 | pending | pending |
