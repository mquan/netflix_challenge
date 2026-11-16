# Round 4: combining both branches

Branch `round4-combined` = `score-iteration-2` merged with `moonshot-local-kernels`
(clean merge). The source branches are unchanged. The protocol is the same as
ROUND3's refitted residual tree: full probe split, five user folds, a contextual
ridge fit on outer-training users, and a LightGBM residual whose round count is
chosen on a user-disjoint inner split and then refit on all outer-training users.
Base-model epochs were still chosen on probe, so these are **development CV**
numbers, as before.

Previous best: 0.8773860815 (ROUND3 three-way average). **Round-4 best:
0.8754534170** (`R4-C`), −0.0019327 overall, and all five folds improve.

## Blends

`scripts/round4_boost.py` reads the frozen 21-column union bank from the
moonshot worktree, checks its hashes, and appends extra columns.
The control reproduces ROUND3's `BOOST-residual.refit` exactly.

| Run | Change vs previous row | Dev CV | Fold RMSE |
|---|---|---:|---|
| R4-control | union bank only (reproduction) | 0.8774504869 | .87848 .87734 .87380 .87880 .87886 |
| R4-A | + 19 unblended score-iteration-2 columns and new kNN on T2/T3/T6 | 0.8761138185 | .87716 .87589 .87246 .87767 .87740 |
| R4-B | + time-decayed kNN, 256-neighbor graph kNN on six bases | 0.8758047393 | .87692 .87569 .87212 .87723 .87707 |
| **R4-C** | + converged I2R/T6R and their kNN; tree lr .05→.02 | **0.8754534170** | .87659 .87521 .87166 .87698 .87684 |

Ablations and non-promoted candidates:

| Run | Dev CV | Note |
|---|---:|---|
| R4-A-lr02 (A columns, lr .02) | 0.8759100881 | lr .02 is worth ≈0.0002 on its own |
| R4-C-l127 (C columns, lr .05, 127 leaves, min leaf 100) | 0.8758178784 | worse than lr .02 |
| R4-NN-C (neural residual, same 55 columns) | 0.8773470264 | far behind the tree |
| equal average of C with NN / l127 / B | 0.87548–0.87587 | no fixed average beats C alone |

The tree's top features by gain are the context metadata, then AutoRec-A3 and
RBM-200. Non-factor models are what the blender values beyond the factor/kNN
family.

Duplicate found: `joint256-reg01.f32` (J6 final) is byte-identical to
`joint256-snap.f32` in the union bank. `knn-J6-p1` equals `knn64-j256snap`.
They are harmless in a tree, but the J6 kNN columns are variants of an
existing base, not a new model.

## Base-model work

| Run | What | Probe RMSE |
|---|---|---:|
| I2R | `--start-epoch 18` warm start of I2 (factors + `.nbr` neighbor weights), same schedule; early stop | 0.8945819821 (was 0.8950571618 at stop) |
| T6R | warm start of T6 (256 factors) from epoch 11; early stop, best epoch 24 | 0.8947630290 (was 0.8994939730) |
| R3R | warm start of RBM-200 from epoch 30; velocities reset. Still running | 0.9101496 at epoch 43 (was 0.9117706) |
| kNN time decay | `--time-decay` weights neighbors by exp(−λ·days); J6 base, λ=.003 | 0.8928612677 (vs 0.8930184131) |
| G2 | top-256 residual item graph (62 s); kNN k=256, λ=.003 on J6 | 0.8927408788 |
| kNN on T6R | 64-graph / 256-graph with time decay | 0.8932845082 / 0.8929569610 |

kNN over I2R's factors alone (ignoring its own neighbor term) is poor
(0.9128) and is not used.

## Reproduce

```sh
PY=/path/to/python-with-numpy-lightgbm   # LightGBM 4.7.0
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY scripts/round4_boost.py --data data/processed/netflix.bin --output artifacts/r4/R4-C \
  --learning-rate .02 --rounds 8000 --extra $(cat artifacts/r4/extra-C.txt) \
  --extra-support $(cat artifacts/r4/support-B.txt) artifacts/r4/knn256td-T6R.support.f32
```

Column lists and reports with all input hashes are copied to `experiments/round4/`.
kNN columns are deterministic and take ≈25 s each. Commands are in the `R4-*` and
`*R-*resume` experiment logs.
