# Phase 0 — Stage C dress rehearsal

**Result.** Retraining on the extra ratings improves a 9-column core bank by
**0.00755** RMSE (every fold: 0.0069–0.0083). A blender fit on a different
holdout transfers with almost no loss (dev CV is 0.00018 *better* than the
transferred score). So a Stage C test score ≈ **dev CV − 0.0074**. The
0.85725 test target corresponds to **dev CV ≈ 0.8646**, which confirms the ≈0.865
proxy in [PLAN_TO_TARGET.md](PLAN_TO_TARGET.md). The current best (0.87039)
maps to ≈0.8630 test; about **0.0058** of dev CV remains.

## Design

The final submission (Stage C) fits the blender on probe columns from models
trained without probe, retrains every model on train+probe, and scores the
qualifying set. The rehearsal runs the same procedure one level down:

| Real Stage C | Rehearsal |
|---|---|
| training ratings T | T minus P2 |
| probe (blender fit set) | P2 = `artifacts/validation/train-matched-20260927.bin` holdout (1,407,566 training ratings dated nearest each user's probe dates; see HOLDOUT.md) |
| qualifying (scored) | original probe |
| retrain on T + probe | retrain on T (P2 added back) |

Three numbers, all on the original probe, from `scripts/rehearsal_blend.py`
(round4_boost recipe: contextual ridge + LightGBM residual tree, learning rate
0.02, 8,000-round cap, extra meta, seed 2026, rounds chosen on an inner user split):

- **a**: blender fit on P2, applied to columns from the models trained without P2.
- **b**: the same blender applied to columns from models retrained on all
  training ratings at the **P2-selected** epochs. This is the Stage C analogue.
- **c**: the usual five-user-fold dev CV on the b columns.

`a − b` is the retraining gain. `c − b` is how far our dev CV metric is from a
Stage C score. Probe labels choose nothing in a or b: epochs come from P2, and
the blender is fit on P2.

## Core bank

| Column | Recipe | P2-selected epoch | a (no P2) | b (retrained) | Gain |
|---|---|---:|---:|---:|---:|
| MH3 | masked history, categorical MSE | 4 | 0.92295 | 0.91741 | 0.0055 |
| MT0 | MH3, 12-epoch budget | 12 | 0.91626 | 0.91102 | 0.0052 |
| MT1 | + session context, 32 time bins | 10 | 0.90529 | 0.89893 | 0.0064 |
| MT2 | MT1 + suffix masking | 12 | 0.90807 | 0.90149 | 0.0066 |
| MT3 | MT1 at rank 128 | 7 | 0.90561 | 0.89886 | 0.0067 |
| K1 | joint 256 factors, reg .02, decay .97 | 41 | 0.90295 | 0.89422 | 0.0087 |
| kNN-256td | time-decayed 256-neighbor kNN on K1 residuals (+ support) | — | 0.90137 | 0.89239 | 0.0090 |
| kNN-64 | 64-neighbor kNN on K1 residuals, power 1 | — | 0.90155 | 0.89260 | 0.0089 |
| RBM-100c | R2 plain → R2C conditional chain | 60 → 130 | 0.90961 | 0.90201 | 0.0076 |

The P2-selected epochs equal the ones the original probe-selected runs used
(K1 41, plain RBM 60, conditional RBM 130; the masked-history refits already
used holdout-selected epochs), and every b column reproduces its
original-probe score (K1 0.89422 vs 0.89412; RBM 0.90201 vs 0.90198).

## Blends

| Bank | Fit (P2, in-sample) | a | **b** | c | a − b | c − b |
|---|---:|---:|---:|---:|---:|---:|
| 5 masked-history columns | 0.85955 | 0.89635 | 0.89007 | 0.88976 | 0.00628 | −0.00030 |
| + K1 + two kNN | 0.84584 | 0.88410 | 0.87649 | 0.87628 | 0.00760 | −0.00021 |
| **+ RBM-100c (core, 9 columns)** | 0.84309 | 0.88350 | **0.87595** | 0.87577 | **0.00755** | **−0.00018** |

Fold differences for the core bank: a − b = .00760 .00827 .00690 .00735 .00762;
c − b = −.00013 −.00019 −.00032 −.00013 −.00015.

## Reading

- **Retraining gain ≈ 0.0075** for a mixed bank. Factor and kNN columns gain
  most (≈0.009) and masked-history columns least (≈0.006). The full 74-column
  bank is weighted toward factor/kNN columns, so its gain is probably at least
  this large.
- **Our dev CV is not optimistic about the blender.** Fitting the blender on a
  differently distributed holdout (P2 is much easier: 0.843 in-sample) costs
  only 0.0002 against fitting it on probe. In real Stage C, probe and
  qualifying come from the same distribution.
- **Mapping:** test ≈ dev CV − 0.00755 + 0.00018 = dev CV − 0.0074, so
  0.85725 test ↔ **dev CV ≈ 0.8646**.

## Caveats

- P2 is not a true future set: its ratings sit next to probe dates but before
  them. The real probe is the latest data, from the same period as qualifying,
  so retraining on it probably helps qualifying at least as much. The published
  0.007–0.009 agrees. Treat 0.0074 as a conservative offset.
- Nine columns, one blender seed. The architectures and hyperparameters were
  chosen on probe in earlier rounds; the rehearsal only removes probe from
  epoch selection and blender fitting.
- The Netflix qualifying labels were never released. This calibrates an
  estimate; it is not a test score.
- Runs shared the machine with another session (load average 35–60), so
  wall times overstate cost.

## Reproduction

```sh
PY=/path/to/venv/python   # numpy, lightgbm, torch
D=artifacts/validation/train-matched-20260927.bin
$PY scripts/rehearsal_view.py --holdout $D --original data/processed/netflix.bin \
  --output artifacts/rehearsal/matched-train-original-probe.bin
# (a) trainers on D (epochs chosen on P2); graphs on D
build/netflix joint --data $D --model artifacts/rehearsal/A-K1.bin --predictions artifacts/rehearsal/A-K1.p2.f32 \
  --factors 256 --iterations 80 --learning-rate .008 --regularization .02 --decay .97 --patience 8 --threads 6
build/netflix rbm --data $D --model artifacts/rehearsal/A-rbm100.bin --predictions artifacts/rehearsal/A-rbm100.p2.f32 \
  --hidden 100 --iterations 60 --threads 6 --patience 5 --learning-rate .005 --visible-rate .004 \
  --conditional-rate .003 --decay .98
build/netflix graph --data $D --graph artifacts/rehearsal/A-neighbors64.bin --neighbors 64 --history-cap 256
build/netflix graph --data $D --graph artifacts/rehearsal/A-neighbors256.bin --neighbors 256 --history-cap 256
# masked-history selection checkpoints, scored on P2 and on the original probe
$PY scripts/masked_history_predict.py --checkpoint artifacts/masked/MT1-select.select.pt --data $D \
  --output artifacts/rehearsal/A-MT1-select.p2.f32   # likewise MH3, MT0, MT2, MT3, and with the view dataset
# everything downstream (probe scoring, kNN, (b) retrains at the selected epochs)
scripts/rehearsal_chain.sh k1
scripts/rehearsal_chain.sh rbm
$PY scripts/rehearsal_blend.py --spec artifacts/rehearsal/spec-core.json --output artifacts/rehearsal/P0-core \
  --threads 1 --processes 5
```

Code added for the rehearsal: `--fixed-epoch` for `rbm`; `--fingerprint-data`
for `eval-joint`, `eval-rbm` and `knn`, which lets a model trained on one
dataset score another whose training and implicit arrays are identical
(checked) and whose probe differs; `scripts/rehearsal_view.py`,
`scripts/masked_history_predict.py`, `scripts/rehearsal_chain.sh` and
`scripts/rehearsal_blend.py`. Reports are in `artifacts/rehearsal/P0-*.json`.
