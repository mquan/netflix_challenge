# Round 8 — parallel bets alongside the Phase 0 rehearsal

Started 2026-10-02, while the Phase 0 rehearsal (PLAN_TO_TARGET.md) held most
of the CPU. Every arm is judged against the R7-MT56 bank with `round4_boost.py`
(learning rate 0.02, 8,000-tree cap, extra meta, seed 2026):
**control 0.8704870**, folds .8717022 .8702757 .8666393 .8720116 .8718174.
Promotion needs all five folds better (PLAN_TO_TARGET.md operating rules).

## MPS pilot

MT1 config, 300 batches, `--max-batches 300`, under load ≈19:

| Device | Wall | Probe RMSE after 300 batches |
|---|---:|---:|
| MPS | 165 s | 0.939429 |
| CPU, 2 threads | 205 s | 0.939468 |

All ops run on MPS and losses match CPU. The speedup is only ≈1.35×; the
value is moving the PyTorch families off the shared CPU.

## Results

| Arm | Change vs R7-MT56 | Dev CV | Δ | Fold Δ | Verdict |
|---|---|---:|---:|---|---|
| R8-A3-dist | + MT1/MT2 P1, P5, std, entropy as tree features (epoch-12 checkpoints) | 0.8705527 | +0.0000657 | +.000027 +.000063 +.000056 +.000140 +.000041 | Not retained |
| R8-meta | + 16 LLM movie-tag tree features (kind, audience, tone, popularity, franchise size, 10 genre flags) | 0.8704448 | −0.0000422 | −.000141 +.000002 −.000008 −.000100 +.000036 | Noise-level; not promoted |
| R8-C | + user-kNN column (extra) + its weight sum and neighbor count (tree only) | 0.8704436 | −0.0000435 | −.000112 +.000003 −.000003 −.000139 +.000034 | Noise-level; not promoted |

R8-meta and R8-C improve the same two folds (0 and 3) by similar amounts, so
single-seed results couldn't separate them from LightGBM round-selection noise
(±0.0002 per fold, ROUND6). Together, with three blender seeds, they pay on
every fold:

| Bank | Dev CV (seeds 2026/7/13) | Δ vs R7-MT56-s3 | Fold Δ |
|---|---:|---:|---|
| R7-MT56-s3 | 0.8703893 | — | — |
| **R8-CM-s3**: R7-MT56 + user-kNN column/support/count + 16 movie-tag features | **0.8702639** | **−0.0001254** | −.000146 −.000173 −.000070 −.000173 −.000064 |

Which half of R8-CM carries the gain is not yet separated.

Arms on top of R8-CM (three blender seeds):

| Bank | Dev CV | Δ vs R8-CM-s3 | Fold Δ | Verdict |
|---|---:|---:|---|---|
| R8-CM-A1-s3: + A1 item-side twin, holdout epoch 4 | 0.8702626 | −0.0000013 | +.000013 +.000024 −.000028 −.000055 +.000040 | Not retained |
| **R8-CM-MT8-s3**: + MT8 (MT3 config, seed 13), holdout epoch 7 | **0.8701814** | **−0.0000825** | −.000084 −.000057 −.000077 −.000177 −.000017 | **Retained; new best** |

| **R8-CM-MT89-s3**: R8-CM-MT8 + MT9 (MT2 suffix config at rank 128 / hidden 512, seed 13), holdout epoch 8 (0.87752) | **0.8701238** | −0.0000576 vs R8-CM-MT8-s3 | −.000059 −.000004 −.000120 −.000036 −.000069 | **Retained; new best** |

**Current best: R8-CM-MT89-s3 0.8701238** (−0.0002655 vs R7-MT56-s3). With
the Phase 0 offset (0.0074), the dev-CV target is ≈0.8646.

A1 overfits early: holdout 0.94736 0.93472 0.92736 **0.92525** 0.92562
0.92874 0.93337 (selected epoch 4; the refit was stopped at epoch 9). MT8's
holdout curve tracks MT3's almost exactly (selected epoch 7, 0.87527).

`R8-meta` tags come from `artifacts/r8/movie_tags.tsv` (17,770 titles tagged
from model knowledge; ≈2,470 marked low confidence), columns built by
`artifacts/r8/meta_columns.py`. The result agrees with Pilászy & Tikk (2009):
with ≈5,600 ratings per movie, content metadata adds little.

## Bet C — user-user kNN on K1 residuals

Branch `bet-c-user-knn` (`build/netflix user-knn`; binary copied to
`artifacts/r8/bin/netflix-uknn`). Joint-model user vectors screen the top 300
raters of the target movie, shrunk residual Pearson (α 50, power 2) picks 100
neighbors. Standalone: K1 0.8941175 → 0.8937946 probe; holdout 0.8724956 →
0.8720811. ≈10 min wall per dataset at 3 threads.

## Bet B — Bayesian timeSVD++ flipped: broken, under diagnosis

Branch `bet-b-bayes-timesvdpp` (`build/netflix bayes`, binary copied to
`artifacts/r8/bin/netflix-bayes`). Rank 64, 128 sweeps, 3 threads, nice 10
(B-p2-r64, B-full-r64) were stopped at sweep 14. Single-sample held-out RMSE
bottoms out at sweep 6 and then rises while train RMSE keeps falling:

| Run | Best sample RMSE (sweep 6) | Sweep 14 | Train RMSE at 14 |
|---|---:|---:|---:|
| Holdout (train-matched) | 0.94265 | 0.94669 | 0.7439 |
| Probe (full train) | 0.97238 | 0.97869 | 0.7445 |

**Correction.** Stopping these runs was partly a misreading: with
`--burn-in 16`, no posterior mean existed by sweep 14, and a single Gibbs
sample is 0.03–0.04 worse than the mean (plain BPMF on the holdout: sample
0.931 → 0.938 while the mean goes 0.944 → 0.8981 by sweep 16). The diagnosis
also found a real defect: every auxiliary group (sessions, item-bins, day
vectors) started from random values and stuck in an overfit state (noise_sd
collapsing to 0.746). Fix `0c1f29d`: zero-init auxiliary groups, lean defaults
(both bags with factors, time terms bias-only). On a 10% holdout subset at
rank 16 over 40 sweeps, posterior mean: plain BPMF 0.8975, lean 0.8887, old
all-groups random init 0.8972.

Relaunched as **B2-p2-r64 / B2-full-r64** (binary
`artifacts/r8/bin/netflix-bayes-0c1f29d`, rank 64, 128 sweeps, 4 threads,
nice 10, ≈87 s/sweep). Judge by `mean_probe_rmse` at snapshots, not single
samples. Gate: standalone ≤0.890 or blend gain ≥0.0005.

B2 posterior-mean RMSE by sweep:

| Sweep | Holdout | Probe |
|---:|---:|---:|
| 32 | 0.87744 | 0.90122 |
| 48 | 0.87265 | 0.89669 |
| 64 | 0.87057 | 0.89480 |

## In progress

- **R8-CM-MT8-B64-s3**: Bet B sweep-64 snapshot on the R8-CM-MT8 bank.
- **MT10**: MT3 config, seed 7, on MPS.
- **A2 transformer sequence encoder** (branch `bet-a2-sequence`,
  `scripts/masked_sequence.py`): holdout selection only on MPS; refit added if
  the curve is competitive with MT2/MT3.
