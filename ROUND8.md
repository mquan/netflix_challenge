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
| 96 | — | 0.89355 (sweep 102: 0.89318) |
| 128 | **0.86803** | **0.89269** |

Final rank-64 column 0.892686 probe, better than K1 (0.894117), the best
SGD factor model. Holdout still improving at sweep 128, so 128 is selected.

| Bank | Dev CV | Δ | Fold Δ | Verdict |
|---|---:|---:|---|---|
| **R8-CM-MT8-B64-s3**: R8-CM-MT8 + B2 sweep-64 snapshot | **0.8694494** | **−0.0007320** vs R8-CM-MT8-s3 | −.000648 −.000660 −.000726 −.000736 −.000891 | **Retained; passes the 0.0005 gate** |

Largest single-arm gain of round 8. Calibrated test estimate ≈0.8620
(9.50% better than Cinematch's 0.9525 test RMSE); ≈0.0048 left to ≈0.8646.

On the R8-CM-MT89 bank (0.8701238), three blender seeds:

| Bank | Dev CV | Δ vs R8-CM-MT89-s3 | Fold Δ | Verdict |
|---|---:|---:|---|---|
| R8-CM-MT8-B64-s3 (no MT9, sweep 64) | 0.8694494 | −0.0006744 | −.000589 −.000655 −.000606 −.000700 −.000822 | Superseded |
| **R8-MT89-B128-s3**: + B2 sweep 128 | **0.8692319** | **−0.0008919** | −.000985 −.000794 −.000734 −.000920 −.001027 | **Retained; new best** |
| R8-MT89-B128-MT10-s3: + B2 sweep 128 + MT10 (MT3 config, seed 7, epoch 7) | 0.8692484 | −0.0008754 | −.000856 −.000756 −.000823 −.000877 −.001065 | MT10 not retained (+0.0000165 vs R8-MT89-B128) |

**Current best: R8-MT89-B128-s3 0.8692319** (−0.0011574 vs R7-MT56-s3).
Calibrated test estimate ≈0.8618 (9.52% better than 0.9525); ≈0.0046 left
to ≈0.8646. MT10 adding nothing suggests MT seed replicates are saturated.

## A2 — transformer sequence encoder

Branch `bet-a2-sequence` (`scripts/masked_sequence.py`, MT2-style masking).
Slow start, then catches up after epoch 6:

| Epoch | 1 | 3 | 5 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2 holdout | 0.91683 | 0.89972 | 0.89198 | 0.88798 | 0.88364 | **0.88348** | 0.88362 |
| MT2 holdout | 0.90077 | 0.88605 | 0.88125 | 0.87824 | 0.87795 | 0.87779 | 0.87793 |

Selection finished at epoch 11 (0.88235; epoch 12 0.88402). A2-refit's
final-epoch probe RMSE is 0.90708 (epoch 12).

On the R8-MT89-B128 bank (0.8692319), three blender seeds:

| Bank | Dev CV | Δ | Fold Δ | Verdict |
|---|---:|---:|---|---|
| **R8-B128-A2-s3**: + A2 refit epoch 11 | **0.8686353** | **−0.0005966** | −.000546 −.000606 −.000608 −.000723 −.000500 | **Retained; new best** |
| R8-B128-B3s80-s3: + B3 (rank 128, seed 7) sweep-80 snapshot | 0.8688725 | −0.0003593 | −.000387 −.000372 −.000452 −.000285 −.000300 | Retained (on top of the rank-64 column) |

A2 trails MT2 by 0.005 standalone yet adds as much as Bet B: the attention
encoder's errors differ from the bag-based MT family. A1 (0.925 standalone)
added nothing, so closeness to the MT family's accuracy seems to matter.

**Current best: R8-B128-A2-s3 0.8686353** (−0.0017540 vs R7-MT56-s3).
Calibrated test estimate ≈0.8612 (9.58% better than 0.9525); ≈0.0040 left.

## In progress

- **R8-A2-B3s80-s3**: both retained arms together.
- **B3-p2-r128 / B3-full-r128**: rank 128, 160 sweeps; sweep 105 probe
  0.89128, holdout 0.86618 at sweep 101.
- **A2 transformer sequence encoder** (branch `bet-a2-sequence`,
  `scripts/masked_sequence.py`): holdout selection only on MPS; refit added if
  the curve is competitive with MT2/MT3.
