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

| Bank | Dev CV | Δ vs R8-B128-A2-s3 | Fold Δ | Verdict |
|---|---:|---:|---|---|
| **R8-A2-B3s80-s3**: R8-B128-A2 + B3 sweep-80 snapshot | **0.8682695** | **−0.0003658** | −.000372 −.000431 −.000360 −.000329 −.000337 | **Retained; new best** |

A2 and B3 stack fully (B3 alone added −0.0003593).

**Current best: R8-A2-B3s80-s3 0.8682695** (−0.0021198 vs R7-MT56-s3).
Calibrated test estimate ≈0.8609 (9.62% better than 0.9525); ≈0.0037 left.

B3 finished at sweep 160: probe **0.890438** (best single factor model),
holdout 0.86517 at sweep 152 and still improving, so sweep 160 is selected.
A2b (A2 with `--suffix-prob 0`) selected epoch 10, holdout **0.88052**
(A2: 0.88235).

On the R8-A2-B3s80 bank (0.8682695), three blender seeds, 4 processes:

| Bank | Dev CV | Δ | Fold Δ | Verdict |
|---|---:|---:|---|---|
| R8-A2b-s3: + A2b epoch 10 | 0.8679934 | −0.0002761 | −.000353 −.000310 −.000316 −.000226 −.000176 | Retained |
| R8-B3f-s3: B3 sweep 80 → sweep 160 | 0.8681450 | −0.0001245 | −.000149 −.000149 −.000174 −.000054 −.000096 | Retained |
| **R8-A2b-B3f-s3**: both | **0.8678750** | **−0.0003945** | −.000452 −.000466 −.000364 −.000400 −.000290 | **Retained; new best** |

As with MT1/MT2, the random- and suffix-masked A2 variants are complementary.

**Current best: R8-A2b-B3f-s3 0.8678750** (−0.0025143 vs R7-MT56-s3).
Calibrated test estimate ≈0.8605 (9.66% better than 0.9525); ≈0.0033 left
to ≈0.8646.

## Stopped / paused (2026-10-04)

- **A2c** (A2b at 4 layers, d_model 128, ff 512; selection + refit at once on
  MPS) was killed at selection epoch 8 / refit epoch 5. The two processes
  held 110+ GB of unified memory on the 96 GB machine (MPS attention keeps
  full attention matrices); swap reached 38/39 GB and load ≈110. Its holdout
  curve was not ahead of A2b at matched epochs (epoch 7: 0.88772 vs 0.88720;
  epoch 6: 0.89043 vs 0.88886), so capacity is not what limits A2. **Not
  retried.**
- **B4** (rank 128 + item-bin and drift factors, seed 13) was paused at
  sweep 32; a reboot then killed it. At sweep 32 it led B3 slightly
  (holdout mean 0.87495 vs 0.87585; probe 0.89964 vs 0.90053).

## Restart (2026-10-04, afternoon)

The scratchpad venv used by earlier sessions was gone. It's rebuilt at
`~/.cache/netflix-venv` (Python 3.12, torch 2.14.1, LightGBM 4.7.0,
NumPy 2.5.3) so it survives session cleanup.

- **B4r-p2-r128 / B4r-full-r128**: B4 resumed from its sweep-32 checkpoints
  (`--init-model artifacts/bayes/B4-*-r128.sweep32.bin`; rerunning without
  `--init-model` starts from scratch). 5 threads each, ≈180 s/sweep.
- **A2e** (A2b with `--seq-len 768`, twice the history context for heavy
  raters), selection + refit on MPS. Pilot: 200 batches in 75 s, peak MPS
  driver memory ≈25 GB per process (A2b: ≈12.5 GB). Over a full epoch the
  driver cache grew to 34–36 GB per process; with another session's 22 GB
  process, swap reached 25 GB, so **A2e-refit was stopped** in epoch 1.
  A2e-select continues alone; the refit runs after it if the curve beats
  A2b, at the selected epoch count.

### Bet D — neural residual blend averaged with LightGBM

`scripts/round8_nn_blend.py` reads a round4_boost.py report (`--like`) and
rebuilds exactly its inputs (hash-checked; its ridge control matches to every
digit). Per outer user fold, an MLP learns the contextual-ridge residual from
the prediction columns' mean and deviations plus every robust-standardized
tree feature; epochs are chosen on round4_boost's inner user split, then
refit. The result is averaged 50/50 (predeclared) with the report's
cross-fitted LightGBM column.

On the R8-A2b-B3f bank (LightGBM 0.8678750, three blender seeds):

| Arm | NN alone | 50/50 with LightGBM | Δ | Fold Δ | Verdict |
|---|---:|---:|---:|---|---|
| R8-NN1: 2×128 GELU, lr 1e-3, dropout .1, one seed (epochs 6 7 4 7 7) | 0.8683618 | 0.8678037 | −0.0000713 | −.000094 −.000047 +.000001 −.000081 −.000136 | Four folds better, fold 2 flat; not promoted |
| **R8-NN2**: lr 3e-4, dropout .2, three NN seeds (epochs 15–28) | 0.8679367 | **0.8676503** | **−0.0002248** | −.000243 −.000213 −.000242 −.000158 −.000268 | **Retained; new best** |
| R8-NN3: 3×256, lr 3e-4, dropout .3, weight decay .05, batch 8192, three seeds | 0.8679811 | 0.8676862 | −0.0001888 | −.000258 −.000144 −.000156 −.000159 −.000227 | Not retained (worse than NN2) |

Averaging the NN2 and NN3 networks before the 50/50 gives 0.8676572, no
better than NN2 alone. Regularized and seed-averaged, the network alone
(0.86794) nearly matches LightGBM (0.86788), and their errors differ enough
that the predeclared average pays on every fold. From now on, each bank is
scored as LightGBM-s3 averaged with NN2-config `round8_nn_blend.py`.

**Current best: R8-NN2 (R8-A2b-B3f-s3 LightGBM ⊕ NN2) 0.8676503**
(−0.0027390 vs R7-MT56-s3). Calibrated test estimate ≈0.8603 (9.69% better
than 0.9525); ≈0.0031 left to ≈0.8646.

### B4 and A2e finished (2026-10-05)

- **B4** (rank 128 + item-bin and drift factors, seed 13), sweep 160:
  probe **0.889660** (best factor model; B3 0.890438), holdout 0.86421
  (B3 0.86517), still improving, so sweep 160 is selected.
- **A2e** (A2b with `--seq-len 768`) selection, holdout by epoch:
  0.91219 0.90009 0.89702 0.88982 0.88527 0.88134 0.88022 0.87690 0.87617
  0.87576 0.87610 **0.87519** 0.87648 0.87580; epoch 12 selected. A2b's best
  was 0.88052, so doubling the history window gains 0.0053 standalone, the
  largest standalone improvement in the A2 family. Refit (12 epochs, one MPS
  process) started 2026-10-05 07:20.

### Stopped before closing the machine (2026-10-05 ≈07:55)

Both jobs were stopped and restart from scratch (each had ≈30 min of
progress; neither resumes mid-run). Before relaunching, check
`ps aux | grep -E 'masked_sequence|round4_boost'` first.

1. **R8-B4-s3** (≈1.5 h): R8-A2b-B3f-s3's command plus
   `artifacts/r8/B4-full-r128.s0160.f32` as the last `--extra`, output
   `artifacts/r8/R8-B4-s3`. To restart, move aside `experiments/R8-B4-s3.*`
   (the script refuses an existing output `.json`) and rerun the command
   recorded in `experiments/R8-B4-s3.json` with
   `~/.cache/netflix-venv/bin/python`. Then score it as
   `scripts/round8_nn_blend.py --like artifacts/r8/R8-B4-s3.json --output
   artifacts/r8/R8-B4-NN2 --lr 3e-4 --dropout .2 --seeds 2026 7 13 --epochs
   60 --patience 5` and compare with R8-NN2 (0.8676503).
2. **A2e-refit** (≈6–8 h, one MPS process): `rm -f artifacts/masked/A2e-refit*`
   and move aside `experiments/A2e-refit.*`, then rerun the command in
   `experiments/A2e-refit.json`. Blend `artifacts/masked/A2e-refit.refit.e12.f32`
   onto the best bank (LightGBM-s3, then NN2).

Relaunched both from scratch on 2026-10-05 ≈20:35 local; the stopped records
are kept as `experiments/R8-B4-s3-stopped.*` and `experiments/A2e-refit-stopped2.*`.
The A2e refit's first batch reproduces the stopped run's train RMSE (0.94454).

### Both relaunched jobs finished (2026-10-06)

- **R8-B4-s3** (finished 2026-10-05 21:19): LightGBM dev CV **0.8677590**,
  −0.0001160 vs R8-A2b-B3f-s3 (0.8678750); fold Δ −.000091 −.000052 −.000259
  −.000031 −.000148, all five better. Retained as the new LightGBM bank. Not yet
  comparable with the overall best R8-NN2 (0.8676503) until it is scored with
  NN2 (`round8_nn_blend.py --like artifacts/r8/R8-B4-s3.json --output
  artifacts/r8/R8-B4-NN2 …`, as above).
- **A2e-refit**: all 12 epochs completed (epoch 12 train RMSE 0.81134);
  `artifacts/masked/A2e-refit.refit.e12.f32` is ready but not yet blended.

**Current best is unchanged: R8-NN2 0.8676503.** Next: (1) NN2 on R8-B4-s3;
(2) R8-B4-s3's command plus A2e e12 as an `--extra` (output
`artifacts/r8/R8-B4-A2e-s3`, ≈1.5 h), then NN2 on that. Swap was at 24.5 of
25.6 GB with no jobs running, so check memory before launching both together.

### Next two jobs launched (2026-10-06)

- **R8-B4-NN2**: NN2 config (`round8_nn_blend.py --like artifacts/r8/R8-B4-s3.json`,
  lr 3e-4, dropout .2, seeds 2026 7 13, 2 threads), ≈2 h. Compare with R8-NN2
  (0.8676503).
- **R8-B4-A2e-s3**: R8-B4-s3's command plus `artifacts/masked/A2e-refit.refit.e12.f32`
  as the last `--extra` (77 prediction columns), ≈1.5 h; then NN2 on it.

### R8-B4-A2e-s3 finished (2026-10-06)

LightGBM dev CV **0.8676311**, −0.0001279 vs R8-B4-s3 (0.8677590); fold Δ
−.000125 −.000103 −.000040 −.000175 −.000196, all five better. The A2e column
pays on top of A2/A2b, so it is retained as the new LightGBM bank. R8-B4-A2e-NN2
(NN2 on this bank) was launched right away, alongside the still-running R8-B4-NN2.

### R8-B4-NN2 finished (2026-10-06)

NN2 on R8-B4-s3: NN alone 0.8678560, LightGBM 0.8677590, 50/50 **0.8675498**,
−0.0001005 vs R8-NN2 (0.8676503); fold Δ −.000046 −.000021 −.000210 −.000056
−.000171, all five better. **New best: R8-B4-NN2 0.8675498** (calibrated test
estimate ≈0.8601, 9.70% better than 0.9525; ≈0.0029 left to ≈0.8646).
R8-B4-A2e-NN2 (same scoring on the 0.8676311 A2e bank) is still running.

### R8-B4-A2e-NN2 finished (2026-10-06)

NN2 on R8-B4-A2e-s3: NN alone 0.8677031, LightGBM 0.8676311, 50/50
**0.8674131**, −0.0001367 vs R8-B4-NN2 (0.8675498); fold Δ −.000171 −.000165
−.000089 −.000113 −.000145, all five better.

**Current best: R8-B4-A2e-NN2 0.8674131** (−0.0029762 vs R7-MT56-s3).
Calibrated test estimate ≈0.8600 (9.71% better than 0.9525); ≈0.0028 left to
≈0.8646. Both new columns (B4 drift factors, A2e 768-step history) paid
through LightGBM and again through NN2.

### A2f (`--seq-len 1536`) pilot and selection launch (2026-10-06)

A2e's 768-step window gained 0.0053 standalone, so A2f doubles it again. Since
attention memory per batch grows with sequence length, A2f halves
`--batch-tokens` to 32768. Pilot (300 batches, `experiments/A2f-pilot.*`): MPS
driver memory 34.4 GB at batch 300 (A2e: 25 GB), swap flat, ≈575 users/s.
By the ≈1.4× growth rule, budget ≈48 GB, so run it as the only MPS process.
**A2f-select** launched: A2e-select's command with `--seq-len 1536
--batch-tokens 32768 --epochs 18 --patience 3`. Compare per-epoch holdout with
A2e (best 0.87519 at epoch 12). Note that batch size also changes, so the
comparison is not purely about window length.

A Grand Prize gap audit (`GAP_AUDIT.md`) is being written in parallel.

A2f-select holdout by epoch so far: 0.90729 0.89307 0.88197 0.87627
**0.87328** (epoch 5). That already beats A2e's best (0.87519 at epoch 12) by
0.0019, and the run is still improving. Driver memory ≈49 GB; swap 28 GB, but
29% is free and load is ≈7 (no thrashing).

### A2f-select finished (2026-10-07)

Holdout by epoch: 0.90729 0.89307 0.88197 0.87627 0.87328 0.87093 0.87412
0.87001 0.87091 **0.86860** 0.87040 0.87118 0.87223; epoch 10 selected (4.7 h).
That is −0.0066 vs A2e (0.87519) and the largest standalone gain in the A2
family. History length keeps paying: 384→768 gained −0.0053 and 768→1536
gained −0.0066. **A2f-refit** (10 epochs, `--save-epochs`) launched. Then:
R8-B4-A2e-s3's command plus `artifacts/masked/A2f-refit.refit.e10.f32`
(LightGBM-s3), then NN2.

### A2f-refit finished (2026-10-07)

10 epochs in 3.2 h. Refit probe RMSE **0.89510** (A2e 0.89859, A2b 0.90204).
**R8-A2f-s3** launched: R8-B4-A2e-s3's command plus
`artifacts/masked/A2f-refit.refit.e10.f32` (78 prediction columns), then NN2.

### R8-A2f-s3 finished (2026-10-07)

LightGBM dev CV **0.8675304**, −0.0001007 vs R8-B4-A2e-s3 (0.8676311); fold Δ
−.000082 −.000101 −.000143 −.000110 −.000067, all five better. Retained.
R8-A2f-NN2 launched.

### R8-A2f-NN2 finished (2026-10-07)

NN2 on R8-A2f-s3: NN alone 0.8675901, LightGBM 0.8675304, 50/50 **0.8673301**,
−0.0000830 vs R8-B4-A2e-NN2 (0.8674131). Fold Δ +.000016 −.000034 −.000143
−.000192 −.000062: four folds better, fold 0 flat or slightly worse. This fails
the predeclared all-five-folds rule, so **R8-B4-A2e-NN2 (0.8674131) stays the
promoted best**; R8-A2f-NN2 is the best by mean. The LightGBM step did
improve on all five folds. NN2's gain over LightGBM shrank here (−0.000200 vs
−0.000218).

### A2g (`--seq-len 3072`) pilot fails; deprioritised (2026-10-07)

Pilot with `--batch-tokens 16384`: driver memory 50 GB by batch 100, then a
crash in backward with `MPSGraph does not support tensor dims larger than
INT_MAX`. Running at 3072 would need chunked or checkpointed attention in
`masked_sequence.py`. History length per user (train + implicit) compared
with the window:

| Window | Users with more ratings than the window | Probe rows affected |
|---|---:|---:|
| 384 | 32.6% | 33.3% |
| 768 | 16.4% | 16.8% |
| 1536 | 5.2% | 5.3% |
| 3072 | 0.7% | 0.7% |

Going 768→1536 changed the context for 16.8% of probe rows; 1536→3072 would
change only 5.3%, so A2g's ceiling is roughly a third of A2f's gain. Not
pursued for now. Open question: A2f also halved `--batch-tokens`, so part of
its −0.0066 may come from smaller batches (more optimizer steps), not the
longer window. An A2e-with-`--batch-tokens 32768` arm would separate the two.

**A2h-select** launched: A2f with `--batch-tokens 16384`. It tests the batch
size question above directly at seq-len 1536. Compare per-epoch holdout with
A2f (best 0.86860 at epoch 10).

### Bet 1 overlaps with another session (2026-10-07)

`scripts/round9_latent_features.py` (+ `tests/test_round9_latent.py`) exports
24 label-free K1-joint256 columns to `artifacts/r9/lat/`: 8+8 projections onto
the top item singular vectors, the 6 exact bias terms and the full dot product.
It rebuilds K1's probe prediction on all 1,408,395 rows (max |Δ| 2.1e-6).
Meanwhile a separate checkout (`../netflix_round9`, branch
`round9-breakthrough`) is already running the same arm, R9-LAT-s3 (promoted
R8 bank + 8 user/8 item PCs + 6 bias terms). To avoid duplicating that work,
this session does **not** launch its own LightGBM run of these features. If
R9-LAT pays, `lat-dot` and `lat-bif` are the extras this export adds.

### Bet 3 (ENNBlend third blender): dead on the fold-0 smoke run (2026-10-07)

`scripts/round9_enn_blend.py` trains 400 tiny nets as one batched model, each
on 4 random prediction columns plus log user and item support, with 8 hidden
units, on the ridge residual. They are stacked by a ridge for each user-support
quartile, fit on inner-valid users only. Its hash checks and ridge control
match R8-A2f-s3 exactly. On fold 0 (128 nets): ridge 0.8699187, **ENN alone
0.8698794** (−0.00004 vs ridge; LightGBM −0.0013, NN2 −0.0012), ENN ⊕ LightGBM
0.8690705, the predeclared 1/3-each average **0.8687859 vs NN2 ⊕ LightGBM
0.8684651 (+0.0003 worse)**. The rating target was no better. Nets that see 4
of the 78 columns cannot find much that the ridge has not already used. The full
5-fold run is not launched. Not retained.

Ownership: the `round9-breakthrough` session (`../netflix_round9`) owns bets 1
and 2. This session leaves them alone and continues the A2 sequence axis.

### A2h-select finished; refit and blend chain launched (2026-10-08)

A2h (1536-step history, 16384 batch tokens) completed successfully. Holdout
RMSE by epoch: 0.90575 0.88868 0.87732 0.87514 0.87344 0.87010 0.86966
0.86890 0.86744 0.86815 0.86747 **0.8674023** 0.86804 0.86896 0.86767.
Epoch **12** is selected, improving on A2f's 0.8685955 by **0.0011932**.
This supports smaller batches at a fixed history length; it does not isolate
how much of the earlier A2e→A2f gain came from batch size versus window length.

`scripts/round8_a2h_chain.py` now runs the following dependent stages, stopping
on any failure and refusing existing experiment/output prefixes:

1. **A2h-refit:** train on the full development training set for exactly 12
   epochs, with probe ratings excluded. Same configuration and source as the
   selection run; save each epoch, but use only the holdout-selected epoch 12.
2. **R8-A2h-s3:** R8-A2f-s3's frozen bank plus
   `artifacts/masked/A2h-refit.refit.e12.f32`, using the same three LightGBM seeds.
3. **R8-A2h-NN2:** the existing NN2 configuration with three seeds and its
   predeclared 50/50 average with LightGBM. Compare every fold against both
   the promoted R8-B4-A2e-NN2 and best-by-mean R8-A2f-NN2 controls.

The chain's preflight verified the selection source/dependency hashes, both
datasets, and every frozen control-bank input. Plan checks verified the fixed
epoch and sole additional column; a simulated stage failure verified that
later stages do not launch. No model or blender algorithms changed.

Live status: `experiments/A2h-chain.status.json`; combined log:
`artifacts/r8/A2h-chain.out`; individual stage logs: `experiments/<name>.log`.
The chain runs detached and sequentially (one MPS trainer at a time), followed
by the CPU blenders. Completion writes the score, fold deltas, and whether the
all-five-fold promotion rule passed into the status file. Budget roughly
6–8 hours of awake runtime based on the selection run; actual time may vary.

**Promoted best remains R8-B4-A2e-NN2, 0.8674131 development CV.** A2h's
ensemble evaluation is pending; its holdout score is a different metric.

### A2h ensemble completed and promoted (2026-10-09)

All three stages completed successfully. **R8-A2h-NN2 reached 0.8670985
development CV**, improving on R8-B4-A2e-NN2 by 0.0003147 and R8-A2f-NN2 by
0.0002317. Its fold deltas versus R8-B4-A2e-NN2 were −0.0003994, −0.0002457,
−0.0003311, −0.0004373, and −0.0001592. All five folds improved, so it passes
the predeclared promotion rule and is the new promoted best. These are
development CV results, not an independent test score. Exact values and stage
provenance are recorded in `experiments/A2h-chain.status.json` and the three
stage experiment records.

### A2i chain launched: suffix-masked A2h (2026-10-09)

Next A2 arm. Batch size and capacity are both poor bets now. A2h's halved
batch doubled the time per epoch (A2f 1028 s → A2h 2249 s), so 8192 batch tokens
would need roughly 16 h of selection and a 15 h refit. A2c already showed that a
larger model does not help. A2 (suffix-masked) and A2b (random-masked) stacked
as complementary columns, but every long-history column since A2e uses A2b's
`--suffix-prob 0`. The bank's only suffix-masked column is still the 384-step A2.

**A2i** = A2h's command with `--suffix-prob .5` and nothing else changed.
`scripts/round8_a2i_chain.py` runs four dependent stages, stopping on any
failure and refusing existing names or output prefixes:

1. **A2i-select** (holdout epoch selection, 18 epochs max, patience 3).
2. **A2i-refit** at the selected epoch, with probe ratings excluded.
3. **R8-A2i-s3:** R8-A2h-s3's frozen bank plus the A2i column, same three LightGBM seeds.
4. **R8-A2i-NN2:** the existing NN2 configuration and 50/50 average.
   Compared per fold against the promoted R8-A2h-NN2 (promotion rule: all
   five folds better) and against R8-B4-A2e-NN2.

The preflight verified A2h's selection source, data hashes, and every
R8-A2h-s3 control-bank input. A simulated run of stages 2–4 on A2h's selection
confirmed that the epoch is fixed and that only one column is added. Startup:
MPS driver memory 37.7 GB at batch 200, swap flat at 3.5 GB, about 38 s per 100
batches (A2h was similar). Budget roughly 15–18 h of awake runtime.
Status: `experiments/A2i-chain.status.json`; log: `artifacts/r8/A2i-chain.out`.

**Promoted best remains R8-A2h-NN2, 0.8670985 development CV.**

### A2i finished: no gain, not promoted (2026-10-10)

The chain completed all four stages. A2i-select holdout by epoch: 0.90621 0.88733
0.87872 0.87650 0.87390 **0.87101** 0.87238 0.87294 0.87147. Epoch 6 is
selected, 0.0036 worse than A2h (0.86740), and the curve is noisier after the
peak. R8-A2i-s3 LightGBM 0.8672849 vs R8-A2h-s3 0.8672909 (−0.000006; fold Δ
+.000027 −.000019 +.000038 −.000009 −.000068). **R8-A2i-NN2 0.8670796**, −0.0000189
vs R8-A2h-NN2. Fold Δ +.000043 −.000047 +.000007 +.000036 −.000133: three folds
are worse, so it fails the promotion rule. At 1536 steps the suffix-masked
objective adds nothing beyond A2h. Not retained.

**Promoted best remains R8-A2h-NN2, 0.8670985 development CV.**

### A2j chain launched: A2h with half the batch (2026-10-10)

**A2j** = A2h's command with `--batch-tokens 8192`. Nothing else changes: still
1536-step history and `--suffix-prob 0`. It continues the trend from 32768 to
16384 tokens (A2f 0.86860 → A2h 0.86740 holdout).
`scripts/round8_a2j_chain.py` is the A2i chain with only that setting changed.
It runs selection, then a fixed-epoch refit, then R8-A2h-s3 + A2j LightGBM, then
NN2. Promotion is judged against R8-A2h-NN2 (all five folds better).
Startup: MPS driver memory 42 GB at batch 700, swap falling, about 227 users/s.
That matches A2h's 217 users/s, so an epoch should take roughly as long as A2h's
(≈40 min), not twice as long. Status: `experiments/A2j-chain.status.json`;
log: `artifacts/r8/A2j-chain.out`.
