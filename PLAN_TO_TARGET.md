# Plan to reach a 10% improvement over Cinematch

**Target.** 10% better than Cinematch's 0.9525 test RMSE, which is
**0.85725 test RMSE**. The Grand Prize scores (0.8554 quiz, 0.8567 test) are
not the target.

**Bottom line.** Our best score is **0.87039 development CV** (R7-MT56-s3).
The Phase 0 rehearsal ([PHASE0_REHEARSAL.md](PHASE0_REHEARSAL.md)) measured
test ≈ dev CV − 0.0074 on our pipeline, so the target is **dev CV ≈ 0.8646**
and the remaining gap is about **0.0058**. It's that small because our probe
numbers don't include the retrain on train+probe that every prize team did. The plan has four phases:

0. Measure the true gap on our own pipeline.
1. Build infrastructure so every model can be retrained for the final submission.
2. Run four ranked model bets.
3. Do the final train+probe production run.

## Where we are

| Milestone | Dev CV (probe) |
|---|---:|
| R5-meta, when the goal was set (2026-09-27) | 0.87481 |
| R6-RBM5-s3: six conditional RBM columns | 0.87312 |
| **R7-MT56-s3: date-aware masked-history network (current best)** | **0.87039** |
| **Target: 0.85725 test (measured offset 0.0074)** | **≈0.8646** |

Rounds 6 and 7 together took about 0.0044 off the score in roughly four days.
Round 7 alone took 0.0027.

## How the target maps onto our metric

- We score on the probe set, with probe ratings removed from training. Prize
  teams retrained on train+probe before predicting the qualifying set. Rendle,
  Zhang and Koren (2019) note that quiz RMSE is typically about 0.007 lower
  than probe RMSE for this reason; the published reports suggest 0.007–0.009
  for blends.
- So 0.85725 test corresponds to about **0.864–0.866 dev CV**.
- **Measured in Phase 0 (2026-10-03):** retraining gains 0.00755 on a
  9-column core bank, and a transferred blender loses only 0.00018 against dev
  CV, so test ≈ dev CV − 0.0074 and 0.85725 ↔ **0.8646**. This is probably
  conservative, since P2 is less "future" than the real probe.
- The qualifying labels were never released, so no local run can produce a
  real test score. The final claim will be a calibrated estimate.

## Phase 0 — Measure the real gap (about 3 days)

This is a dress rehearsal of the final submission, using a second probe (P2).

1. Build P2 from training: for each probe user, take their latest ≈3 training
   ratings before their probe dates. This mirrors how Netflix cut the probe. It
   only uses dates and user IDs, never rating values (same idea as
   `scripts/train_holdout.py`).
2. Train a core bank of 10–12 columns (best joint, kNN, RBM, masked-history
   MT1/MT2) on train minus P2. Fit the `round4_boost.py` blender on P2.
3. **(a)** Predict probe with those models. **(b)** Retrain the same models on
   full train (P2 included) at fixed epochs, predict probe, and apply the
   P2-fitted blender.
4. Score both on probe. b − a is **our own train+probe gain**, which calibrates
   the target. Score b is also an **honest held-out number**, since probe played
   no part in any choice in this pipeline.

What this phase delivers: a calibrated dev-CV target, a tested Stage C
procedure, and a check on how optimistic our repeatedly probe-selected dev CV
has become.

## Phase 1 — Infrastructure for scale (runs in parallel with Phase 0)

- **Column registry.** One manifest per column: command, dataset variant,
  selected epoch, seed, and hashes. A runner regenerates any column on three
  datasets: train (dev CV), train minus P2 (rehearsal), and train+probe
  (qualifying predictions). The final submission needs every column retrained;
  that's about 70 columns today, and nothing currently writes qualifying predictions.
- **Compute: local only (decided 2026-10-02).** Everything runs on this Mac,
  on the GPU where it works and the CPU otherwise. The 14 CPU cores are shared
  by several sessions (load average 20–26 when checked). The machine has an
  M3 Max GPU (30 cores), and PyTorch 2.14.1 in the masked-history venv reports
  MPS as available. The masked-history network currently runs on CPU
  (`device='cpu'`, 2 threads, ≈10 min/epoch), so the GPU is idle.
  - First step: a throughput pilot (`--max-batches`) of MT1 on `--device mps`
    against CPU. Check that the scores match CPU closely and that every op is
    supported on MPS. Any speedup is unmeasured until this runs.
  - If MPS works, move all PyTorch families (masked history, its item-side
    twin, the sequence encoder, the neural blender) to the GPU. The C++ models
    (joint, kNN, RBM, Bayesian timeSVD++) stay on CPU. That splits the work
    across two resources that don't compete.
  - If MPS doesn't work, the PyTorch arms stay on CPU and the milestones below
    stretch. In that case, prioritize bets by compute cost: A3 and D are cheap,
    A1, A2 and B are expensive.
- **Probe hygiene.** Keep selecting epochs on the train-matched holdout, not the
  probe (the round-7 protocol).

## Phase 2 — Model bets, ranked by expected value

### Bet A: Scale the masked-history family (proven axis)

Evidence: round 7 took 0.0027 off the blend; session/date context improved the
standalone model from 0.9110 to 0.8989.

1. **Item-side twin.** Encode the target movie's other dated raters (stage 4 of
   RESEARCH_NEXT.md, never run).
2. **Sequence encoder.** Replace the bag-of-embeddings + session MLP with a
   transformer over the date-ordered history. Keep masked targets and the 50/50
   mix of random and suffix masks.
3. **Distribution features for the blender.** Give the blender P(1..5) and
   entropy. They are already computed.
4. **Seed and masking-regime replicates.** Each one paid ≈0.00015. Run them only
   on spare compute.

Plausible gain: **0.0015–0.003**.

### Bet B: Close the factor-model quality gap with a different learner

Evidence: our best factor model scores 0.8941 on probe. Published timeSVD++
reached 0.8762 quiz (≈0.883 probe-equivalent), so we are about 0.01 behind
standalone. Round 6 concluded the factor axis was saturated, but it only tested
our own SGD models.

- **Bayesian (Gibbs-sampled) timeSVD++ "flipped".** "Flipped" adds an item-side
  implicit bag of the users who rated each movie. On MovieLens 10M, Rendle,
  Zhang and Koren (2019) found it beat every published method, including
  CF-NADE and AutoRec. That paper doesn't test it on Netflix, so this is a bet.
  Posterior averaging also gives errors that differ from our SGD models.
- Gate: standalone ≤0.890, or a blend gain ≥0.0005.

Plausible gain: **0.0005–0.0015**.

### Bet C: Complete the neighborhood family

Evidence: residual kNN columns carried ≈85% of GBDT gain importance (B6). The
repo has no user-user kNN, and the integrated model I2 was stopped at 0.8951
while still improving.

- Add user-user kNN on residuals of the joint and masked-history models.
- Run I2 to convergence with multithreading.

Plausible gain: **0.0003–0.001**.

### Bet D: Blender

- A neural blender over raw columns + distribution features, averaged 50/50
  with LightGBM. Fixed averaging beat both components in COMBINED_BRANCHES.md.
- Three blender seeds as the standard for any promoted result.

Plausible gain: **0.0002–0.0008**.

### Gap budget

| Bet | Plausible dev-CV gain |
|---|---|
| A — masked-history scale-out | 0.0015–0.003 |
| B — Bayesian timeSVD++ flipped | 0.0005–0.0015 |
| C — neighborhood completion | 0.0003–0.001 |
| D — blender | 0.0002–0.0008 |
| **Total** | **0.0025–0.0063** vs. a ≈0.0058 gap |

These ranges are judgment calls based on the per-round gains so far, not
measurements. Read honestly: the middle of the combined range (≈0.0045) closes
most of the gap, and reaching 0.8646 needs A and B to land in the upper half of
their ranges, or one more axis.

## Phase 3 — Final production run (Stage C)

1. Freeze every architecture, epoch and blender setting.
2. Retrain each retained column on train+probe at its fixed epoch, keeping
   qualifying pairs as implicit feedback.
3. Predict the qualifying set, apply the probe-fitted blender, and clip to [1, 5].

The qualifying labels were never released, so the final number is an estimate
calibrated by the Phase 0 rehearsal, not a test score.

## Operating rules

- **Rounds of 2–3 days**, with at most 4 parallel arms each. Each arm uses the
  round-7 protocol: a holdout selection run plus a concurrent refit that saves
  every epoch.
- **Promotion:** an arm is promoted if it improves all five folds against the
  current best bank (single blender seed 2026), confirmed with three seeds. An
  architecture gets more compute only if it gains ≥0.001.
- **Kill rule:** drop a converged arm that fails its gate. Freeze a family that
  gains under 0.0003 across two rounds.
- **Ledger:** keep one ROUNDn.md per round, plus the column registry.

## Milestones (rough; they depend on the MPS pilot)

| Milestone | Content | Target dev CV |
|---|---|---|
| M0 ✅ 2026-10-03 | Rehearsal done: offset 0.0074, target 0.8646 (registry not built yet) | — |
| M1 (≈day 10) | Bets A1–A3 done, Bayesian timeSVD++ running | 0.868 |
| M2 (≈day 20) | Bets B and C done, blender upgrade | 0.866 |
| M3 | Calibrated target reached; Stage C run | ≈0.8646 |

## Decisions needed

1. ~~Target~~ — decided: 10% over Cinematch (0.85725 test, ≈0.865 dev CV).
2. ~~Compute~~ — decided: local GPU (MPS) where it works, CPU otherwise.
3. ~~Order~~ — Phase 0 done first; see [PHASE0_REHEARSAL.md](PHASE0_REHEARSAL.md).

## Sources

- Rendle, Zhang, Koren (2019), *On the Difficulty of Evaluating Baselines*:
  [arXiv 1905.01395](https://arxiv.org/abs/1905.01395)
- Repo notes: [ROUND6.md](ROUND6.md), [ROUND7.md](ROUND7.md),
  [RESEARCH_NEXT.md](RESEARCH_NEXT.md), [RESULTS.md](RESULTS.md),
  [HOLDOUT.md](HOLDOUT.md), [COMBINED_BRANCHES.md](COMBINED_BRANCHES.md)
