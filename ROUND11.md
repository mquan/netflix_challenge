# Round 11: day-specific user factors inside Bayesian timeSVD++

Branch `claude-bayes-session-factors` (worktree
`.claude/worktrees/claude-bayes-session-factors`), started 2026-10-12 from
`round4-combined` at 075c931. The Bayes sampler (`src/bayes.hpp`, tests and
build wiring) is copied from `round9-breakthrough` 91fde00. The round-9 checkout
is not touched.

Reference: **R9-LAT-A2h, 0.8668706 dev CV**. Target ≈0.8646, so the gap is ≈0.0023.

## Why this axis

- **Same-day context is the strongest signal this project has found.** Giving
  the masked-history net same-day implicit movies and visible same-day ratings
  (MT1, ROUND7) paid −0.0010 in the blend, the largest single gain since round 4.
  Probe rows are each user's latest ratings, and many of them fall on days that
  also have training ratings.
- **No factor model in the bank models that signal well.** BellKor's p_u,t
  (day-specific user factors, eq. 14) is the main term that separates their
  timeSVD++ from SVD++. Our SGD attempt (J4 `--day-factors 128`) overfits after
  one epoch, which is expected: most user-days have only a few ratings.
- **This weakly identified regime is what MCMC is for.** The sampler already has
  a `session` group (one-hot user-day). Every Bayes run so far (B2–B5) used it
  bias-only (`--session 1`, b_u,t). `--session 2` adds the p_u,t factors and has
  never been run on its own. The only evidence is the bet-b ablation (0c1f29d),
  which turned on factors for *all* time groups at once, on a 10% user subset at
  rank 16: 0.8892, vs. 0.8887 for the lean defaults. That does not isolate
  session factors.
- **It is a new factor-model mechanism, not another Bayes setting.** The settings
  that saturated (rank, item-bin, drift, item-freq) all refine terms the SGD
  models already have.

Risk: pilot gains shrink a lot by the blend. B5's item-freq pilot was −0.0010 at
rank 16, then −0.00056 standalone at rank 128 and −0.00005 in the blend. The
pilot gate is therefore set high.

## Plan

1. **Pilot (P2, matched to the round-9 B5 pilot).** Full train-matched holdout,
   rank 16, 40 sweeps, burn-in 16, seed 29, 2 threads, `--item-freq 1`.
   Arms: control `--session 1` vs. `--session 2`. Gate: ≥0.002 posterior-mean
   gain, all five folds better.
2. **Full run.** B5 command (rank 128, 160 sweeps, `--item-bin 2 --drift 2
   --item-freq 1`) plus `--session 2`. Memory is about #user-days × 128 × 4 B per
   array. Check it with a 2-sweep pilot and watch swap.
3. **Blend.** Run `scripts/round10_features_run.py`-style LightGBM-s3 + NN2 on the
   frozen R9-LAT-A2h config with the new column added. Promote only if all five
   folds improve.

## Log

- 2026-10-12 22:55: pilot arms `R11-S2-pilot-{control,session}` launched
  (binary `artifacts/r11/netflix-bayes-s2`, outputs `artifacts/r11/S2-pilot-*`).

## Error analysis of the promoted blend (2026-10-12)

Breakdown of R9-LAT-A2h-NN2.avg probe errors (scratch script, not committed):

| Rows | Share of rows | Blend RMSE | Share of squared error |
|---|---:|---:|---:|
| No training rating on the probe day | 63% | 0.893 | 67% |
| Some training rating on the probe day | 37% | 0.806–0.842 | 33% |
| Gap to last training date 31–180 d / >180 d | 29% / 5.7% | 0.916 / 1.009 | — |
| Same-day implicit set = only the row itself | 22% | 0.916 | — |
| Same-day implicit set ≥ 10 movies | — | ≈0.815 | — |

No single column beats the blend in any bin. Per-bin bias is ≤0.009, and refitting
the blend per bin gains <0.00003, so nothing is left in recalibration. Ratings
of 1 and 5 hold 55% of the error. Shrinkage is strongest for users with no same-day
training ratings.

**Consequence for this round.** A one-hot p_u,t factor carries no data for the 63%
of rows without same-day training ratings: it stays at its prior. `--session 2`
can only help the low-error third. The signal that reaches every row is *which
movies* the user rated that day, from implicit train, probe and qualifying pairs.
That is the MT1 signal. Next design, if the pilot shows any life: a
**session-bag** group in the FM. It holds the items in N(u, day) with weight
|N(u,day)|^−½ and factors, organised as relational blocks per user-day, like the
existing user-bag per user.

## S2 pilot: day-specific user factors are dead (2026-10-12)

| Arm (P2, rank 16, 40 sweeps, seed 29) | Posterior-mean RMSE | Fold changes |
|---|---:|---|
| Control `--session 1 --item-freq 1` | 0.8856975 | — (round 9's identical run: 0.8857266) |
| `--session 2` (p_u,t factors) | 0.8855767 | −.000129 +.000022 −.000090 −.000136 −.000271 |

That is −0.00012, far below the 0.002 gate, with fold 1 worse. Not scaled. This
is what the error analysis predicts.

## Session-bag group (2026-10-12)

New FM group `session-bag` in `src/bayes.hpp` (`--session-bag 1|2`, off by
default). Each user-day session is a relational block with
q_s = |N(u,d)|^−½ Σ_{j ∈ N(u,d)} y_j over that day's implicit movies (train,
probe and qualifying). Each training and probe row joins its session's block.
Blocks are sampled between the user and item blocks, with the same exact
residual updates. Bag parameters use Hogwild across items, followed by an exact
recomputation of the caches. Checkpoint magic moves to NFPBAYS3.

Checks: with one thread, the off-path is bit-identical to the old sampler. With
several threads even the old binary differs from run to run, because Hogwild
bags are nondeterministic. Tests now include the session bag in the
residual-exactness test (1 and 3 threads, drift ≈5e-8), the train/probe feature
agreement test and the probe-label isolation test.

Pilot `R11-SB-pilot-{sbag2,sbag1}`: the same P2 configuration as the S2 control,
plus `--session-bag 2` or `1`. Gate as before: ≥0.002 vs. control, all folds.
