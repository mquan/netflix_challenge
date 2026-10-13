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
