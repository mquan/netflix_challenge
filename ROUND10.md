# Round 10: Bayesian frequency group at full scale (B5)

Started 2026-10-10. Control: **R9-LAT-A2h, 0.8668706 development CV** (A2h's
79-column bank plus 22 K1 latent/bias features, LightGBM-s3 ⊕ NN2 50/50; branch
`round9-breakthrough`). Target ≈0.8646 dev CV (PHASE0_REHEARSAL), gap ≈0.0023.
These are development scores, not test scores.

## Why B5

Of the remaining candidates, only the Bayes item × rating-frequency group had
direct evidence. The round 9 pilot (`netflix_round9/artifacts/r9/bayes-pilot-comparison.json`)
ran rank 16 for 40 sweeps on full P2, and the group improved the posterior mean
0.8867 → 0.8857 (−0.00099), with all five folds better. Bayes was the largest
round 8 axis.

Candidates considered and not run now:
- **Retrieval context on A2h.** A3 was retrieval at a 768-token window, and it led
  A2e by 0.002 holdout at epoch 6 before crashing (exit 247). Retrieval only
  changes the context for users whose history overflows the window, and few
  users overflow A2h's 1536 window. A3 was also trending to ≈0.873, worse than
  A2h's 0.8674.
- **The rest of the latent features** (RBM hidden units, kNN neighbour features).
  Cheap, and the next CPU job after B5.

## B5 chain

`scripts/round10_b5_chain.py` (use `--check` for a dry run, which verifies all frozen
R9-LAT-A2h inputs):

1. **B5-full-r128.** B4's exact command (rank 128, `--item-bin 2 --drift 2`,
   seed 13, 160 sweeps, snapshots every 16) plus `--item-freq 1`, so the only
   difference from B4 is the new group. The binary is the round 9 sampler
   `artifacts/r10/bin/netflix-bayes-91fde00` (NFPBAYS2 checkpoints), built from
   `round9-breakthrough` 91fde00. Sweep 160 is predeclared as the column, as for
   B3/B4, so no P2 run is needed.
2. **R10-B5-s3.** The R9-LAT-A2h LightGBM config, unchanged, plus the B5 column.
3. **R10-B5-NN2.** The same NN2 settings and the fixed 50/50 average.
4. Paired fold comparison against R9-LAT-A2h-NN2. Promotion requires all five folds better.

Pilot (2 sweeps, 5 threads): **154 s/sweep, 4.8 GB peak**, swap flat. B4 took
≈180 s/sweep. Expected ≈7 h of sampling, then ≈3 h of blending.
Launched 2026-10-10 17:42 PDT under `caffeinate -i nice`. Status:
`experiments/B5-chain.status.json`; log: `artifacts/r10/B5-chain.out`; sampler
log: `experiments/B5-full-r128.log`.

Gate: standalone, compare `mean_probe_rmse` at sweep 160 with B4 (0.88966).
For the blend, promote only if all five folds improve.
