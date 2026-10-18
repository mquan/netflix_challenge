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

## B5 finished: small standalone gain, not promoted (2026-10-11)

The chain completed at 01:42 PDT. Standalone, B5's posterior mean at sweep 160 is
**0.88910 probe vs B4's 0.88966** (−0.00056). It was 0.0005 ahead of B4 at
every sweep compared (64, 80, 94).

| Blend | R9-LAT-A2h | + B5 | Change | Fold changes |
|---|---:|---:|---:|---|
| LightGBM-s3 | 0.8671122 | 0.8670916 | −0.0000206 | −.000006 +.000084 +.000019 −.000125 −.000074 |
| NN2 alone | — | 0.8670695 | −0.000059 | +.000014 +.000034 −.000110 −.000092 −.000142 |
| 50/50 average | 0.8668706 | **0.8668212** | **−0.0000495** | +.000007 +.000082 −.000061 −.000133 −.000142 |

Folds 0 and 1 are worse, so it fails the promotion rule. Each Bayes variant has
paid less than the last: B2 −0.00089, B3 −0.00036, B3 at sweep 160 and B4
−0.00012 each, B5 −0.00005. A likely reason, not tested, is that the SGD joint
models and the RBM already carry the frequency signal. The Bayes settings axis
now looks saturated, like the A2 settings axis.
**Promoted best remains R9-LAT-A2h, 0.8668706.**

## KNNF: neighbour features in both blenders (launched 2026-10-11)

These are the kNN features from GAP_AUDIT Bet 1, which R9-LAT skipped. `build/netflix knn` gains
`--extract M --extract-prefix P`. For each probe row it takes the M most similar
movies the user rated (neighbors256 graph order) and writes four features per
slot: similarity, training rating, residual under the base model, and
exp(−|Δdays|/500). It also writes log1p of the number of rated neighbours.
Empty slots are 0, and probe ratings are never read.

The run used K1 (`artifacts/K1-joint256.bin`) with the bank's kNN settings
(256 neighbours, power 1, shrink .5, time decay .003) and M = 3, giving 13 features in
`artifacts/r10/knnfeat/`. The prediction reproduces the existing K1 kNN column
exactly (0.8922894675), so the change does not alter predictions. 98% of probe rows have
at least one rated neighbour.

`scripts/round10_features_run.py --run-id R10-KNNF --features artifacts/r10/knnfeat/K1.*.f32`
adds them to the frozen R9-LAT-A2h config, then runs NN2 and the 50/50 average.
Promotion requires all five folds better than R9-LAT-A2h-NN2. R10-B5's blend
stages took 80 min. Status: `experiments/R10-KNNF-chain.status.json`.

### KNNF finished: no gain, not promoted (2026-10-11)

| Blend | Change vs R9-LAT-A2h | Fold changes |
|---|---:|---|
| LightGBM-s3 | +0.0000103 | +.000093 +.000064 +.000032 +.000020 −.000158 |
| NN2 alone | −0.0000307 | −.000101 +.000073 −.000001 −.000027 −.000096 |
| 50/50 average | −0.0000115 (0.8668592) | −.000016 +.000091 +.000053 −.000011 −.000175 |

This is noise level: LightGBM is worse on four folds. Per-neighbour detail adds nothing beyond
the K1 kNN prediction and support columns already in the bank. Together with B5,
this suggests the bank is saturated on neighbourhood and Bayes-factor signal.
**Promoted best remains R9-LAT-A2h, 0.8668706.**
