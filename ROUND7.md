# Round 7 — masked history against the R6 bank, then temporal context

Reference: R6-RBM5, single seed 2026, **0.8731935** development CV RMSE
(61 prediction columns). All blends below use the same recipe
(`round4_boost.py`, learning rate 0.02, 8,000-tree cap, extra meta, seed 2026)
with `artifacts/masked/reference` as the hash-checked union-bank root, since
`/private/tmp/netflix-moonshot` no longer exists.

## Masked history adds to the RBM-heavy bank

| Blend | Columns added to R6-RBM5 | Dev CV | Δ vs R6-RBM5 | Fold Δ |
|---|---|---|---|---|
| **R7-MH3** | MH3 (categorical, MSE, 4 epochs) | **0.8725848** | **−0.0006086** | −.000567 −.000794 −.000638 −.000617 −.000426 |
| R7-MHall | MH1 + MH2 + MH3 | 0.8725850 | −0.0006085 | −.000554 −.000676 −.000620 −.000677 −.000515 |

The gain against R5-meta was −0.00067; against the bank with six conditional
RBM columns it is still −0.00061 on every fold, so the masked-target network
is nearly orthogonal to the RBMs. MH1/MH2 add nothing beyond MH3.
R7-MH3 is the new single-seed best.

## Temporal extensions (in progress)

The pilot model never saw a date, while the probe is each user's latest
ratings. `scripts/masked_history.py` now has opt-in flags (defaults reproduce
the pilot bit-for-bit; `tests/test_masked_history.py` adds an isolation test
for the new inputs):

- `--context session`: each target's user code is shifted by an MLP over the
  user's other movies on the target date (implicit, own movie removed), the
  visible same-day ratings, and date features (same-day counts, days since the
  first/before the last interaction, absolute date).
- `--time-bins N`: item × date-bin output biases.
- `--suffix-prob p`: for a fraction p of users, the hidden targets are the
  latest 10–40% of ratings by date instead of a random half.
- `--save-epochs`: the full refit writes every epoch's probe column; the column
  used is the one at the epoch chosen by the separate training-only selection
  run, so the two run concurrently without the probe influencing the choice.

Each arm: selection run on the train-matched holdout (12-epoch budget,
patience 3) and a concurrent full refit saving every epoch; the blended
column is the refit at the holdout-selected epoch.

| Arm | Change | Holdout by epoch | Selected | Standalone probe |
|---|---|---|---|---|
| MT0 | MH3, 12-epoch budget | .9127 .9039 .8988 .8958 .8935 .8919 .8905 .8897 .8890 .8879 .8882 .8876 | 12 | 0.911022 |
| MT1 | MT0 + session context + 32 time bins | .8982 .8885 .8834 .8808 .8788 .8768 .8764 .8763 .8760 .8750 .8755 .8754 | 10 | **0.898931** |
| MT2 | MT1 + suffix masking (p = 0.5) | .9008 .8916 .8861 .8846 .8813 .8792 .8782 .8780 .8778 .8779 .8779 .8774 | 12 | 0.901488 |

Session context is worth 0.012 standalone (0.9110 → 0.8989), which puts a
single masked-history network level with the best conditional RBM.

| Blend | Bank | Dev CV | Δ | Fold Δ |
|---|---|---|---|---|
| R7-MT0 | R7-MH3 + MT0 | 0.8721300 | −0.0004549 vs R7-MH3 | −.000527 −.000348 −.000383 −.000496 −.000520 |
| R7-MT1 | R7-MT0 + MT1 | **0.8711298** | −0.0010002 vs R7-MT0 | −.000833 −.001000 −.001100 −.000923 −.001145 |
| R7-MT2 | R7-MT0 + MT2 | 0.8712032 | −0.0009268 vs R7-MT0 | −.000800 −.000893 −.000999 −.000996 −.000947 |

| **R7-MT12** | R7-MT0 + MT1 + MT2 | **0.8708250** | −0.0003048 vs R7-MT1 | −.000245 −.000343 −.000180 −.000422 −.000334 |

Suffix masking scores worse on the holdout (which is not a temporal suffix)
but nearly matches MT1 in the blend, and the two are complementary.
Round 7 so far: 0.8731935 → **0.8708250** (−0.0023685, single seed), every
fold better at each step. Next arms, judged against R7-MT12: MT3 (MT1 at
rank 128 / hidden 512) and MT4 (MT1 + 7-day period channel, `--week-days 7`).
