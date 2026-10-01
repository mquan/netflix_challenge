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

| Arm | Change | Holdout RMSE by epoch (train-matched selection set) |
|---|---|---|
| MT0 | MH3, 12-epoch budget | .912748 .903926 .898826 .895772 .893511 … |
| MT1 | MT0 + session context + 32 time bins | .898199 … |
| MT2 | MT1 + suffix masking (p = 0.5) | .900775 … |

At one epoch session context is 0.0145 better than the control, already ahead
of the control's third epoch.
