# New base-model families: shortlist

Written 2026-10-18 after HR2a/HR2b/HR4 showed the blender already extracts what
A2h and B4 know. Control: **R9-LAT-A2h, 0.8668706 dev CV**; gap ≈0.0022.

## What history says pays

- First arrival of a **new learner family** paid 0.0006–0.0009 (Bayes MCMC B2
  −0.00089, A2 transformer −0.0006, conditional RBM −0.0031 on an early bank).
  Later variants of the same family paid 0.0001–0.0004, then nothing.
- Saturated or dead (do not retry): SGD factor variants, item kNN on SGD
  residuals, RBM seeds/sizes, MT seeds, A2 settings, item-side twins (A1, IA1),
  user kNN, metadata/tags, distribution-head features, Bayes session/day
  variants (ROUND11), residual trees and latent codes of A2h/B4 (this branch),
  local path structure (LC2/LC3 on the other branch).
- 63% of probe rows (67% of squared error) have no same-day training rating, and
  error grows with the gap since the user's last training day (ROUND11).

## Candidates (families with no model in the bank)

| # | Family | Why it might be new information | Evidence against | Cost |
|---|---|---|---|---|
| N1 | **Selection-aware network**: A2-style encoder with a second head trained to predict *which* movies a user rates (train + probe + qual pairs as positives, frequency-corrected negatives) next to the rating head (MOONSHOTS §4, Vernade & Cappé) | Every current model uses the rated set only as input. None is trained to explain the choice, which is the one signal qual pairs carry about the test period. Probe/qual choices are post-last-training-day, which is exactly where the error is | Implicit input already captures much of it. "Unobserved ≠ disliked" | Code: 1 day on top of `masked_sequence.py`. P2 pilot ≈4–6 h MPS |
| N2 | **Dual-context ordinal model** (CF-UIcA): user-history encoder × encoder of the target movie's raters, interacting before an ordinal head (MOONSHOTS §2) | A2 sees the movie only as an embedding. Raters' context is per-query, not per-movie | A1 item twin dead alone. LC3: neighbour context adds ≈0 once pooled | 2 days of code; item histories need sampling. Higher MPS memory risk |
| N3 | **PT BK4-style integrated model** with low-rank time-dependent biases (GAP_AUDIT, est. 0.0001–0.0005) | Best published single predictor (0.8713 quiz) vs our I2R 0.8946 | SGD factor axis saturated in the blend | 1–2 weeks of C++; high risk |
| N4 | Gaussian-visible / flipped conditional RBM | Another family member | Audit estimate ≈0.0001 | ≈1 day CPU |

## Recommendation: N1 first

It is the only candidate that adds a new *training target* rather than a new
architecture over the same targets. It reuses the A2 trainer and honest P2
protocol, and the cost to a verdict is about a day.

Protocol:
1. **N1-pilot** on T−P2 at A2-select's settings but shorter (same seed, 4–6 h
   MPS). The arm differs from a matched rating-only control only by the selection
   head (loss weight λ ∈ {0.1, 0.3}, chosen on P2 user fold 1).
2. **Cheap honest gate:** add the P2 column to the 11-honest-column ridge on P2
   (HR1 protocol, folds 2–4 fit, 0 evaluate). Continue only if it beats the
   matched control's column by ≥0.0005 there (pilot gains shrink ≈10× in blend).
3. If it passes: full-T refit with fixed epochs → probe column →
   `hr_blend_chain.py --columns` (gate ≥0.0003, all five folds).

Stop rule: if the selection head does not beat its matched control on P2,
drop N1 and try N2 only if the other branch's encoder finds non-pooled signal.

One MPS job at a time; check swap and other sessions before launching.

## N1 as launched (2026-10-18 23:10 PDT)

The first version adds the selection signal as **input**, not as an auxiliary head:
`scripts/choice_sequence.py` (copy of the A2h trainer at bet-a2-sequence bf3c719)
with `--choice-tokens 1`. Implicit-only pairs (probe, qualifying and, in the
holdout dataset, P2) become sequence tokens at their dates with a sixth star
value CHOICE. At prediction time the queried pairs are dropped from the choice
tokens. On the holdout dataset there are 5,633,092 choice events, and all
1,407,566 P2 queries are among them. The run uses A2h-select's exact settings
and seed, so its per-epoch P2 curve (best 0.86740 at epoch 12) is the matched
control. Run: `experiments/N1-select.*`. GPU memory is similar to A2h
(≈45–60 GB), so no blend job runs alongside it.
An auxiliary selection head (λ·BCE) is the follow-up if tokens alone help.

## HR5: snapshot columns (screen)

`scripts/hr_column_screen.py` fits a linear correction of R9-LAT-A2h's
out-of-fold residual on each of 204 unused snapshot columns (epoch snapshots
of A/MT models, Bayes sweeps), using the blend's own user folds. Best single
columns: MT8 e12 −0.0000248, A2b e14 −0.0000165 (5/5 folds), A2i e6 −0.0000138.
Over-trained epochs beyond the selected one score highest, which agrees with
BellKor/BigChaos's advice to over-train for the blend. 19 columns have
delta < −3e-6 with ≥4 folds better. A batch blend test (HR5-blend) is queued
after N1 frees memory. Expected ≈−0.0001 at most.

### N1 P2 curve so far (vs A2h-select, same settings and seed)

| Epoch | N1 | A2h | Δ |
|---:|---:|---:|---:|
| 1 | 0.90567 | 0.90575 | −0.00008 |
| 2 | 0.88790 | 0.88868 | −0.00078 |
| 3 | 0.87960 | 0.87732 | +0.00228 |
| 4 | 0.87455 | 0.87514 | −0.00059 |
| 5 | 0.87187 | 0.87344 | −0.00157 |

`scripts/n1_chain.py` (running) waits for selection, then the honest gate
(≤ −0.0010 beyond A2h), then the full-T refit and the blend test.
