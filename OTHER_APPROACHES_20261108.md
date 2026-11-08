# Other approaches while production runs (Claude, 2026-11-08)

The user asked for other approaches while the production retrain runs. This
is reading only; nothing was launched. Constraints in force: new-family pilots
closed for good, no cloud GPUs, moonshot on hold.

## What the record already closes

Checked against GAP_AUDIT.md (winners' techniques), STRUCTURAL_PLAN.md,
NEXT_EXPLORATIONS.md ("Not worth reopening"), ROUND8–13 and GOAL.md closed bets:

- **Blend side.** Latent user/movie features are in the promoted R9-LAT.
  Closed: ENNBlend/third blender (R8), snapshot/epoch trajectory diversity
  (LC6), A2 code features (HR4), metadata, seed bagging (≈0.00005).
- **Base-model variants.** Closed: Bayes B5 frequency, session and day
  variants (R10–11); kNN neighbour features (R10); MT/RBM seeds; bigger
  transformer (A2c); E1, E3, E5, SO1, H8, H9.
- **Side information.** The headroom map found none beyond N1. A7 showed
  Stage C absorbs probe-sibling and user-shift signal.
- **Per-user blending on probe** (Pragmatic Theory, BigChaos) can't be
  measured under user folds, and A7 suggests Stage C absorbs it.

## Not yet tested

### 1. Blender training-data scaling (new; never measured)

The final blender (LightGBM-s3 ⊕ NN2) learns from probe rows only: 1.4M
ratings. Under 5 user folds each fold trains on 80% of probe users; the
production blender trains on 100%. Nobody has measured how blend RMSE
depends on the number of blender training rows.

- **Diagnostic (cheap, existing columns only).** Refit the frozen R9 recipe
  with each fold's training users subsampled to 50% and 75%. Same columns,
  folds and evaluation rows, one fixed subsample seed. Cost ≈ 2 full-bank fits
  ≈ 2–3 CPU hours. No GPU, no new models.
- **What it tells us:**
  1. If 100% → 50% costs ≥ 0.0006, the curve is steep enough that doubling
     the blender's rows could plausibly gain ≈ 0.0003+. That would justify the
     follow-up below.
  2. The 80% → 100% step also tells us how much the dev CV under-states the
     production blender. That is a calibration term, like the Stage C offset.
- **Follow-up if it pays:** add **P2 rows** (the probe-matched holdout,
  `artifacts/validation/train-matched-20260927.bin`) as extra blender
  training rows. That needs P2-held-out versions of all 79 columns. Some
  exist (A2h-select, B2/B4-p2, P2 RBMs, the rehearsal bank); the rest need a
  retrain of the bank on train−P2. That costs about as much as a production
  pass, using the same queues. Predeclare: keep it only if it gains
  ≥ 0.0003 in the full-bank CV on all five folds.
- **Prior:** uncertain. Tree/NN blenders over 79 correlated columns with
  ~120 features are usually data-hungry at the margin, but it may be flat.

### 2. Longer Bayes chains (production-side, low prior)

B3/B4 were still improving at sweep 160, and Bayes has the highest
quality→blend slope (0.14). Running them to 320 sweeps would cost ≈ 12 h
of CPU per chain, measured on dev by resuming the dev chains
(`--init-model`). Prior ≤ 0.0001.

### 3. Blend-gain epoch selection (STRUCTURAL_PLAN lane 3a; partly covered)

This means choosing each column's epoch by P2 blend gain rather than
standalone RMSE. LC6 already found *adding* trajectory snapshots doesn't pay,
which bounds this from above. Prior ≤ 0.0001–0.0002.

## Bottom line

Together, the open items plausibly add **0.0003–0.0008**, against a
**0.0041** gap. Only item 1 is new and cheap to test. Closing the gap would
need a closed decision reopened: new families, cloud GPUs for scale, or the
moonshot (≈ 10–15% to pass its gates).
