# Full KNN3 neural study with exact, bounded memory operations

The model remains the original400-input H128/rank32 MPS context residual NN:
all1,128,164 known rows, five nested user folds, seeds2026/7/13 plus the original
fold offsets, batch8192, maximum40 selection epochs, patience4, original
learning rate/dropout/AdamW regularization, and selected-budget full refits.
Both fixed half means with the already replayed tree50/tree154 are retained.
There is no smaller model, shorter budget, row sample, feature subset search,
new coefficient fit or qualifying-outcome selection.

The prior full-size worker parked behind a16GiB peak reserve. Its scaler and
input assembly allocated redundant full-size float64 temporaries. The new
execution changes storage and buffer lifetimes:

- Complete original185-column float64 ridge data and399-column float32 feature
  data are recovered from verified full-size caches. All original C-order
  matrix fingerprints match exactly. The cache's400th OOF ridge column is an
  operator-test input only; the actual worker recomputes each original inner
  and outer ridge baseline from the complete185-column matrix and original
  sufficient statistics, then appends it exactly as before.
- Float32 input assembly avoids an intermediate full float64 stack. Source
  maps are released after both complete inner/full neural inputs are built.
- Eligible-row percentiles and reduction order are retained. The private
  float64 clipped buffer is reused for the original squared deviations;
  normalization is applied in8192-row blocks with the same operations.
- Saved-model replay uses the unchanged original reference normalization on
  every actual held row, plus all original actual neural model replays.

The transformation proof covers every real row and all400 columns. Original
and compact inner/full statistics match bitwise on the complete outer-fold0
eligible sets (721,484 /901,418 rows). Every normalized float32 value across
all1,128,164 rows also matches for both reference scalers. The test input uses
the actual known OOF ridge baseline; this is an exact operator proof, not a
claim that an unrun reference training trajectory was compared. The actual
neural worker recomputes the original fold-specific baselines and fits all
five complete folds. Fixtures also cover constant/extreme values, baseline
rounding and multiple transform block boundaries.

The new process has an enforced10GiB own-RSS budget beneath the unchanged24GiB
total-RSS /8GiB disk continuous guard. It requires two stable samples below
14GiB other RSS while exclusively reserving the shared GPU epoch slot, with
unchanged compressor/swap/disk checks. This changes memory reservation, not
model size or training budget. An overrun preserves partial evidence and
stops only the owned process; no smaller automatic retry is permitted.

Only the still-idle pre-fit coordinator23052 was replaced. It had no child,
no admitted neural run and no neural fold artifacts. Its declaration, prepared
inputs and source files are preserved. All original source/model/column/data/
fold/seed/optimizer/stopping settings remain identical. The raw50-epoch
trajectory waits at its normal durable epoch boundary and resumes when this
full neural study releases the shared slot. Native CF96 and RBM200 continue.

Evidence: experiments/upstream-knn3-scaling-parity/{plan,cache-verification,verification}.json;
experiments/upstream-knn3pair2-compact-queue/{recovery,plan,admitted,launch}.json;
experiments/upstream-knn3pair2-compact-mps-context/manifest.json.
Twenty-nine tests pass. At launch no global model score or qualifying gain is
claimed; all five folds, all15 actual saved-model replays and separate complete
stored-mean/canonical-score checks are still required.
