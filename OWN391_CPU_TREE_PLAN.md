# Full-budget CPU residual tree on the matched391 bank

Add a distinct learner to the admitted matched-B391 raw/native context bank.
Retain all original467 design and617 nonlinear feature operators. Use the
existing fixed residual-tree recipe: five original user-held outer folds,
three seeds2026/7/13 plus fold, ridge lambda1000 with unpenalized bias,
inner even-group fitting/odd-group validation excluding the outer fold,
31leaves, min1000rows/leaf, learning rate0.03, feature fraction0.9,
bagging fraction0.8/frequency5, max2000rounds/patience75, two threads,
deterministic column-wise LightGBM. Select each budget on inner validation;
fresh full eligible-fold refit, or zero correction for a zero budget.
Average three corrections in float64, add the original eligible ridge
baseline, clip[1.05,4.95] and save float32 held predictions.

Predeclare the fixed50/50 comparison with the independently verified own352
neural OOF partner used in the current known benchmark. Retain its disclosed
historical A/B realization caveat. The new tree raw provider itself is
matched B on both queries. No nonlinear OOF expert prediction is a tree
feature. Neither eligible ridge, inner selection nor tree fitting may use
the held outer labels. Canonical labels and original user groups are fixed.

Avoid the known Torch/LightGBM OpenMP collision: a separate source-builder
process writes complete feature and ten eligible-baseline streams over a
bounded binary pipe; the LightGBM process never imports Torch or the builder.
Keep one618-column float32 feature matrix in RAM, replacing only its last
baseline column for each selection/refit phase. No large feature file or
memory-mapped disk cache. Rebuild the complete streams independently through
the separate391 operator reader for actual saved-model replay and full
compensated canonical tree/fixed-half scoring.

CPU only, no GPU or primary-lane acquisition. Reserve8GiB headroom before
launch; cap the complete dedicated worker process group at8GiB and all
research physical/RSS at24GiB, compressor at5GiB, swap growth at128MiB and
free disk at18.75GiB. Admit after stable compressor-below3GiB observations
to leave room for the current native run. Preserve this new group's RAM on
guard or owner loss and never signal another job. All current numerical
sources, training budgets and owners remain intact. No automatic retries.

Source and pipe hashes, exact original group statistics/ridge coefficients,
every selected budget and actual saved tree, held prediction bits and both
complete canonical scores must verify before a development promotion.
This study neither fits qualifying outcomes nor generates qualifying
predictions or recovered-label audits.
