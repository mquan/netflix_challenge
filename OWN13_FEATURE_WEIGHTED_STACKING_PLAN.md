# Context-dependent stacking of our complete13 families

Declared before reading either new own352-fold-bag or complete13 recovered
audit. This is a substantive next method, separate from their four frozen
constant-weight candidates. Their results do not select these features,
regularization, folds or component list.

Use every one of the13 eligible original OOF/full-query pairs declared in
ALL_COMPLETE13_CALIBRATION_PLAN.md. Preserve the exact duplicate-base alias
and every source expectation; do not select component subsets. Fit only on
the complete1,128,164 known labels. All input predictions must trace to our
freshly trained models, including the unchanged15-model own352 fold bag.

Reproduce the interaction form of the released native FWLS implementation:
one float64 column per model prediction times context feature, plus an
unpenalized intercept. Use the original fwls15 context list from
`artifacts/upstream-repro/fresh/voting-3.toml`, flattened with first-occurrence
deduplication as in `src/blend.rs::resolve_voting`. The list contains15 names
but repeats vf090; there are14 distinct gates, hence183 total coefficients
for13 models. Read only our original `preds_new/vf` counterparts, bound to
the completed fresh vfeat1 receipt and original train/qual row identities.
Keep original finite float32 feature values; no new feature screening,
scaling, clipping or parameter optimization.

Use lambda30 on all182 interaction coefficients, the released fwls15k4
regularization. The intercept is unpenalized. Preserve our original five
held-user groups rather than importing the released random-row partition.
Accumulate full Gram/RHS sufficient statistics separately for all ten
original groups. Each of five fold models fits the other eight groups;
score only its two held groups. These are cross-fitted meta-model scores
conditional on existing OOF features, not fully nested end-to-end CV.
They must never be reported as an untouched official Test result.

The sole declared production candidate is the uniform five-fold bag.
Predict each query row with each fold model, clip that prediction in
float64 to[1.05,4.95], sum the five clipped values in fold order, divide
once by five, and store float32. Known predictions use only the held-user
fold model. Keep all folds; no score-dependent fold or model selection.

Before recovered audit, independently rebuild every interaction and all
ten groups' statistics, re-solve all five actual saved models, reconstruct
every held and qualifying prediction and uniform-bag value, verify all
source hashes, and freeze all vectors. Audit only the complete canonical
recovered groups, with exact counts and compensated float64 clipped RMSE.
Strict success remains RMSE below0.85725 on both groups. Retain negative
results. No recovered outcome enters fitting, feature selection or stopping.

This intermediate182-interaction model tests the public stacking method
on our completed nonlinear families. The larger352-raw-predictor FWLS
study remains a separate subsequent expansion; this does not claim that
the full public model inventory or its numerical performance is reproduced.
