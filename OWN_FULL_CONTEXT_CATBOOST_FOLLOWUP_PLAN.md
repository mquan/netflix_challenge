# Full-context high-capacity tree follow-up

This is the next implementation plan if the queued own17 primary misses the
target. It does not change any live or frozen source, model, training budget,
baseline or queue. Do not start this study ahead of the existing owned chain.

The current known-only own16 simplex assigns the new standalone RBM400
weight0.0016778366438841104, while tree154 receives0.180674067 and multiple
H128 neural fits receive most of the remaining weight. The preceding complete
H128 folds selected6–9 epochs, well below their40-epoch ceiling. These are
known-only diagnostics: they motivate a different learner and richer tree
interactions rather than more epochs of the same converged neural recipe.
They are not a new development score or recovered audit result.

Train two full CatBoost residual-tree arms against the independently verified
scalar and learned-context operators already being assembled. Preserve every
original raw predictor and all225/375 contexts. Include the eligible-fold
ridge1000 baseline as the last raw float32 feature. Each arm retains the
original five outer user folds and three seeds2026/7/13 plus fold, and the
same even/odd inner split excluding the entire outer fold. No nonlinear OOF
expert prediction or held outer outcome enters model features or fitting.

Predeclare CPU CatBoost1.2.10, RMSE loss, depth8, learning_rate0.03,
l2_leaf_reg10, iterations2000, early_stopping_rounds100, border_count64,
bootstrap_typeBernoulli, subsample0.8, rsm0.9 and thread_count2. Fit to known
residuals from each eligible ridge baseline. An inner model that fails to beat
the zero-correction baseline selects zero iterations. Otherwise use its
best_iteration+1. Freshly initialize and refit each selected positive count on
all outer-training rows; retain every selected actual saved model. Keep zero
slots and divide corrections by three. Add float64 CatBoost predictions in
seed order to a float64 correction, add the full eligible ridge baseline,
clip[1.05,4.95], and save float32 held predictions.

Pin task_typeCPU, grow_policySymmetricTree, eval_metricRMSE and
allow_writing_filesFalse explicitly; save owned final models separately.
The parameter semantics and CPU support are described in the official
[common parameters](https://catboost.ai/docs/en/references/training-parameters/common)
and [bootstrap options](https://catboost.ai/docs/en/concepts/algorithm-main-stages_bootstrap-options).
These references support the configuration; they do not establish an expected
score improvement or a measured memory/runtime bound for this dataset.

Isolate the CatBoost process from Torch. A bounded original-operator producer
streams the complete raw matrix and the ten eligible ridge baseline vectors,
then exits before tree fitting begins. Retain one full raw float32 matrix,
replacing only its final baseline column. Quantization must be fitted on the
applicable inner/full training population, never held outer or qualifying
rows. Pin every source, runtime version, operator hash and model inventory.
Budget16GiB for the entire dedicated producer/consumer group,24GiB aggregate
in each physical/RSS domain, the unchanged original baseline and resource
limits, full remaining-output reservation and both original lanes. Suspend
and preserve the same process/stack on resource or accounting failure.

Independently rebuild both complete raw streams and eligible ridge systems;
reload all actual saved models, verify every held prediction bit and compute
separate float64-reclipped compensated canonical CV scores. Select the lower
complete tree-only known CV score, scalar on a tie within1e-12. Finish both
arms before selection. Report actual measured peak resource use; no CPU mock
or partial model inventory establishes production completion.

Generate all2817131 selected-arm qualifying predictions in original8192-row
blocks from every original fold/seed model and independently replay all
operator/input/model/fold/bag bits. No fitting or selection on qualifying
outcomes. Only then append this matched tree-only pair to all seventeen
original own17 families in original order. Predeclare own18 simplex primary,
ridge10 comparison with unpenalized intercept, and standalone tree control.
Independently verify all262143 simplex supports,19 ridge equations and all six
frozen vectors before three fixed recovered audits and separate canonical
checks. Known fitted blend RMSE is not nested CV; recovered audits remain
adaptive rather than official or untouched Test results.

Goal completion still requires the original independently trained primary
strictly below0.85725 on both full recovered groups. This plan itself adds no
model, score, plot point or target achievement.
