# Independent moonshot investigation

Worktree: `/private/tmp/netflix-moonshot`; branch: `moonshot-local-kernels`.
Forked from `8de6482`. No changes were made to the original checkout, models,
training processes, or experiment logs. The dataset is read from the original
checkout. Completed models and prediction columns were copied into this
worktree's ignored `artifacts/snapshot/`; their hashes are recorded in
`experiments/moonshot-inputs.sha256`.

## What the current project has established

The documented best is **0.8866646513 five-user-fold development CV RMSE**
(B4). It is a contextual ensemble of factor and learned item-neighborhood
models. This is not an untouched final-test score: base checkpoints were
selected on the entire probe. The latest branch adds RBM, residual kNN and
GBDT blending. At inspection, two joint models, RBM and AutoRec were running.
Their mutable outputs were excluded from this investigation.

The original B4 cannot be reproduced from currently available completed
artifacts: its three legacy prediction columns are absent, and the J2 column
is being rewritten by a trainer. The paired experiment here freezes six
available completed components and applies identical blending to control and
treatment. Its absolute score must not be described as a change from B4.

A remaining hypothesis is that shared representations are too restrictive.
More factors, random seeds and AutoRec sizes can improve fit, but do not
establish whether each user needs a different nonlinear relationship among movies.

## Measured results

All full-probe rows: **1,408,395**. Lower RMSE is better.

| Controlled comparison | RMSE | Change versus its control |
|---|---:|---:|
| Frozen J3 source model | 0.8978688248 | — |
| J3 + GP, noise 1 | 0.8937029033 | -0.0041659215 |
| Six frozen components, existing five-context user-fold blender | 0.8860971552 | — |
| Add matched kernel-average column | 0.8854364317 | -0.0006607235 |
| Add one GP column, noise 0.1 | 0.8850761906 | -0.0010209646 |
| Add all three GP columns | **0.8850493143** | **-0.0010478409** |

The six controls are J1, J3, J4, N1, residual kNN over J3, and residual kNN
over J4. All use copied completed artifacts. The existing C++ ensemble uses
identical five user folds, five context functions and ridge 1e-5 in every arm.
No GP arm is compared against an older blend with different input columns.

All five folds improve. Excluding the 10% of users used in the initial pilot,
the all-GP gain remains **0.0010409345 RMSE**. This is a robustness check,
not an untouched holdout: the blend uses all folds for cross-fitting and the
source checkpoints used probe selection. The paired user-cluster approximate
95% interval for the full-set change in **MSE** is [-0.00199567, -0.00171609].
It does not adjust for architecture selection or shared fitted blend weights.
Raw results: `experiments/gp-six-paired.json` and `experiments/gp-six-*.log`.

The complete GP inference run took **140.4 seconds**, with one BLAS thread and
process niceness 15. The pilot took 26 seconds; sandbox restrictions prevented
lowering its process priority, but it remained a one-thread run. No GPU work
was launched. Two targeted numeric/leakage tests passed.

**Decision:** retain the GP component. The one-column version captures almost
all of the ensemble gain, while the noise-1 variant has the strongest standalone
score. Neither meets the proposed 0.005 ensemble step-change threshold. The
result supports further nonlinear local modeling, but does not prove that
jointly learned local experts will reach that threshold. The next large bet
is the two-context ordinal model described below; its empirical value remains
unmeasured.

## Ranked research bets

These are research hypotheses, not promised score improvements. A gain of
**0.005 RMSE on the same frozen full-probe ensemble** is the proposed threshold
for calling a result a step change. A 0.001 overall gain with improvements in all folds is enough to
retain a cheap new ensemble component. These thresholds express practical
priorities; they are not statistical guarantees.

### 1. Local nonlinear preference functions — tested first

**Hypothesis:** movies that occupy nearby positions in a global embedding can
interact differently for each user. A local model can capture nonlinear taste
patterns and account for redundant evidence from highly similar movies.

[LLORMA, ICML 2013](https://proceedings.mlr.press/v28/lee13.pdf) replaces a
single global low-rank matrix with overlapping local models. Its Netflix
experiments use random 90/10 splits, explicitly unlike the competition split.
Its score is therefore not a forecast for this project.

[Lawrence and Urtasun, ICML 2009](https://www.cs.utoronto.ca/~urtasun/publications/lawrence_urtasun_icml09.pdf)
derive nonlinear matrix factorization through Gaussian processes. They do not
report Netflix results. Their conditional prediction equation motivates our
pilot; our frozen-embedding residual model is not a reproduction of their
jointly learned model.

**Implemented pilot:** reuse the stable J3 item embeddings as a geometry, then
infer a separate nonlinear residual function for each query user. Normalize
item vectors and form the RBF kernel `K(i,j)=exp(4*(cos(i,j)-1))`. Select the
32 nearest movies from that user's training history, then predict

```
prediction = frozen_joint_prediction + k(query, history) @
             solve(K(history, history) + noise*I, training_residuals)
```

The inverse covariance discounts redundant neighbors. Three fixed noise
values (0.1, 1, 10) and a matched kernel-average control are emitted in one run.
The neighbor selection searches each user's complete training history; it is
not restricted to the existing global top-64 item graph. An optional product
kernel `exp(-abs(day1-day2)/time_scale)` tests session locality.

**Larger bet if retained:** learn the nonlinear geometry from training-only
masked targets, then add 16–64 overlapping local experts with sparse routing.
Compare geometry learning and local experts separately against this fixed
kernel. Train on 99 million ratings and retain the exact probe split. Use a
training-only inner holdout for early stopping. Stop expanding if the pilot's
ensemble contribution disappears with the stronger completed models.

### 2. Rating-distribution prediction from both user and item context

**Hypothesis:** reducing the history to real-valued means loses distinctions
between polarized and consistently middling ratings. Predict five ordinal
probabilities, then use `sum(k * P(rating=k))` as the RMSE-optimal point estimate.

[CF-NADE, ICML 2016](https://proceedings.mlr.press/v48/zheng16.pdf) combines
masked autoregressive context training with ordinal rating structure. Its
Netflix test split is random 10%, with a further training validation split;
its headline numbers are not directly comparable with this probe.
[CF-UIcA](https://arxiv.org/abs/1612.07146) explicitly combines dependencies
along both user and item histories. That is the useful next architectural
change beyond the active branch's user AutoRec.

**Concrete experiment:** two sparse context encoders, one for the user's other
movie ratings and one for other users' ratings of this movie, feeding an
ordinal distribution head. Exclude the target rating from both contexts at
training and inference. Start with 128-dimensional rating-specific embeddings,
a 256-unit hidden layer, and sampled item context; evaluate a user-only ablation
at equal compute. Predict the full probe in its existing row order. Time and
rating-frequency features can condition the head. Use the existing temporal
model as an optional prior, with a separate ablation.

**Risk:** item histories are huge; unbiased sampling and target exclusion are
critical. Training would contend with the currently occupied GPU, so no neural
training was launched here. This is the highest-priority larger architecture
experiment after the cheap kernel test.

### 3. Integrate over uncertain models instead of selecting one checkpoint

**Hypothesis:** point-estimate overfitting is limiting larger factor models;
posterior predictive averaging can recover capacity without the same shrinkage.
[Bayesian PMF, ICML 2008](https://doi.org/10.1145/1390156.1390267) integrates
parameters and hyperparameters. [Distributed stochastic-gradient MCMC](https://arxiv.org/abs/1503.01596)
reports improved Netflix RMSE relative to its SGD comparator; this does not
establish an expected gain over our temporal ensemble.

**Concrete experiment:** start a separate 64-factor residual BPMF chain, retain
20–40 posterior draws after training-only convergence diagnostics, and average
predictions. Compare against the same architecture's MAP estimate and an
ordinary checkpoint average. Only then scale to 128 factors or temporal
features. Use exact conditional updates for a small correctness fixture before
considering stochastic-gradient sampling. Adding arbitrary noise to the
existing SGD loop is not a valid substitute for a posterior sampler.

**Risk:** convergence and runtime, and a static model may lose temporal signal.
Keep a frozen temporal mean function and judge the contribution to the ensemble.

### 4. Model which movies users choose to rate

**Hypothesis:** selection itself carries rating information beyond an averaged
implicit-feedback vector. [Vernade and Cappé](https://arxiv.org/abs/1509.09130)
model the association between rating propensity and rating value.

**Concrete experiment:** an auxiliary observation-propensity head plus an ordinal
rating head with shared representations. Training, probe and qualifying pairs
can supply observed-interaction indicators under this project's transductive
protocol; only training ratings supply rating values. Sample unobserved pairs
with frequency correction, and test the auxiliary objective against an otherwise
identical rating-only network.

**Risk:** unobserved does not mean disliked, and correcting selection toward a
uniform catalog can hurt a benchmark that evaluates voluntarily rated movies.
Model the conditional rating among selected items; do not blindly inverse-weight
ratings or assign missing pairs one star.

## Why not start with fashionable ranking models?

[EASE](https://arxiv.org/abs/1905.03375) supplies an interesting full-rank linear
model, but its main evidence is ranking accuracy with sparse implicit feedback.
Treating all missing Netflix ratings as zero optimizes the wrong target here.
Similarly, better NDCG from graph or diffusion recommenders is insufficient
evidence of lower explicit-rating RMSE. Their losses and evaluation splits
would need an explicit adaptation before an expensive run.

## Reproduction and integrity checks

The experimental binary is macOS-specific because it uses Accelerate to compute
the item kernel. Python evaluation needs NumPy. It does not modify the normal
`netflix` CLI. For the original dataset, kernel storage is about 1.26 GB; total
process memory also includes the dataset, checkpoint and temporal indices.

```sh
make moonshot-gp
make test-moonshot-gp
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATA=/Users/quan/workspace/netflix_challenge/data/processed/netflix.bin
nice -n 15 build/moonshot_gp --data "$DATA" \
  --model artifacts/snapshot/joint128-reg005.bin \
  --output artifacts/gp-full --neighbors 32
bash scripts/moonshot_compare.sh "$DATA" artifacts/gp-full gp-six
```

Variants `.0`, `.1`, `.2` correspond to noise 0.1, 1 and 10; `.3` is the
matched average. All output columns are float32 in original probe order.
`--user-modulo 10` is the pilot mode: **unsampled rows retain base predictions**.
Use its `.mask.f32` with `scripts/moonshot_evaluate.py`; do not interpret a
pilot column as a full-probe treatment. Full runs use modulo 1.

Tests verify the posterior solve against an analytic result and verify that
changing probe labels leaves all prediction columns bit-identical (including
the temporal kernel). Existing checkpoint fingerprints bind inputs to exact
rows. GP residuals use training ratings only. No recovered quiz/test labels
are accessed. Neither our fold CV nor its uncertainty estimates remove the
pre-existing probe checkpoint-selection bias or the effect of inspecting
multiple hypotheses.
