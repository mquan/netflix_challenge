# Score progression

All development scores use the original Netflix **probe split**: its 1,408,395
labels are removed before training, predictions are clipped to `[1, 5]`, and
RMSE is computed over the complete probe set. Quiz/test numbers from the 2009
leaderboard are never mixed into this table.

**Interpretation correction:** C5/C6 fit blend weights and report their error on
the same probe rows. They are development fit scores. From B1 onward, blending
also reports five-fold error with folds assigned by user. This prevents a row's
label from fitting its own blend weights, but base-model epoch selection still
uses the probe. Neither number is an untouched final-test score, and percentage
improvements against Cinematch's different quiz/test sets are not justified.

The legacy `time-svdpp` command fits drift after freezing all static parameters;
it is a staged approximation, not the published jointly trained timeSVD++ model.
The new `joint` command trains the temporal baseline, latent factors, implicit
feedback and user-factor drift together. Prior claims that static factorization
was at its limit were premature: the optimization scheme also needed testing.

## Joint-training experiments (2026-09-27)

Commands, source/binary hashes, timings and complete epoch logs are kept in
[`experiments/`](experiments/). Best models and raw probe prediction columns are
saved locally under `artifacts/`. No recovered quiz/test labels are used.

| Run | Strategy and outcome | Probe RMSE |
|---|---|---:|
| J1 | Joint 64-factor model with exact deferred implicit SGD and shuffled ratings. Regularization 0.0015; best epoch 6. Stopped manually after epoch 12 because six consecutive epochs worsened. | 0.9000114063 |
| B1 | Ridge blend with intercept of the three legacy components and J1; weights cross-fitted in five user folds. Gains demonstrate complementary errors. | **0.8936900925 CV** (0.8936836795 fit) |
| J2 | Joint 128 factors, stronger regularization 0.02, 40 epochs; best epoch 40. Regularization needs more training than the old residual model. | 0.8961722036 |
| J3 | Joint 128 factors, regularization 0.005; best epoch 9, automatic stop after epoch 15. Improves J1 but eventually overfits. | 0.8978688266 |
| B2 | Add J3 to B1 and allow blending weights to depend on history counts, movie popularity, date and daily activity. This combines two changes; it is not a pure context ablation. | 0.8914751148 CV (0.8914446469 fit) |
| J4 | Warm-start J3 with 128 per-user-day factors. Best at first new epoch; subsequent epochs overfit, stopped after epoch 6. About 9 GB extra parameters; modest standalone gain. | 0.8970498562 |
| B3 | Add J2 and retained J4 to B2; same five user-fold contextual blending. | **0.8901342372 CV** (0.8900970503 fit) |
| J5 | Joint 256 factors, regularization 0.01: best epoch 17, early stop at 22. Strongest standalone factor model so far, but relatively small blend gain in B6. | 0.8951664067 |
| G1 | Build a top-64 item-neighbor graph from training residuals, using capped histories for discovery. Graph construction has no rating-prediction RMSE. | Not applicable |
| N1 | Learn explicit/implicit neighbor weights over G1 and frozen temporal baseline. Best epoch 4, stopped after epoch 8. Weaker alone but useful in B4. | 0.9100421231 |
| B4 | Controlled addition of N1 to B3's seven-column contextual blend. All five user folds improve; local movie relationships add complementary signal. | **0.8866646513 CV** (0.8866238775 fit) |
| B5 | Nonlinear calibration: same B4 inputs, plus pairwise products of centered model predictions. Small improvement in all five user folds, much smaller than adding the neighborhood family. | **0.8865814901 CV** (0.8865091808 fit) |
| B6 | Add completed J5 to B5 with identical blending settings. Another small but consistent improvement in all five folds. | **0.8863207420 CV** (0.8862375747 fit) |
| G2 | Same training-only sampled graph discovery as G1, retaining 128 neighbors rather than 64. | Not applicable |
| N2 | Capacity ablation: learn 128-neighbor model with N1 optimizer unchanged. Best epoch 5; later epochs worsened. | 0.9072719035 |
| N2i | Evaluate a frozen copy of N2's epoch-5 checkpoint while training continues; preserve its exact prediction column for blending. | 0.9072719035 |
| B7 | Add frozen N2i to B6's quadratic contextual blend. Each of five held-out user folds improves over B6. | **0.8858460517 CV** (0.8857544990 fit) |
| N3 | Temporal-locality ablation: 64 neighbors, with explicit/implicit contributions multiplied by `exp(-0.002 * absolute_day_gap)`. Best epoch 8; worse standalone than N1. | 0.9108568308 |
| J6 | Six-epoch warm-start pilot from J2, adding movie-by-rating-frequency latent factors; rate 2e-5, regularization 0.02. | 0.8958286794 |
| J7 | Matched J6 continuation control without frequency factors; same initial model, seed, rates and iteration budget. J6 improves on this by 0.0001167435 RMSE. | 0.8959454229 |

J1's log has an intentional SIGTERM exit, not a numerical failure; its best
checkpoint and predictions were saved before termination. Subsequent joint runs
support automatic early stopping. The legacy residual-model regularization
results do not transfer automatically to joint optimization.

The next structural experiment is learned item-item neighborhood regression:
directly model relationships among movies, rather than compressing every
relationship into the same latent space. The graph is approximate and temporal
biases are frozen, so this is not a faithful reproduction of the full published
integrated neighborhood/factor model. A weaker standalone predictor can still
help the blend, but no step-change in score is promised.

**Observed neighborhood outcome:** N1 is worse alone than the factor models,
yet B4 improves over B3 by `0.0034695859` RMSE. All five folds improve. B5's
quadratic calibration adds only `0.0000831612`; model diversity delivered the
larger gain. B4 code is retained in commit `cffd6bf`, tagged
`milestone-probe-cv-0.886665`. From N2 onward, each experiment also archives
actual code/build/test files in `.source.zip`, not just source hashes.

As of the 2026-09-27 progress checkpoint, B7 is the best recorded user-fold
blend score. It uses the frozen N2i epoch-5 column, adding `0.0004746903` RMSE
improvement over B6. N2i's model and prediction hashes are in its manifest;
the completed N2 run may have continued changing its conventional artifact.
The next comparison is a blend using N3's temporal neighborhood and a direct
test of J6's frequency-factor column against the matched J7 control. Both must
use frozen prediction columns and the same five user folds.

## Checkpoints

| Checkpoint | Strategy | Best probe RMSE | Change | Conclusion |
|---|---|---:|---:|---|
| C0 | Regularized global, user, and movie biases | 0.98256528 | — | Validates parsing, holdout removal, persistence, and the scorer. Main effects alone are insufficient. |
| C1 | BellKor-style temporal/frequency baseline: item time bins, user drift, per-user-day effects, and rating-scale drift | 0.93491601 | -0.04764927 | Temporal behavior is a large signal, but has no user–movie interaction model. Cinematch used a different evaluation split. |
| C2 | 32-factor SVD++ residual model over C1, with qualifying/probe pairs used only as unlabeled implicit feedback | 0.90267216 | -0.03224385 | Collaborative factors produce another large gain. Validation bottoms out at epoch 7 while training error keeps falling, so regularization—not more epochs—is the next lever. |
| E1 (rejected) | C2 with factor regularization raised `0.0015→0.01` and bias regularization `0.02→0.05` | 0.91221075 | +0.00953859 | Excess shrinkage prevents the factor model from fitting useful interactions. Reject this branch and restore the original regularization. |
| C3 | Increase SVD++ capacity from 32 to 64 factors with C2 regularization restored | 0.90079729 | -0.00187487 | Additional factors help, but this configuration again bottoms at epoch 7. Add temporal drift next; this does not establish a factorization ceiling. |
| C4 | Add timeSVD++ user-factor drift `αᵤ·devᵤ(t)` to the retained C3 model, holding static factors fixed | 0.90055257 | -0.00024472 | Preference drift is measurable but small at this capacity. Retain it as a distinct ensemble component; pursue diversity through blending next. |
| C5 | Blend C2's 32-factor model with C4's 64-factor time-aware model; fit one global weight on probe | 0.90019305 | -0.00035952 | A 27.6%/72.4% blend improves both components, confirming useful residual diversity. Add an independently initialized factor model next. |
| E2 (ensemble-only) | Independently initialize C3 with seed `42` instead of `2009` | 0.90100211 | +0.00044954 vs. C4 | Slightly worse alone, but its different errors make it useful to the ensemble. |
| C6 | Least-squares blend of C2, C4, and E2 | **0.89970985** | -0.00048320 | First sub-0.90 checkpoint. Diverse initialization contributes despite weaker standalone accuracy; continue adding structurally different predictors rather than near-duplicates. |

Lower is better. `Change` is relative to the preceding local checkpoint, not
the historical Netflix leaderboard.

## C2 epoch progression

Configuration: 32 factors, factor learning rate `0.008`, decay `0.9`, factor
regularization `0.0015`, residual-bias learning rate `0.003`, residual-bias
regularization `0.02`, deterministic seed `2009`.

| Epoch | Train RMSE | Probe RMSE |
|---:|---:|---:|
| 1 | 0.79231082 | 0.92150733 |
| 2 | 0.77295689 | 0.91491104 |
| 3 | 0.75756079 | 0.90848587 |
| 4 | 0.74514171 | 0.90506988 |
| 5 | 0.73448198 | 0.90347725 |
| 6 | 0.72573754 | 0.90268807 |
| 7 | 0.71919498 | **0.90267216** |
| 8 | 0.71437302 | 0.90303331 |
| 9 | 0.71065301 | 0.90335701 |
| 10 | 0.70766251 | 0.90366407 |
| 11 | 0.70520013 | 0.90433171 |
| 12 | 0.70312632 | 0.90476062 |

The growing train/probe gap after epoch 7 is clear evidence of overfitting. It
motivated the stronger-regularization E1 test below; E1 showed that the tested
increase was too aggressive.

## E1 rejected experiment

Probe RMSE progressed from `0.92992445` at epoch 1 to `0.91221075` at epoch 10
and was still improving slowly, but it remained materially worse than C2 at
equivalent compute. This indicates that `0.01` factor regularization is too
strong; the next experiment restores `0.0015` and increases capacity from 32 to
64 factors. From E1 onward, training automatically persists the best probe
epoch rather than the final epoch.

## C3 epoch progression

Configuration matches C2 except for 64 factors. The automatically retained
checkpoint is epoch 7.

| Epoch | Train RMSE | Probe RMSE |
|---:|---:|---:|
| 1 | 0.79230182 | 0.92220181 |
| 2 | 0.77264247 | 0.91392825 |
| 3 | 0.75664735 | 0.90873538 |
| 4 | 0.74328303 | 0.90460784 |
| 5 | 0.73032909 | 0.90233964 |
| 6 | 0.71715439 | 0.90126568 |
| 7 | 0.70452108 | **0.90079729** |
| 8 | 0.69363213 | 0.90125582 |
| 9 | 0.68505337 | 0.90190850 |
| 10 | 0.67847258 | 0.90293623 |

Doubling factor capacity gives a real but diminishing improvement. Because
both C2 and C3 turn upward at epoch 8, simply increasing factors or epochs is
unlikely to cross 0.90 reliably. The next strategy is timeSVD++ user-factor
drift, which models preferences changing with rating date.

## C4 epoch progression

Configuration: start from retained C3 epoch 7; learn only 64-dimensional
per-user temporal drift with learning rate `1e-5` and regularization `50`, as
reported for BellKor's timeSVD++ drift parameters.

| Epoch | Train RMSE | Probe RMSE |
|---:|---:|---:|
| 1 | 0.68638454 | 0.90065504 |
| 2 | 0.68422377 | 0.90059204 |
| 3 | 0.68278229 | 0.90056344 |
| 4 | 0.68173228 | 0.90055276 |
| 5 | 0.68092585 | **0.90055257** |
| 6 | 0.68028346 | 0.90055883 |

The small, consistent gain confirms temporal preference drift, but the plateau
shows it cannot supply the next large jump by itself. The next checkpoint will
blend independently trained 32- and 64-factor predictors, including C4, to
exploit differences in their residual errors.

## C5 blend

The best clipped blend assigns `0.276` to C2 and `0.724` to C4. The
unclipped least-squares solution assigns `0.70881013` to C4, close to the grid
optimum and evidence that the result is not a grid-search artifact. The blend
reaches `0.90019305`, just `0.00019305` above the next milestone. The next
experiment trains the 64-factor architecture with a different deterministic
seed and adds it as a third diverse predictor.

## E2 and C6 progression

E2 uses the C3 configuration with seed `42`:

| Epoch | Train RMSE | Probe RMSE |
|---:|---:|---:|
| 1 | 0.79224665 | 0.92086159 |
| 2 | 0.77264202 | 0.91372265 |
| 3 | 0.75675381 | 0.90801171 |
| 4 | 0.74343085 | 0.90465367 |
| 5 | 0.73036151 | 0.90235560 |
| 6 | 0.71702676 | 0.90120653 |
| 7 | 0.70422144 | **0.90100211** |

The C6 constrained-sum least-squares weights are `0.11436320` for C2,
`0.49503203` for C4, and `0.39060478` for E2. After clipping to `[1,5]`, the
blend scores **`0.89970985`**. E2 is weaker than C4 by itself yet receives a
large weight, demonstrating that residual diversity—not standalone rank—is the
important ensemble criterion. The next strategy should add a structurally
different predictor, such as a neighborhood residual model, rather than merely
another random seed.
