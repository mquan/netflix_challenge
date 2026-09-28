# Round 6: the algorithm for a 10% improvement over Cinematch

## Target

The goal is a 10% RMSE improvement over 0.9525, Cinematch's score on the
Netflix Prize **test** set: **0.85725** test RMSE, the Grand Prize line. The
winning submission scored 0.8567 there.

This repository measures probe RMSE with probe ratings physically removed from
training. Prize teams retrained every model on train+probe before predicting
quiz/test, which adds 1.4 million of the most recent ratings and moved blends
by roughly 0.007–0.009 RMSE in the published reports. The practical proxy for
the goal is therefore a probe development CV near **0.864–0.866**. Nothing in
this repository can confirm a test-set number; the final stage below is the
protocol that would produce one.

Reference points on the probe development metric:

| System | Probe dev CV RMSE |
|---|---:|
| R5-meta residual tree, 55 columns (start of round 6) | 0.8748094 |
| Best single model before round 6 (kNN on J6) | 0.8927409 |
| Proxy for 0.85725 test | ≈0.865 |

## The algorithm

Three stages. Stages A and B are implemented here; stage C is the standard
final-submission procedure and is specified but cannot be scored locally.

### Stage A: a diverse bank of converged base predictors

Every base predictor writes one float32 column in probe order. The bank mixes
four model families whose errors are known to be complementary:

1. **Jointly trained timeSVD++** (`netflix joint`): global mean, user bias with
   linear drift, per-user-day bias, item bias with 30 time bins, frequency-aware
   item bias, user and per-day rating-scale effects, user/item factors, implicit
   feedback over every known (user, movie) pair including probe and qualifying
   pairs, and user-factor drift. Round 6 adds Hogwild multithreading
   (`--threads N`) and a separate implicit-factor regularization so that the
   long, strongly regularized schedules that this family needs fit in wall
   time. Recipe for one member: 256–512 factors, learning rate .008, factor
   regularization .02, decay .97 per epoch, up to 80 epochs, early stopping
   with patience 8. Members differ by factor count, regularization, decay,
   seed, and optional per-user-day factors (`--day-factors`).
2. **Integrated neighborhood + factor model** (`netflix integrated`): the joint
   model plus learned explicit/implicit item-item weights over a top-k residual
   graph, all fitted by one SGD error (Koren 2008).
3. **Non-factor models**: conditional RBM with 100–200 hidden units, user and
   item denoising AutoRecs, and learned neighborhood weights over a frozen
   temporal baseline.
4. **Residual item kNN on every strong base** (`netflix knn`): similarity
   weighted residuals over the 64- and 256-neighbor graphs, power 1, shrink .5,
   optional time decay exp(−.003·days). Each kNN column also emits a support
   column (neighbor mass) that the blender uses as a feature.

Selection rule for a new column: it stays only if adding it to the frozen bank
improves the stage-B development CV in the nested five-user-fold protocol.

### Stage B: nested residual-tree blender

`scripts/round4_boost.py`. For each of five user folds: a contextual ridge
(weights vary with history size, popularity, date and same-day activity) is fit
on the training users; LightGBM then learns its residual from the bank plus
meta-features (support counts, rating mean/std, dates, same-day activity,
movie age, user one-star and five-star fractions, unrated-pair volume, active
days, movie rating std). The tree's round count is chosen on a user-disjoint
inner split, then refit on all training users of the outer fold. Several
LightGBM seeds are averaged. The held-out fold is scored only once.

### Stage C: final test-set protocol

1. Fix every architecture, hyperparameter and epoch count from stages A–B.
2. Retrain each base model on train+probe ratings with `--fixed-epoch` at the
   selected epoch, keeping the qualifying pairs as implicit feedback.
3. Predict the qualifying set with each retrained base and kNN column.
4. Apply the stage-B blender fitted on the probe-order columns (the blender was
   trained on out-of-training probe predictions, so it transfers to the
   retrained bank), clip to [1, 5], and submit.

## Round-6 experiments

| Run | Strategy | Probe RMSE |
|---|---|---:|
| K1 | joint, 256 factors, reg .02, decay .97, 6 threads; best epoch 41, early stop at 49 | **0.8941174874** (previous best joint model J6: 0.8951664) |
| kNN on K1 | neighbors256 graph, power 1, shrink .5, time decay .003 | **0.8922894675** (previous best column: 0.8927409) |
| kNN on K1 | neighbors64 graph, power 1, shrink .5 | 0.8924935588 |
| K2 | K1 recipe at 512 factors, 8 threads; stopped after epoch 15 | 0.9024858 at epoch 15 vs K1's 0.9023167: the curves coincide, so factor count is not the constraint at this regularization |
| R6-K1 | R5-meta columns + K1 + both kNN columns (+ support), same blender settings and seed | 0.8748217102 vs R5-meta 0.8748094089: **no ensemble gain** |
| R4C | conditional RBM: R3R RBM-200 (best epoch 75, 0.9083470) warm-started with rating-specific user, user-day and movie-frequency visible biases; bias rates user .003, day .01, frequency .012/100 | **0.9020408672** at the 130-epoch cap, still improving |
| R4D | R4C with 3× bias rates (user .01, day .03, frequency .012) | **0.8998687596** at the 130-epoch cap, still improving (−0.0085 vs R3R) |
| R4DR | R4D continued from epoch 131 | running |
| R5-meta-s3 (other session) | R5-meta averaged over LightGBM seeds 2026, 7, 13 | 0.8747153673, every fold better than 0.8748094 |
| **R6-RBM** | R5-meta columns + R4D + R4C, single seed 2026 | **0.8738203523** dev CV vs 0.8748094089 (−0.0009891); every fold improves |
| R6-RBM-s3 | R6-RBM averaged over seeds 2026, 7, 13 (comparable to R5-meta-s3) | running |
| P1 | conditional-bias RBM, 500 hidden units, from scratch | running |

R6-RBM fold RMSEs: .8747686 .8736287 .8702200 .8753575 .8751383 against
R5-meta's .8759153 .8744987 .8711588 .8763427 .8761425 (−0.0011, −0.0009,
−0.0009, −0.0010, −0.0010). The contextual ridge over the same columns moved
from 0.8779652 to 0.8766080, so the gain is in the columns, not in tree noise.
Two conditional RBM columns are worth about as much to the ensemble as the whole
of round 5, and R4D was still improving when it hit its epoch cap. Output:
`artifacts/r6/R6-RBM.refit.f32`; report with input hashes in
`experiments/round6/R6-RBM.json`.

## Status against the target

| Milestone | Probe dev CV |
|---|---:|
| Start of round 6 (R5-meta) | 0.8748094 |
| Round 6 so far (R6-RBM) | 0.8738204 |
| Proxy for the 0.85725 test target | ≈0.865 |

The remaining gap is about 0.009 on the development metric. The evidence in
this round says where it has to come from: not from more factor models, but
from (1) the RBM family reaching BellKor's 0.893–0.895 range (larger hidden
layers, longer schedules, day-specific hidden units were reported unhelpful),
(2) new history-conditioned families (the masked categorical history model
proposed in `RESEARCH_NEXT.md`), (3) kNN and blender variance reduction, and
finally (4) the stage-C retraining on train+probe, which is not measurable here
but is part of any test-set claim.

R6-K1 fold RMSEs were .8761155 .8745638 .8710674 .8761543 .8762190 against
R5-meta's .8759153 .8744987 .8711588 .8763427 .8761425: two folds better, three
worse, all within 0.0002. The contextual ridge over the same columns improved
from 0.8779652 to 0.8779091, so the new columns carry a little information, but
the tree's inner round selection swung from 1606 to 706 rounds on fold 0, and
that variance dominates. Two conclusions drive the rest of round 6:

1. **The factor/kNN axis is saturated.** A better timeSVD++ (−0.0010 standalone,
   −0.0005 on its best kNN column) is worth nothing to a bank that already holds
   thirteen members of that family. New ensemble gains must come from model
   families whose errors differ: the RBM (currently 0.9083, versus 0.8928–0.8951
   quiz for BellKor's conditional variants) is the largest single deficit.
2. **The blender's round selection is a variance source** of about ±0.0002 per
   fold. Seed averaging (R5-meta-s3, in progress from the other session) or
   selecting rounds on several inner splits should be treated as part of the
   algorithm, not tuning.

`src/rbm.hpp` now supports `--conditional-biases`: rating-specific user, user-day
and movie-frequency biases on the visible softmax (BellKor 2009, section VI),
sharing the joint model's session and frequency-bin definitions. User and day
biases are user-private and updated in place by the owning thread without
momentum; frequency biases are reduced across threads like item biases. A plain
checkpoint warm-starts with zero biases, so its first evaluation is unchanged;
`tests/test_joint.py` checks probe-label isolation, reload and the warm start.

Results and the blend comparison against R5-meta are appended below as they
complete.
