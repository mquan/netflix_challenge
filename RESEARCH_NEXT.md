Research recommendation after round 5 — 2026-09-27 (Los Angeles)

**My preferred larger experiment is a masked categorical history predictor,
with session context added as a separate ablation.** Start with the user's
history; add the movie's other raters only if the first stage contributes to
the ensemble. A context-conditioned RBM is a smaller, particularly well
supported alternative. Both address gaps confirmed in the implementation.
This is a research recommendation; no new model was trained for this note.

The reference is the completed R5-meta result, **0.8748094089 development CV
RMSE**, using 55 prediction columns. The three-seed blend, item AutoRec, and
resumed RBM were still running when inspected. Their eventual results should
be frozen before making new comparisons. The latest observed resumed RBM
checkpoint scored about 0.90864; that is a live intermediate result.

**What the previous experiments establish**

| Direction | Evidence here | Consequence for the next bet |
| --- | --- | --- |
| More temporal factors and tuning | Small standalone improvements; already included in the stronger bank | Low priority for another broad sweep |
| Residual item neighborhoods and temporal GPs | Meaningful gains, followed by substantial expansion | A new kernel bandwidth is less differentiated now |
| Local residual experts | Local versus global ensemble difference about 0.0000052 | No basis for scaling the tested local-expert recipe |
| User random-feature kernels | Combined improvement about 0.0000574 on the earlier ridge bank | Weak evidence for expanding that approximation |
| Neural and distributional blending | Mean neural heads tested; initial distributional head selected zero epochs | These tests did not train representations from raw rating histories |
| User AutoRec | Scalar rating inputs, random input removal, reconstruction of all observed training entries | A masked-target categorical objective remains untested |
| Item AutoRec | Same reconstruction distinction; current run is incomplete | Its eventual result does not test joint user/movie conditioning |
| Training-only residual tree | Improved J3, but its fixed blend failed to improve R5 | Do not expand this pilot merely because standalone J3 improved |

Sources for these local findings: [ROUND2.md](ROUND2.md),
[ROUND3.md](ROUND3.md), [ROUND4.md](ROUND4.md), [ROUND5.md](ROUND5.md),
and [HOLDOUT.md](HOLDOUT.md).

**Why the masked-history direction deserves a real experiment**

The previous [MOONSHOTS.md](MOONSHOTS.md) already proposed CF-NADE/CF-UIcA.
They remain proposals: the source and experiment manifests contain no
implementation or completed run. This recommendation is a more specific
test of that unexecuted idea, not a claim that it was never discussed.

In [scripts/autorec.py](scripts/autorec.py), `batch_input()` removes a random
fraction of input ratings, but returns all training targets. The loss then
reconstructs all those entries, including ratings still visible in the input.
[scripts/autorec_item.py](scripts/autorec_item.py) does the same. This is a
legitimate denoising objective, but part of its training signal rewards
reconstructing information already supplied. It does not exclusively optimize
the missing-rating problem faced at inference.

There is a second distinction: a centered scalar rating multiplies one input
vector. A model with separate learned embeddings for `(movie, rating)` can
represent liking and disliking a movie differently. Our RBM already has this
categorical representation, but uses contrastive-divergence training and lacks
the output context described below. The proposed network combines categorical
history with direct training on withheld targets.

[CF-NADE, ICML 2016](https://proceedings.mlr.press/v48/zheng16.pdf)
provides the main architectural precedent: autoregressive rating prediction,
parameter sharing across rating levels, an ordinal objective, and a factored
parameterization for scale. It reports stronger Netflix results than its
AutoRec and RBM baselines. Its experiment uses a random 90/10 split, so its
absolute RMSE is not a forecast for our original probe.

[CF-UIcA](https://arxiv.org/pdf/1612.07146) extends history conditioning along
both axes: other movies rated by this user and other users rating this movie.
Its Netflix evaluation also uses random splits. It motivates the second stage;
it does not establish that a two-axis model will improve our current ensemble.

My inference from our results is that this is a worthwhile representation
change: RBM and AutoRec contribute to a bank dominated by factor/neighborhood
predictions, while rating-scale and session metadata produced the last measured
gain. Feature importance alone does not prove which mechanism will help.

**Concrete first experiment**

Use one shared sparse user-history encoder with 64-dimensional categorical
embeddings, a 256-unit hidden layer, and a factored decoder that evaluates only
the requested targets. These dimensions are proposed pilot settings. Share an
encoding among multiple targets from the same masked history, avoiding a
separate complete-history pass per rating.

At each training visit, split a user's training ratings into context and
targets. Target rating values are absent from every input, and only targets
contribute to the loss. Preserve the project's separate unlabeled interaction
channel. Predict a five-star probability distribution and emit its expectation,
`sum(k * p[k])`, in the existing probe row order. Do not use the most likely
star as the point prediction for RMSE.

Run the following staged comparisons, advancing only after a useful result:

1. **Masking control:** scalar histories and an MSE head, with loss restricted
   to removed targets. Compare to the current reconstruction objective using
   matched architecture, mask generation, and target-update accounting.
2. **Categorical treatment:** replace scalar histories with rating-specific
   embeddings and use a five-class head. On this same categorical architecture,
   compare MSE on the expected rating with a categorical/ordinal objective.
   This distinguishes representation changes from the choice of loss.
3. **Session treatment:** add strongly regularized user rating-scale effects,
   activity frequency, and a separate summary of available ratings near the
   query date. Compare to the otherwise identical model without this context.
   Use a population fallback where a day has no known ratings.
4. **Two-axis extension:** add sampled other-rater context for the target
   movie, with the target value excluded from both axes. Compare against the
   user-only version at a recorded compute budget. Sampling makes this an
   adaptation inspired by CF-UIcA, not an exact reproduction.

A more exploratory session variant would encode the *identities* of other
movies in the query day's known interaction set, including unrated pairs
allowed by our transductive protocol. This can describe the kind of rating
session even when its ratings are unavailable. Existing SVD++ and RBM implicit
vectors aggregate the entire history; activity counts summarize only its size.
Test this separately against counts and global implicit history, exclude the
query movie from this session summary, and never supply unknown rating values.
This is a project-specific hypothesis, with no measured gain claimed.

**Smaller alternative: complete the RBM's conditional output**

Our [src/rbm.hpp](src/rbm.hpp) stores item/rating visible biases and computes
`rbm_visible(model, item, hidden, probabilities)`. It receives no user, day,
or frequency. Its conditional connections affect hidden units only.

[BellKor's 2009 report, section VI](https://www.asc.ohio-state.edu/statistics/dmsl/GrandPrize2009_BPC_BellKor.pdf)
adds rating-specific user, user-day, and movie-frequency biases to the visible
softmax. The report lists a 200-hidden-unit user/day variant at 0.8951 and a
frequency-aware variant at 0.8928 **quiz** RMSE. These are different trained
variants, not an isolated estimate of a single feature's effect. They also use
a different evaluation/training setup from ours. The report says frequency
modulation of the hidden-visible weights added little beyond frequency biases,
and day-specific hidden units were not encouraging.

For our experiment, retain 200 hidden units and introduce the missing output
bias families in stages. Use a copied checkpoint, comparing both the treatment
and unchanged architecture after the same continuation budget. A frozen-hidden
calibration can serve as a quick diagnostic, but would not establish the value
of joint retraining. Sparse user/day storage and shrinkage are necessary because
many sessions have very few known ratings. This is a distinct untried extension
of the existing RBM, with stronger historical support than another width sweep.

**How to decide whether either approach earns more compute**

Freeze the completed R5 predictor bank and its hashes. Record a throughput and
memory pilot before committing to full training. Use training-only validation
for epoch selection, then refit on all allowed training ratings at the selected
budget. The matched holdout in HOLDOUT.md is reusable, but its much easier J3
score shows that it is an imperfect proxy for the original probe. A later
temporal holdout can test sensitivity; do not reinterpret its absolute score
as a probe prediction.

Score every promoted column by adding it to the same raw predictor bank and
rerunning the existing nested five-user-fold residual-tree recipe. Compare
control and treatment with identical blender settings. Do not train another
blender directly on the current cross-fitted ensemble output. Report standalone
RMSE, incremental ensemble RMSE, all five fold differences, user-cluster paired
uncertainty, sparse-user slices, training time, and memory.

As an explicit compute decision, a repeatable **0.0005 ensemble RMSE gain** is
enough to retain an affordable component; **0.001 or more** would justify
expanding the architecture. These are proposed thresholds, not expected gains
or significance tests. A short, unconverged pilot can reject bad throughput or
numerical instability, but cannot establish that an architecture has no value.
Repeated probe use still makes every such result development evidence.

I would prioritize the masked-history experiment for architectural upside and
the conditional RBM for a narrower implementation effort. Bayesian posterior
averaging remains untested and is supported by
[large-scale Bayesian matrix factorization research](https://arxiv.org/abs/1503.01596),
but it adds sampling/convergence work to an already crowded factor-model bank.
It is a reserve bet. The present evidence does not justify another broad sweep
of blender capacity, local experts, or the existing user-kernel approximation.
