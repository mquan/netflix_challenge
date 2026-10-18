# Research reset after round 10

2026-10-12. Original proposal below; retrieval and cheap summary pilots are now
complete in [LOCAL_PATHS.md](LOCAL_PATHS.md). On 200,000 P2 queries, triplet
summaries improve a matched pooled + kNN ridge control by 0.00569 RMSE, with
a user-clustered paired MSE interval excluding zero. Retrieval is feasible at
about 1,240 queries/second. This is not an ensemble improvement; the structural
encoder comparison and frozen-bank evaluation remain unrun.

The next experiment should test a missing source of conditional information.
The recommended bet is local matrix completion around each query: preserve the
connections between the target user's history and the target movie's other
raters, then learn how those connections predict the missing rating.

## Evidence for changing direction

- Promoted control: R9-LAT-A2h, 0.8668706425 development CV, from
  `../netflix_round9/ROUND9.md` and its completed comparison report.
- B5: standalone gain 0.00056; final ensemble gain 0.0000495, two folds worse.
- KNNF: final ensemble gain 0.0000115, two folds worse.
- A2i: gain 0.0000189 on its A2h control, three folds worse. A2j was stopped
  after trailing A2h. See `ROUND8.md` and `ROUND10.md`.

These results support saturation of the tested variations, not a claim that
the dataset has reached its irreducible error. Against the approximate 0.8646
development target, the remaining squared-error reduction is about 0.00393,
or 0.52% of current MSE. We need a small amount of consistently predictable
remaining error, but have not established where it is.

The target itself remains an estimate from a nine-column retraining rehearsal.
Re-measuring it with Bayes and A2 is useful calibration; it is not a model gain.

## Hypothesis: preserve the joint rating pattern

For query (u, i), retrieve other users v who rated i and share movies j with u.
The observed three-edge paths are:

    u --rating(u,j)--> j <--rating(v,j)-- v --rating(v,i)--> i

The target rating (u,i) is absent. A learner can examine agreement and
disagreement across several shared movies for each v, together with v's rating
of i. Dates and rating-scale differences can condition how much to trust v.
For example, agreement specifically on the target movie's relevant movie group
may be more informative than one global user similarity score.

This revisits information related to neighborhoods, but changes what survives
aggregation. Existing user/item kNN compresses evidence into similarities and
weighted sums. A2 attends over the target user's history. A1's failed item-side
twin does not by itself reject a model that preserves which other user rated
which shared movie. The inspected primary and round9 code/docs contain no
completed experiment with this query-centered subgraph representation.

[Zhang and Chen, IGMC (ICLR 2020)](https://arxiv.org/abs/1904.12058)
provides a research precedent for predicting explicit ratings from local
user-item subgraphs. It supports the architectural idea, not an expected gain
on this Netflix ensemble. Its published experiments use other datasets; this
proposal would be a temporal, sampled adaptation.

## First experiment: establish information value before scaling

1. **Measure retrieval feasibility.** On a deterministic, rating-independent
   sample of P2 queries, retrieve at most 32 other raters of the target movie
   and 32 movies from the target user's visible history. Preserve their sparse
   intersection. Choose candidates from allowed context using overlap and
   dates; never use the target rating. Report coverage, sparsity, preprocessing
   time, query throughput, memory, and projected full-run cost. These caps are
   pilot settings, not optimized values. Avoid exhaustive scans of popular
   movies' raters for each query.
2. **Cheap signal test.** Summarize the paths using the 5 x 5 x 5 rating-triplet
   histogram, support, time gaps, and per-neighbor agreement summaries. Fit a
   regularized predictor on P2 with user-disjoint fit/selection partitions.
   Compare to a matched simple user-kNN predictor from exactly the same
   retrieved context. A negative summary test weakens this cheap approach;
   it does not prove that all subgraph models fail.
3. **Structural test.** If retrieval is feasible and the pilot is promising,
   compare a small joint subgraph encoder with separately pooled user/movie
   contexts at matched input and training budgets. A graph network or a
   masked attention model can implement it; the key variable is retained
   cross-user correspondence, not the architecture's name. Include a
   structure-destroying ablation that rewires the cross-links while retaining
   degrees and rating counts as closely as possible. Measure how well the
   ablation preserves these statistics; do not attribute a change solely to
   topology if those inputs also changed.

Use rating prediction for the initial learner. A residual variant needs a
source model trained without its target labels: P2-trained source predictions
on P2 for fitting, and the corresponding full-training source predictions on
original probe for application. Never use a model's in-sample errors as if they
were out-of-sample residual targets. The earlier failed J3 residual-tree pilot
in `HOLDOUT.md` is a relevant negative control, not evidence for this bet.

## Evaluation contract

- Reuse the P2 infrastructure for pilot training/selection. P2 is easier than
  original probe; its absolute RMSE cannot establish an ensemble breakthrough.
- P2 labels remain absent from every graph edge, retrieval statistic, embedding
  used as input, and auxiliary training objective that supplies P2 context.
  Split P2 labels into user-disjoint fit/selection partitions for the new
  predictor. Original probe labels never enter any model context.
- For final refit, fix architecture, budget and selection choices; train the
  new predictor on the allowed P2 targets and/or masked original-training
  targets under an explicitly recorded recipe. Exclude each masked target
  from all rating-bearing input paths. Using more masked training targets is a
  separate scale-up, not an unrecorded change to the pilot.
- Add the completed prediction column to the frozen R9-LAT-A2h raw bank;
  rerun its existing nested LightGBM/NN2 recipe and fixed 50/50 average.
  Do not fit a new correction directly to its existing out-of-fold output.
- Report standalone RMSE, ensemble delta, all five fold deltas, user-cluster
  paired uncertainty, sparse-user slices, and compute cost. Retain the existing
  all-five-fold promotion rule. Repeated development selection remains a
  limitation even if the paired interval excludes zero.
- A repeatable ensemble gain of at least 0.0003 is a proposed threshold for
  substantial further investment, not a predicted outcome. Smaller robust
  gains may be retained if cheap, but would not justify another broad sweep.

This bet earns expansion only if preserving joint context adds value beyond
the matched pooled and neighborhood controls. Another settings sweep cannot
answer that question.
