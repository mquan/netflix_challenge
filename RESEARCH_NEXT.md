# Next score strategy: add missing behavior, then rebuild the strong ensemble

Research reviewed 2026-09-29. The score target is RMSE on the original Netflix
probe, with all probe ratings excluded from predictor training. This is a plan,
not a new score claim.

## Decision

**First complete and preserve the R5 predictor bank; in parallel, prototype a
time and frequency conditioned RBM against the reproducible C1 bank.** R5's
historical 0.8748094 is still the strongest recorded development result, but
its 61-input bank is missing. C1, at 0.8792984, is the strongest reproducible
control. A C1 gain cannot by itself establish an R5 gain. The bank audit and
rebuild constraints are in [STRATEGY_RESET.md](STRATEGY_RESET.md).

The new model choice is based on *incremental ensemble information*. The prior
RBM scored 0.91265 alone yet improved a 12-column contextual blend from
0.88418 to 0.88113. That is the largest measured family addition in the local
record ([NEURAL_BLEND.md](NEURAL_BLEND.md)). The current `src/rbm.hpp` conditions
hidden units on known movies, but its visible rating distribution uses only a
movie bias and hidden state. BellKor's 2009 model also conditioned visible
ratings on user, rating day, and daily rating frequency, and reported a
0.8928 RMSE for one frequency-aware RBM variant. Its split and training
set differ from ours, so that figure is motivation rather than a forecast
([BellKor report, section VI](https://www.asc.ohio-state.edu/statistics/dmsl/GrandPrize2009_BPC_BellKor.pdf)).

## Ranked experiments

| Priority | Experiment | Why it could add information | Cost / decision |
| --- | --- | --- | --- |
| 1 | Restore R5 with a dependency-aware rebuild and hash every input | Establishes the actual strongest control, and restores residual kNN, RBM, AutoRec, GP, and other diverse families absent from C1 | Expensive but necessary to validate any claimed new best; do not approximate R5 with a partial bank |
| 2 | Add user, user-day, and item-frequency **visible logits** to the existing conditional RBM | Tests BellKor's specific missing mechanism, including the distribution of 1–5 ratings on bulk-rating days | Medium to high. Begin with a small fixed-epoch arm and matched unchanged-RBM control; promote only on C1 ensemble gain, then retest on restored R5 |
| 3 | Finish evaluating the deep autoencoder already in `scripts/deep_recommender.py` | A distinct dense user-history architecture with high bottleneck dropout and output re-feeding has reported gains on a **different time split** ([Kuchaiev and Ginsburg](https://arxiv.org/abs/1708.01715)) | Current local DR1 run is incomplete (`status: running` at epoch 13). C2-DR1's 0.878908 blend used its hashed epoch-13 snapshot, so it is a pilot, not a final result. Complete the run before comparing |
| 4 | Fit a small user-specific ridge correction on frozen item factors, optionally date weighted | BigChaos describes adaptive user-factor regression; it may capture local taste that global user factors miss ([BigChaos report, section 5.14](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/TocherGrandPrize2009_BPC_BigChaos.pdf)) | Cheap inference pilot. Existing local GP and user-kernel columns overlap this idea, so require a matched incremental C1 gain before scaling |

The RBM pilot should hold its hidden width, training schedule, input events,
and seed equal to an unchanged-RBM control. Add visible-logit terms in stages:
user only; user plus day; then item by daily-frequency bin. Use shrinkage on
sparse user-day and item-frequency cells. Compute frequency from all known
interaction dates, as the existing temporal models do, but use only training
ratings in the visible likelihood. Compare both standalone RMSE and the same
five-fold C1 tree blend. Save the output of **every predeclared epoch**; choose
the epoch on a training-only validation split or on ensemble gain within a
proper nested split, rather than repeatedly choosing by the full probe.

## Validation and stop rule

1. Freeze the 14 C1 input hashes and five user folds. Store candidate hashes,
   source snapshot, dataset fingerprint, seed, epoch, and all fold scores.
2. Check the candidate on the existing training-only matched holdout in
   [HOLDOUT.md](HOLDOUT.md). Its distribution is easier than the probe, so use
   it for direction and checkpoint selection, not an absolute score forecast.
3. On the original probe, fit the C1 and C1-plus-candidate blenders with
   identical settings. Report the paired overall and five fold deltas, user
   history and day-frequency slices, and the candidate's conditional residual
   correlation with C1. Once R5 is rebuilt, repeat against its frozen bank.
4. Continue a costly model family only if it contributes roughly 0.001 RMSE
   or more to the matched C1 ensemble, with consistent fold direction. A
   smaller result may be worth retaining if the model is cheap, but it does
   not justify a large parameter sweep. This is a compute gate, not a claim
   of statistical significance.

The original probe has already driven many architecture and epoch choices;
five user folds protect the blender fit but cannot erase that selection. The
winner's approach explicitly optimized *blend* improvement over standalone
model RMSE, sometimes retaining later epochs whose standalone RMSE worsened
([BigChaos report, section 3.2](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/TocherGrandPrize2009_BPC_BigChaos.pdf)). We should adopt that
objective on training-only validation and spend the original probe sparingly.

## Lower priority now

- Another short run of factor-frequency or 64-neighbor temporal models: the
  matched C1 work already found less than 0.001 combined gain from TN0/PQ2.
- A symmetric movie-side SVD++: BigChaos reported 0.92 standalone probe RMSE;
  it is a possible diversity component, but less compelling than the RBM gap
  ([BigChaos report, section 5.12](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/TocherGrandPrize2009_BPC_BigChaos.pdf)).
- External metadata or ranking-oriented recommenders: there is no matched
  R5 content ablation yet, and the original-probe objective is rating RMSE.
  The existing [enrichment review](ENRICHMENT.md) details a controlled test if
  the higher-priority behavioral experiments stall.
