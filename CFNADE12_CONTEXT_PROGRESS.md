# CFNADE-12 context-blend development progress

Fold 0 completed on 226,746 known ProbeX rows. The three selected neural refit
budgets were 14, 20 and 18 epochs. Every saved neural model replayed bit-for-bit;
independent compensated summation reproduced the fold scores.

| Fixed tree/neural average | Fold-0 RMSE |
| --- | ---: |
| Preceding best context blend | 0.8592903561212312 |
| Context NN with completed CFNADE-12 added | 0.8593164235053121 |
| Difference (new minus preceding) | +0.0000260673840810 |

This fold is slightly worse than the preceding context result. It improves over
the older plain-neural parent, but that is not the current best comparison.
The predeclared five-fold experiment continues unchanged; four folds remain.
There is no new full-development best or qualifying-success claim.

Evidence: `experiments/upstream-cf12-context/nn/0.json` and
`experiments/upstream-cf12-context/partial-fold-0-verification.json`.
`scripts/verify_upstream_cf12_context_partial.py` reuses the unchanged partial
replay/scoring algorithm and additionally validates row identities and the
preceding context blend's frozen prediction hash before comparing scores.

The new neural component uses all 86 predictors; its unchanged tree component
uses the verified preceding 85-predictor fit. Qualifying outcomes are not read.
This is adaptive development evidence, not an official or untouched Test score.
