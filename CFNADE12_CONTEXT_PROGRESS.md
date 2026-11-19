# CFNADE-12 context-blend development progress

Two folds have completed. Every saved neural model in both folds replayed
bit-for-bit; independent compensated summation reproduced their scores.

| Fold | Rows | Neural refit budgets | Preceding context mean | New fixed mean | Difference |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 226,746 | 14, 20, 18 | 0.8592903561212312 | 0.8593164235053121 | +0.0000260673840810 |
| 1 | 225,899 | 20, 17, 20 | 0.8576817970921646 | 0.8575002796649305 | -0.0001815174272342 |

Fold 0 is slightly worse than the preceding context result; fold 1 improves.
Both improve over the older plain-neural parent, but that is not the current
best comparison. The predeclared five-fold experiment continues unchanged;
three folds remain.
There is no new full-development best or qualifying-success claim.

Evidence: fold receipts in `experiments/upstream-cf12-context/nn/` and
`experiments/upstream-cf12-context/partial-fold-{0,1}-verification.json`.
`scripts/verify_upstream_cf12_context_partial.py` reuses the unchanged partial
replay/scoring algorithm and additionally validates row identities and the
preceding context blend's frozen prediction hash before comparing scores.

The new neural component uses all 86 predictors; its unchanged tree component
uses the verified preceding 85-predictor fit. Qualifying outcomes are not read.
This is adaptive development evidence, not an official or untouched Test score.
