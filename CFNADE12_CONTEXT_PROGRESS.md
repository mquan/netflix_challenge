# CFNADE-12 context-blend development progress

All five folds are complete. The fixed tree/neural average improves the preceding
verified development best from **0.8602161957664950 to 0.8600637147575518**
(delta **-0.0001524810089432**). All saved neural models replayed bit-for-bit;
independent compensated summation reproduced the full scores and fixed average.
The NN alone scores 0.8600986905055020.

| Fold | Rows | Neural refit budgets | Preceding context mean | New fixed mean | Difference |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 226,746 | 14, 20, 18 | 0.8592903561212312 | 0.8593164235053121 | +0.0000260673840810 |
| 1 | 225,899 | 20, 17, 20 | 0.8576817970921646 | 0.8575002796649305 | -0.0001815174272342 |
| 2 | 224,006 | 15, 25, 24 | 0.8613082334525111 | 0.8610794493775495 | -0.0002287840749615 |
| 3 | 225,873 | 19, 13, 20 | 0.8619337506392766 | 0.8617462498381866 | -0.0001875008010900 |
| 4 | 225,640 | 25, 22, 23 | 0.8608735291035140 | 0.8606817791142021 | -0.0001917499893119 |

Four folds improve; fold 0 is slightly worse. Across 1,128,164 rows and 462,858
users, paired mean squared error changes by -0.0002623100164213. Its conditional
user-cluster normal 95% interval is [-0.0003198221634376, -0.0002047978694049].
This interval has no correction for historical model/feature selection.
The median per-seed NN budgets for a subsequent full-known-data fit are
**19, 20 and 20 epochs**, frozen before any qualifying audit of this candidate.

Evidence: fold receipts in `experiments/upstream-cf12-context/nn/`, the complete
`nn-replay.json`, `evaluation.json`, `independent-verification.json` and
`comparison-previous.json` in that experiment's report directory. Earlier partial
checks remain in `partial-fold-{0,1,2,3}-verification.json`.
`scripts/verify_upstream_cf12_context_partial.py` reuses the unchanged partial
replay/scoring algorithm and additionally validates row identities and the
preceding context blend's frozen prediction hash before comparing scores.

The new neural component uses all 86 predictors; its unchanged tree component
uses the verified preceding 85-predictor fit. Qualifying outcomes are not read.
This is adaptive development evidence, not an official or untouched Test score.
The qualifying target remains unverified. Production integration must preserve
the original 85-predictor tree component and the matching 86-predictor NN schema,
refit on the actual paired known predictions, replay the saved full-data models,
freeze the final vector and independently check both recovered groups.
