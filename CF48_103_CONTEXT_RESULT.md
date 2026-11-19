# Own103 context NN improves the comparable development best

| Candidate | Known-ProbeX RMSE |
| --- | ---: |
| Previous best: tree91/contextNN91 fixed mean | 0.8599596977902322 |
| New tree91/contextNN103 fixed mean | 0.8595400254184109 |
| **New standalone contextNN103** | **0.8594692563654075** |

The standalone NN is the strongest completed comparable candidate and is
promoted. Its improvement over the previous best is 0.0004904414248247.
Both new candidates use the same 1,128,164 known-ProbeX rows, 462,858 users,
five nested user folds, original seeds, optimizer and stopping rules. All15
saved NN models passed actual inference replay. Independent compensated
scoring confirms NN0.8594692563654073 and fixed mean0.8595400254184110;
the last floating-point digits differ only with summation order.

The103 own predictors include the two completed temporal additions, eight ready
native outputs and both published CF48 trajectory snapshots25/95. The CF48
95-epoch native model is complete in both phases; sender verification and
coordinator transport/source/native-hash checks retain their stated scope.
No downloaded pretrained numerical predictions are used.

Full-known refitting is running. The NN-only candidate was explicitly selected
from this development comparison at2026-11-19T13:42:23UTC, before either new
qualifying outcome. Its fixed weights are NN1/tree0. The separately predeclared
equal mean also proceeds. Both require the actual full-fit saved NN replay and
a frozen query vector before audit. The reused tree keeps its original91
schema; it never receives the expanded NN103 ridge or features.

This is adaptive development evidence, not qualifying success or official Test.
The best verified own qualifying audit remains0.8614776641 /0.8624021195;
both still miss the strict0.85725 target. The new bank retains the previously
disclosed paired-parent realization differences from cached development.

Evidence: `experiments/upstream-cf48-103-context/{evaluation,independent-verification,nn-replay}.json`
and the five actual NN fold receipts. The predeclared standalone audit is bound
in `experiments/upstream-cf48-103-nn-audit/plan.json`.
