# Own107 plus50 original RBM context: new development best

| Completed five-fold candidate | Independently recomputed RMSE |
| --- | ---: |
| Previous standalone NN103 | 0.8594692563654073 |
| New standalone own107 +50 context NN | **0.8594554093063751** |
| Original tree91 / new NN fixed half mean | 0.8594773262608065 |

All15 saved neural models replayed bitwise on all five held-out user folds.
Separate compensated scoring and the exact float32 fixed mean passed.
Same1,128,164 known-ProbeX rows and462,858 users. 2 folds improve and 3
worsen against NN103; only the lower aggregate score decides promotion. No
minimum-gain or all-fold veto applies. This is adaptive known-label development.

107 own raw predictors,183 ridge columns and293 nonlinear features plus ridge
input. The50 additional original learned RBM user components are context,
without rating clipping or centering. Original rank16/hidden64 architecture,
optimizer, seeds, nested user folds and stopping are retained. All numerical
features were trained by us. Qualifying labels did not enter fitting or selection.

The original RBM has now separately completed and verified both20-epoch phases.
The frozen development manifest correctly preserves its earlier first-phase-only
state at preparation. The paired107+50 bank has separate complete column checks.
The new tree comparison is fitting on this same expanded feature cache, with
actual originating NN model/receipt/replay reuse. A full-known neural refit and
frozen-vector qualifying check are next; no new qualifying outcome exists yet.

Best qualifying remains0.8609916544 /0.8619938614, or9.5019568% on the worse
group. Both must fall below0.85725; the goal remains unmet.

Evidence: experiments/upstream-rbm50-context/evaluation.json,
independent-verification.json, nn-replay.json and all five nn/fold receipts.
