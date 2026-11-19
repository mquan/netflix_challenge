# Fixed all91 context development experiment

The completed matching86 context study scored0.8600300601883686 on known
ProbeX. The ongoing all91 study uses the plain residual NN. This experiment
applies the same rank16 context interaction architecture to the full fixed91
bank, while reusing its five independently replayed tree folds.

All six additions remain fixed: CFNADE12, attention32, attention32t, AEX16,
AEX64 and AEX128. The architecture receives228 inputs:227 prepared features
and the fold-fitted ridge baseline. Signals are the original prediction
features, all six added predictions and ridge; context is the original user/item
and voting features. The two groups partition all228 inputs exactly. Optimizer,
seeds, nested user folds, training-only scaling, rank16 and stopping rules stay
the same as the preceding context study. No feature subset or rank search.

`scripts/upstream_extension_context.py` waits for the current extension runner
and matching86 production queue to exit, then requires two actual physical
memory admission samples before acquiring the original extension fitting lock.
It verifies the parent manifest, independent score, all five tree replay
records, every saved tree artifact and copied prepared arrays before fitting.
It uses separate NN fitting/replay subprocesses and independent compensated
scoring of the four fixed arms. Existing source files and model caches remain
frozen. Insufficient memory leaves the new controller waiting.

The reference is the stronger of the verified matching86 fixed mean and the
completed all91 plain fixed mean. Only a lower overall known-ProbeX RMSE is a
candidate improvement. No minimum gain or all-fold veto is imposed. Recovered
qualifying labels and scores do not select features, architecture or weights.
There is no automatic production/audit in this experiment: an improved result
requires saved-model replay and a separate matching production plan before
another frozen recovered-label audit. The actual goal remains below0.85725 on
both recovered groups, which the first85 audit did not achieve.
