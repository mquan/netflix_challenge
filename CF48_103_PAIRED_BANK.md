# Paired inputs for the own103 context iteration

`scripts/upstream_cf48_103_bank.py` retains the frozen paired91 native
realization and appends the same fixed twelve predictors used in development:
two completed temporal models, eight previously unused completed native outputs,
and our CF48 trajectory's published epoch25 and completed epoch95 predictions.

Both query sets now have 179 ridge columns and 239 nonlinear features. The
last ridge column remains the intercept. New raw predictions are clipped to
[0,6]; the nonlinear features use the same predictions minus float32 3.6.
The original context and voting columns remain byte-for-byte unchanged.

`scripts/verify_upstream_cf48_103_bank.py` checks every retained column, every
appended transform, input/output hashes, the completed CF48 intake receipt,
and exact known row identities. New known columns must also match the originating
development cache. Label file dtypes differ (paired int8, development float64),
so this last identity check compares exact rating values rather than demanding
identical NPY bytes. Each file retains its own hash binding.

The paired91 parent has the previously documented nine raw-column differences
from cached development. Those differences are preserved and disclosed; this
is not a claim to reproduce the development cache exactly.

This bank is preparation, not a score promotion. The active five-fold context
experiment must finish and its saved models must replay before deciding on a
full refit. A future combination must keep the reused tree91 on its original
167-column ridge and 227-feature bank, while the new NN103 uses this expanded
bank. Fit the NN's ridge and weights only on known ProbeX. Freeze the resulting
fixed float32 mean before a recovered qualifying audit. The prior best qualifying
audit remains authoritative until a better actual result is verified.
