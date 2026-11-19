# Original RBM50: both native phases complete and checked

Our original rbm50lat finished both20-epoch native phases. A separate checker
verified the frozen pinned-source recipe, native executable, complete input
and output hashes, both full epoch traces, learned factor shapes/finiteness,
query shapes/finiteness and all100 paired support vectors. Every support value
on both query sets was replayed bitwise from the original trainx user factors,
matching original public predfeat semantics. Fulltrain factors are separately
preserved. Qualifying training targets remain zero; no recovered outcomes enter
fitting or the completion check.

Evidence: experiments/upstream-latent90-local/rbm50lat-receipt.json and
rbm50lat-completion-verification.json. The earlier first-phase receipt remains
an accurate historical record; its frozen downstream manifests are unchanged.
The whole two-phase model is now complete, but this alone is not a qualifying
score or a new best. The ongoing neural study uses107 own predictors plus
these50 learned components; the fixed tree follow-up is queued separately.

Original Tx20lat remains queued behind measured memory admission. Its40
components are not yet included; full90-support completion is not claimed.
