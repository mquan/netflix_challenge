# All90 latent neural control on the admitted Apple GPU backend

Replace only the verified idle107/all90 CPU controller, with no child or NN
model cache, by a separately declared float32 MPS control. Use the already
checked333-feature cache,334 inputs, hidden128 and rank32. The same-backend
actual small fit/save/reload test passes bitwise. The representative337-input
benchmark measured3.36x faster small synthetic fitting; no full-data accuracy
or speed is claimed before the real study.

The dataset, labels, outer/inner user folds, architecture, optimizer, dropout,
weight decay and stopping are fixed. GPU dropout uses its own seeded stream;
this is a new backend recipe, not CPU weight parity. Both fitting and actual
saved-model inference replay use MPS, with explicit float32 and a per-process
allocation cap. Conservative memory admission and the shared fit lock remain.
All original native and current tree fits continue unchanged.

Run this ready control while the110-predictor/all90 batch awaits complete
Tx20lat and rank1000 native proofs and fixed-batch preparation. The small GPU
cost permits a useful control rather than a long CPU fit blocking that batch.
If the larger batch queues during actual control fitting, it waits for the
control's full five folds/replay. No active fit is terminated. Both studies
require complete saved-model replay and independent known-row scores before
promotion, followed by predeclared full fitting/freeze before qualifying access.
Best qualifying remains0.8609266298 /0.8619141823; the10% goal is unmet.
