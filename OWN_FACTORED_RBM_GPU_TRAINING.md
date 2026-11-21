# Factored RBM training-state prototype

This extends the [separate study plan](OWN_FACTORED_RBM_GPU_PLAN.md),
[prediction equations](OWN_FACTORED_RBM_GPU_PREDICTION.md) and
[array-only user reader](OWN_FACTORED_RBM_GPU_ROWS.md). The new module implements
initialization, full epoch iteration, prediction and in-memory checkpoint state.
It is not an admitted production executable. No real dataset was opened by
these tests, no MPS operation was run and no production RBM was fitted.

`scripts/upstream_factored_rbm_training.py` defaults to the declared400 hidden
units/rank50/15 epochs/batch500/seed128 arm. Initialization uses fresh CPU Torch
random draws in D→Q→P order with the source scales, and masked training-visible
counts initialize item-category biases with the native global fallback and
epsilon. The seed and shuffle are an explicitly separate Torch/NumPy numerical
realization; Rust RNG/f32 bit identity is not claimed. Frequency biases are
analytically zero under LR_BIF0 and are identified as such in saved state.

Each epoch visits every nonempty training user once. CD steps follow the source
five-epoch schedule; the final partial batch uses its actual number of users.
Shared model and momentum tensors live on the selected device. Full user/day
biases and momentum remain explicitly on CPU, and selected global slots are
gathered and scattered without dropping any training day. Query prediction
visits every original query row, restoring source order and checking coverage.

Complete-epoch snapshots contain every actual shared/local model tensor and
momentum, the RNG state, recipe, counters and a caller-supplied input-admission
digest. Restore checks tensor shapes, dtypes, finite values, complete batch
counts, epoch counters, dimensions and the expected input digest. Synthetic
serialization tests show exact CPU continuation of the next epoch, including
parameters, momentum, RNG and query predictions. The digest is a caller assertion,
not full source verification; a future worker must bind it to actual independently
checked source/data/map records. A snapshot alone does not prove a training
history or establish a producer's provenance.

An incomplete epoch cannot be re-entered, predicted or checkpointed through
this API. A future production resource guard must preserve RAM inside the same
call stack, under its original owner/watchdog/baseline, instead of raising a
resource exception and attempting to call the epoch again. The synthetic fatal
guard test deliberately demonstrates that such a partial restart is refused.
The callable guard in this library is not itself an admission/supervision system.

Seven new synthetic CPU checks cover initialization, full nonempty-user
visitation, cold bias slots, CD/shuffle configuration, complete-state
serialization/continuation, independent scalar query prediction, invalid
snapshots and refused partial retries. Reduced dimensions and short epochs
exist only inside the synthetic tests. Alongside the earlier19,26 CPU checks
pass. They do not shorten the declared first production15+15 study.

Still required before production: full raw-file/source/split/map admission and
independent row/disjointness replay; actual MPS mathematical and throughput
checks; measured complete model/metadata/temporary/checkpoint memory and disk;
atomic/fsynced original checkpoint files and receipts; original resource guards,
primary/GPU ownership and watchdogs; complete15+15 execution and separate
saved-model full-query replay. No production queue or launch command is added
here. Original native1200 and all earlier queues remain unchanged. No new
qualifying score or progress across the below.85725/both-groups threshold is
claimed by these implementation checks.
