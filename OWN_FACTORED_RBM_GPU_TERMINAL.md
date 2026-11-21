# Complete terminal RBM predictor and durable output prototype

This extends the [training-state implementation](OWN_FACTORED_RBM_GPU_TRAINING.md)
and [combined input-admission routine](OWN_FACTORED_RBM_GPU_ADMISSION.md).
It is not an admitted production worker or a new prediction candidate.

`scripts/upstream_factored_rbm_terminal.py` retains every learned predictive
tensor: P,Q,D, visible/hidden biases and all user/day biases. Saving and replay
require an externally supplied frozen recipe, the exact phase/input-admission
digest, complete epoch/nonempty-batch counters and full original dimensions.
The model cannot authorize itself using only its own recorded configuration.
Production still requires the declared400-unit/rank50/full15+15 recipe.

This terminal format is separate from the complete resumable optimizer snapshot
tested earlier. Momentum is not needed for inference and is explicitly absent;
training resume/extension is unsupported. No learned prediction tensor, training
row, factor, epoch, category or query is dropped. Both phase terminal models
must be retained. Actual storage and serialization peaks still require admission.

Replay uses the actual loaded tensors and all original reader histories,
conditionals, day slots and query rows. It does not fabricate optimizer state.
CPU tests show exact source-order query-vector replay after model serialization
and `weights_only` loading, with separately tested scalar prediction equations.
Actual MPS replay/support/tolerances remain unverified.

Output writes are exclusive and staged. The directory chain and parent entries
are fsynced before payload writes. Each1MiB chunk requests additional disk
reservation from the external guard; the production guard must self-suspend and
resume the same process/write stack until safe. A callable that ignores this
contract does not establish resource admission. Complete bytes are fsynced,
published through a no-clobber hard link, directory-synced, staging-unlinked,
resynced and fully hashed. Existing outputs and failed staging are preserved.

Six new synthetic CPU tests cover complete model/query replay, external recipe
and phase/input rejection, missing tensors, durable model/vector round trips,
failed staging/no retry, exact buffer-byte writes, per-chunk reservations and
the directory flush order before the first payload byte. These join the earlier44
CPU checks. No actual production model was fitted or serialized by these tests.

Remaining work includes the source-bound owner/watchdog/worker integration,
continuous original resource guards, real63-file semantic admission, measured
memory/disk, actual MPS preflight, full15+15 fitting and separate saved-model
full-query replay. The original native1200 and its prior queues remain unchanged.
No new qualifying RMSE or progress across the target is claimed here.
