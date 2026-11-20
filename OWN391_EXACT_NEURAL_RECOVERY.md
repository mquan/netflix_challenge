# Exact continuation of the original paired391 H128 fit

The first H128 worker completed two of five folds and was stopped by its
original resource/watchdog guard during fold2. Its actual process71312,
model tensors, optimizer and RNG remain in RAM. The CF95/TX50n intake35985
also stopped under its guard and stays stopped during neural recovery.
Neither original failure receipt nor frozen numerical plan is overwritten.

An isolated8MiB private fixture confirms that Darwin read-only file mapping
with `msync(MS_SYNC|MS_INVALIDATE)` releases its cached pages, with full
hash and metadata parity. A separate explicit cache observation can release
only previously bound immutable feature/design file pages. It reads no
model/checkpoint/tensor state and makes no global file-cache policy change.
This does not itself prove reduced compressor occupancy or safe admission.

The new supervisor adopts only the exact stopped orphan H128 worker. It
requires three stable compressor-below4.5GiB observations and the original
16GiB reservation, crediting RAM already held by the paused worker. It then
acquires the original GPU lane nonblockingly and rechecks capacity before
continuation. Continuous limits remain own16GiB, total research24GiB for
both physical footprint and RSS, compressor5GiB, swap growth128MiB from the
last recorded original preguard sample, and disk18.75GiB. No guard is raised.

After that same worker finishes, the original independent verifier must
recompute every eligible scaler, normalized input and saved-model prediction;
the separate canonical verifier must also pass. Only then may a new
completion receipt be added at the old queue's expected path so its existing
dependent queues can proceed. Original failure and plan evidence remain.
No model retraining, numerical recipe change, qualifying inference, outcome
access, score promotion or automatic retry is authorized by this recovery.

Worker status and completion evidence live in
`experiments/upstream-own391-neural-exact-recovery/`.
