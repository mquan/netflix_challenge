# Reuse completed own auxiliary training at a safe boundary

The primary source DAG still contains native DNN/RBM/KNN jobs already completed
in the isolated auxiliary tree. Reusing those exact own artifacts prevents
unnecessary retraining. It does not add a predictor or change the frozen
qualifying recipes. Current best development remains0.8602161957664950;
qualifying success remains unverified.

`scripts/upstream_import_backfill.py` first waits for the entire six-endpoint
auxiliary continuation to finish and acquires its original exclusive lock.
It verifies source pin/tracked changes/zero qualifying sentinel, identical
primary data aliases, identical registry/setup receipts, native commands,
all output hashes, and all existing primary conflicts before mutation.

It waits for a completed primary job receipt. It then briefly suspends only
the primary scheduler PID after checking its command and suspension state.
If any live direct child or last owned wrapper-group process remains, it
resumes the scheduler and retries; it never signals an active model wrapper.
Once the scheduler is acknowledged suspended and no wrapper is active, it
links completed auxiliary outputs into the primary prediction tree and
publishes complete native receipts with explicit auxiliary execution-tree,
original-receipt hash and transfer-code lineage. Existing valid files/receipts
are retained. A failed unpublished job rolls back only its newly created
links; previously committed complete jobs remain valid.

The scheduler resumes in a finally block on success or exception. This avoids
injecting unrelated files into an active before/after output scan. Native
source/configuration/hyperparameters, code-bound development caches and
ongoing fits remain unchanged. The primary source runner notices imported
receipts normally when it reaches those jobs and skips duplicate work.

Five checks pass, including actual SIGSTOP/SIGCONT on a dedicated test process,
exception recovery, wrong-PID rejection, source corruption, conflicting owned
files, transactional link failure and idempotent lineage-preserving reuse.
No real training process was suspended by these tests. The new controller
uses its own `experiments/upstream-import-backfill` lock/status/plan/result.
It is not a training job and does not establish qualifying quality.

All six DNN follow-on endpoints have now completed. Their independent paired
verification records and every native receipt output hash have been rechecked
in `experiments/full-upstream-reproduction/dnn-backfill/completion-verification.json`.
The import controller has passed its auxiliary preflight and is waiting for a
completed primary job boundary after `attn-32`. No active native fit has been
suspended or had its prediction-directory scan changed.
