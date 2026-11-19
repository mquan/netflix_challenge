# Keep independently checked snapshot proofs immutable

The item-oriented raw producer publishes fixed epoch pairs5/10/20/30/40/50.
Independent checks run when a pair arrives, again before ensemble fitting, and
again after full raw completion. Previously each successful check would rewrite
only its observation timestamp. A later check could therefore invalidate the
SHA256 lineage recorded by a fitted ensemble despite identical model outputs.

The verifier now reruns every original model/configuration/hash/finite-vector/
known-score check, then serializes publication with a file lock. If a successful
proof already exists, every field except the invocation timestamp must match;
its original bytes and first successful timestamp are retained. Changed scores
or source hashes raise an error and preserve the evidence. Concurrent initial
writers publish one stable report. No raw fit, input, configuration, checkpoint,
capacity, budget or prediction arithmetic changed.

The isolated fixture checks timestamp-only repeats, rejection of changed scores
and source bindings, and three simultaneous writers. The actual epoch5 check
was subsequently rerun: all original checks pass and every proof byte/hash is
unchanged. Its standalone component known RMSE is1.0064540238061566; this is not
a global ensemble improvement. The original50/50 trajectory continues.

Before any paired epoch5 score existed, the pending fixed5/10 neural controller
was stopped while it had no child, input cache or fitted model. Its original
plan/source/logs are preserved under upstream-itemae-virtual-queue-before-
immutable-proof. The same numerical study was freshly declared with only the
snapshot verifier code binding changed. An independent comparison verifies all
other model, input, seed, fold, stopping, checkpoint and half-mean fields are
identical. Controller63754 now waits for epoch10; watcher63850 checks each
original paired snapshot as it arrives.

The already-running raw model plan and all fitting code bindings remain
unchanged. Its parent queue's earlier snapshot-verifier hash is explicitly
superseded by this verifier-publication amendment, recorded before the first
paired diagnostic; the raw queue itself and all numerical model code remain
unchanged. No qualifying outcome influenced this change or checkpoint choice.

Evidence: experiments/upstream-itemae-immutable-proof-fixture/{verification,
amendment-verification,real-recheck-verification}.json;
experiments/upstream-itemae-virtual-queue-before-immutable-proof/amendment.json;
experiments/upstream-itemae-snapshot-watch/{plan,checked-00005}.json.
