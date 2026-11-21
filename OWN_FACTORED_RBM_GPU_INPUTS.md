# Original raw-input declaration for the local RBM study

This extends the [separate study plan](OWN_FACTORED_RBM_GPU_PLAN.md) and
[training-state prototype](OWN_FACTORED_RBM_GPU_TRAINING.md). The input declaration
is metadata only. It does not admit a production dataset or start training.

`scripts/upstream_factored_rbm_inputs.py` carries forward the original expected
SHA256 values for44 freshly ingested train/probe/ProbeX/fulltrain/qual arrays,
18 original split3 arrays and the original ProbeX→probe_3 row map. It cross-checks
the independent split/map receipt and all eight original/known projection inputs.
No expected data hash is replaced with the hash of a current payload. Exact file
sets, source pins, known-label provenance flags and map expectations must agree.

The declaration also binds the original mathematical source files and earlier
CPU-prototype verification/source/test-log hashes. Its default recipe remains
the separate400-unit/rank50/15+15 arm. No downloaded prediction feature or
recovered outcome path is an input. Qualifying target arrays retain their
original zero-target expectation; their current bytes and zero values still
require the later guarded admission scan.

The only command currently provided is `declare`. It reads metadata and code,
and exclusively creates/fsyncs a pending declaration. The generic `verify_files`
helper can fully hash regular bound files in guarded1MiB reads with identity
checks and cache avoidance, but it is not exposed as a production-admission
command. It does not replace full NPY schema/zero checks, raw-row/conditional/day
replay, known mapping/label comparison or train/known-holdout disjointness.

Four new synthetic checks preserve provider expectations, reject conflicting or
incomplete/outcome-reading providers, check every byte of temporary payloads,
and reject guard failures, unsafe paths and mid-read identity changes. They
join the previous26 CPU-only tests. No production payload scan or new model
fit is established by those tests or by the pending metadata declaration.

Remaining gates include the independent full-data semantic verifier, actual MPS
checks, measured memory/disk, source-bound production worker and durable model
receipts, original ownership/resource supervision, complete15+15 execution and
saved-model full-query replay. The original native1200 and prior queues remain
unchanged; no new qualifying score is claimed.
