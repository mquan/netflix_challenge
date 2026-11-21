# Combined RBM input-admission routine, pending actual execution

This joins the [original input declaration](OWN_FACTORED_RBM_GPU_INPUTS.md)
with the [independent array verifier](OWN_FACTORED_RBM_GPU_DATA_AUDIT.md).
The routine is implemented and synthetically tested. It has not scanned or
admitted the actual Netflix inputs, acquired resources or launched a model.

`scripts/upstream_factored_rbm_admission.py` has no CLI. A future source-bound
production controller must supply its original resource/ownership guard. The
routine checks the pending declaration, carries forward original expected hashes,
fully hashes63 raw arrays/map plus bound code/providers, checks exact schemas
and complete payload ends, and independently replays both phase readers. It
checks the entire native holdout's positive-training disjointness, exact known
map/labels, zero qualifying targets and unique qualifying addresses. Input
identities and the declaration are checked again after semantic replay.

Each NPY mapping is decoded from the same authenticated open descriptor that
is mapped. It must match the fully hashed original identity before use; its
descriptor and logical path must remain unchanged through closure. Only primitive
native vector headers are accepted. Bound maps close on both success and failure.
Query fields supplied to the model reader omit ratings; the independent known
metadata and explicit qualifying-zero validators open their separate arrays.

Logical model, momentum and retained metadata byte counts are reported from
actual row/day dimensions. They do not establish the real process/GPU/temp or
serialized checkpoint peak. The external supervisor must monitor continuously,
including during `UserRows` construction, which has long scans and no internal
per-block guard callback. Passing a callable alone is not resource admission.

Seven new tests cover native schemas, trailing bytes, query field isolation,
map cleanup, descriptor/path replacement, object-payload rejection and logical
storage counts. A temporary63-array fixture exercises the complete orchestration
with only declaration and row-count boundaries mocked; all real hashing, reader,
semantic, disjointness, zero and identity checks execute. Its negative case
confirms semantic zero rejection even after a synthetic mocked hash expectation
is changed. This cannot change the actual frozen declaration or its expectations.

These join the earlier37 synthetic CPU checks. No real-data admission, MPS
support, production footprint, durable owned worker execution, full15+15 model
fit or saved-model query replay is claimed. Those gates still precede any new
candidate or qualifying score. Original native1200 and all prior queues remain
unchanged.
