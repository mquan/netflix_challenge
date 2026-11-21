# Independent raw-array verifier for the local RBM study

This extends the [input declaration](OWN_FACTORED_RBM_GPU_INPUTS.md). It implements
independent semantic checks without opening production files, admitting a dataset
or launching training. The declaration and previous prototype sources remain
unchanged. Actual full63-file validation and semantic replay have not run.

`scripts/upstream_factored_rbm_data_audit.py` reconstructs raw training/query
counts, masked positive ratings and item-category counts independently of the
producer. For every user it uses separate set-based day/conditional reconstruction,
explicit source query-row ordering, and raw coordinate comparisons. It checks
every positive rating, masked/query conditional address, local/global user/day
slot, query identity, eligibility rule and complete coverage. Canonical stream
hashes remain identical across different verifier batch sizes. The item universe
must be supplied from the separate declaration rather than trusted from the reader.

The known-row audit compares each mapped raw identity and known label, rejects
duplicate source/known identities and scans all positive training rows against
the entire native holdout. This includes holdout rows outside the mapped scored
subset. Masked training addresses alone are allowed as conditional metadata;
their labels do not enter positive states. A regression specifically injects an
unmapped native holdout identity into positive training and verifies rejection.

A separate check requires the original int8 qualifying-target shape and every
target value zero. The future worker must first establish the original payload
hashes before decoding these arrays. No recovered-outcome path belongs to the
declaration, and query outcomes never enter the model reader or fit.

Seven new synthetic tests cover complete raw→reader replay, producer corruption,
global day/visible metadata errors, mapped and unmapped holdout leaks, labels/maps,
duplicate identities and zero-target schema/values. They join the previous30
CPU-only checks. Passing tiny fixtures establishes verifier behavior, not a
verification result for the actual Netflix arrays or a new model score.

Still required: a source-bound guarded production worker combining full original
file/header/zero checks with complete actual row/map/disjointness replay, measured
memory/disk, actual MPS validation, ownership/watchdogs, durable complete15+15
fitting and independent saved-model full-query replay. Original native1200 and
all prior queues remain unchanged. No production data-admission or qualifying
accuracy claim is made by this prototype.
