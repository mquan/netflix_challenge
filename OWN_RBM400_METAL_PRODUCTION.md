# Full independent deterministic-Metal RBM400 production

The original-scope v2 attempt completed full63-input admission but failed its
actual deterministic MPS preflight before fitting. Retain it and both previous
metadata attempts. The separately reviewed actual device probe now passes the
small/cold-user and FULL500user/17770item/400hidden/50factor cases under the
original tolerances. Gradients and predictions repeat bitwise. Its exact proof
SHA256 is a8ab47ca345a93213de0dd7c9cdb94d0cdeca2b87fb05d8f241a8a1572c40a24.

This is an explicit numerical implementation successor: each duplicate-index
floating addition uses the tested stable single-writer Metal reduction instead
of unsupported Torch MPS index_add_. No atomics are used. CPU sorting handles
integer addresses only; learned floating computation stays on MPS with fallback
and fast-math disabled, strict deterministic algorithms and original tolerances.
Worker, training State and terminal serialization/replay ASTs otherwise change
only their declared imports. Config, initialization/RNG, full CD schedule,
optimizer, dimensions, epochs, tensors, data and prediction equations remain.
This is our named Torch/Metal realization, not a claim of Rust RNG/bit identity.

Control: experiments/upstream-factored-rbm-execution-deterministic-metal.
Study: experiments/upstream-factored-rbm-gpu400-deterministic-metal.
Artifacts: artifacts/upstream-factored-rbm-gpu400-deterministic-metal.

Require the actual full device proof and complete source closures for every
prior attempt. Before launch, source-bound review must approve retiring ONLY
the exact idle v2 consumer owner25257/watch25319, which never admitted a
scientific child. Failed v2 producer/worker/watch must already be terminal with
empty learned artifacts. No trained model, optimizer or RNG state is discarded.
Preserve all old directories and records; never forge their success.

Copy the EXACT ORIGINAL first-scientific baseline bytes, SHA256689232383581ed51cbdffc8a1da5ae047ee94d690ea51b64bccbbab20b447168.
Keep original3GiB data/8GiB GPU workers,24GiB aggregate both physical/RSS,
5GiB compressor/4.5admission,128MiB swap growth,18.75GiB disk floor plus full
remaining outputs,2owned lanes/live watchdog and3stable samples. Resource
pressure preserves the exact worker/frame; no automatic retry or budget cut.
Use the exact ORIGINAL research-peer accounting domain with native counters.

Run EVERY original phase: fresh full63 input admission, fresh actual full MPS
preflight, known fit, saved-model known replay/projection, qualifying fit,
saved-model all-query replay and paired assembly. Both fits require ALL15epochs
and all480189 nonempty users in500user batches, including the final smaller
batch. Retain every epoch receipt and both complete seven-tensor predictors.
No per-epoch predictor snapshots or resumable optimizer are promised.

The matching16-family known-only successor preserves every original own15
family, order, coefficients protocol and label barriers. It cannot use this
probe as a trained predictor or audit partial vectors. Only complete actual
models/queries/paired replay admit it. No recovered outcome is used for fitting
or selecting weights, families or budgets. Both groups must strictly beat
0.85725 to meet the goal; actual device proof alone establishes no score gain.
