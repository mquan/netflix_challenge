# Original latent support reproduction on the coordinator

The goal remains qualifying RMSE strictly below 0.85725 on both recovered
groups, using freshly trained own predictions and locally fitted blends.
The 90 missing latent supports are a concrete gap in public-schema coverage;
their value to our blend is not yet measured.

`scripts/upstream_latent90_local.py` prepares a private pinned source tree and
runs the original `rbm50lat` and `tx20lat` recipes sequentially. Both models
retain their original 20 epochs in each of the two phases, dimensions,
default seeds, optimizer settings and verified own input arrays. The original
native fit executable is reused with its source/build receipt; unchanged
`predfeat` was compiled with the existing BLAS build configuration.

The controller waits for two stable memory samples, with existing native and
meta-fitting physical footprints below 16 GiB, compressor below 5 GiB, swap
growth below 32 MiB per 30 seconds, and free disk above 24 GiB. It never stops
an existing native job and never restarts partial native training automatically.
This is a local fallback; the remote assignment remains independently useful.
An actual verified remote return can avoid subsequent duplicate work at an
idle boundary. Do not interrupt an uncheckpointed local phase to deduplicate.

An important source detail: original `predfeat` indexes the **trainx model's**
factors for both ProbeX and qual rows. The fulltrain factors are also retained,
but changing which factors generate qual supports would change the recipe.
The result must contain 50 RBM user components and 20 Tx user plus 20 Tx item
components, each with both query vectors. Verify native receipts and independently
reproduce each support from saved factors and actual row indices before use.

All inputs, copied implementation files and executables are SHA256 bound in
`experiments/upstream-latent90-local/plan.json`. Completion receipts preserve
the actual predictions, both phases' factors, logs and support hashes. Qualifying
targets are zero; no downloaded numerical features or recovered qualifying
outcomes enter training or support construction.

After completion, add the verified supports at a new experiment boundary.
Preserve current user folds and nested selection, replay saved blend models,
then measure the resulting combination. A development improvement alone does
not satisfy the qualifying target.
