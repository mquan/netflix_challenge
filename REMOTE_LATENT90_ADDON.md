# Remaining remote window: reproduce the90 original latent support features

CFNADE48 is complete95/95 and its terminal pair has been verified at the
coordinator. Preserve its entire source/output/checkpoint tree. Use remaining
remote capacity for the two exact source recipes below. Do not restart CF48.
Keep any already-running CF96 supervisors intact; the96GiB laptop can use
separate owned processes. Do not touch coordinator jobs.

**Keep the existing deadlines: November19,2026,14:33:28UTC for training and
15:03:28UTC for verified artifact return. Do not reset the clock.**

The public bank's220 support columns are130 voting features and90 latents:
50 `rbm50lat.uXX`,20 `tx20lat.uXX` and20 `tx20lat.iXX`. We already have the
130 native voting columns. Reproduce the missing90 from our own training;
do not fetch public parameter or prediction files.

## Isolate the work and reuse only our verified input data

Fetch `origin/research/full-upstream-reproduction` and read this file from it.
Use a separate worktree/output tree; preserve active supervisor checkouts.
The existing native tree should be
`~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh`.
Verify its source HEAD equals
`49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`, with tracked changes restricted to
the previously verified zero qualifying payload. Verify actual
trainx/probex/fulltrain/qual arrays against the source/data hash bindings in
the CF48 origin plan/package. Require all qualifying targets zero.

Create a new owned source tree from that pinned source's tracked code and
configuration files, excluding `data/` and all `preds_*` payloads. Symlink only
the verified own four dataset directories into its `data/`; never run ingest.
Keep outputs under this new tree's `preds_new/`. Record all copied source and
actual dataset hashes. The full resolved original jobs are also available in
`experiments/upstream-all-pipeline-coverage/registry.json` under `new.jobs`.

Build the unchanged `fit` and `predfeat` binaries with Cargo.lock preserved.
Reuse the existing Rust toolchain/dependency cache. On macOS, the established
`blas` build uses `RUSTFLAGS` containing
`-C link-arg=-framework -C link-arg=Accelerate`. Record actual Cargo commands,
features, compiler/version and executable hashes. Do not alter model source.

## Train the two original presets concurrently if resources allow

From the isolated source tree, with all unrelated model-tuning environment
variables cleared, run the exact original configurations:

```sh
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 EPOCHS=20 H=50 UFEAT=1 \
  ./target/release/fit rbm rbm50lat -p pipeline-new.toml --final --preds preds_new

RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 EPOCHS=20 F=20 IFEAT=1 \
  LR_W=0.002 NBR=25 UFEAT=1 WF=0.8 \
  ./target/release/fit tx tx20lat -p pipeline-new.toml --final --preds preds_new
```

Use owned `remote12h_supervisor.py` processes and caffeinate/tmux, with the
existing14:33:28UTC deadline. Preserve default model seeds and remaining source
parameters. Both original20-epoch phases are required for completion. Do not
reduce rank, precision, epochs, data or logical training recipe to fit the window.
If the deadline interrupts an uncheckpointed phase, preserve any completed
phase outputs and report partial work truthfully; do not claim the two-phase
model complete or silently restart from scratch.

After each complete model, hash all native outputs and verify the declared
source outputs, query vector shapes, finite float32 values and factor matrices.
Keep the source configs and complete logs. Query row counts are1,128,164 for
ProbeX and2,817,131 for qual. User factors have480,189 rows; item factors17,770.

## Derive the90 supports and return the small sufficient artifacts

After both original models complete:

```sh
./target/release/predfeat -p pipeline-new.toml --latent-users rbm50lat
./target/release/predfeat -p pipeline-new.toml --latent-users tx20lat --latent-items tx20lat
```

Verify exactly90 paired support columns under `preds_new/lat/`, with finite
float32 values and matching query row counts. Record every support SHA256.

Return a draft release archive containing both phases' `rbm50lat.ufeat.*.npy`,
`tx20lat.ufeat.*.npy`, `tx20lat.ifeat.*.npy`, the two models' terminal query
vectors, complete native logs, source/config/build/actual-command receipts and
all native/support output hashes. The six factor matrices are sufficient for
the coordinator to regenerate the90 row-wise supports with the original
`predfeat` code. If practical, include the support vectors too. Do not put large
arrays directly in Git and do not copy raw datasets. Declare any native outputs
that were hashed remotely but not transferred; do not fabricate local receipts.

Immediately push small progress/provenance/transfer metadata to a new branch
`research/remote-latent90-return`, with the exact draft release tag/asset name,
size and SHA256. Update progress at least every30minutes. Preserve originals
and finish verified return by15:03:28UTC. The coordinator verifies intake and
fits/evaluates its own blends; no qualifying-label scoring or weight selection
is needed on this laptop.
