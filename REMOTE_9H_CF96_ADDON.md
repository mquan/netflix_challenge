# Additional remote lane: our original CFNADE96 checkpoints

Branch: `research/remote9h-cf96-handoff`. This supplements the existing CFNADE48
run; preserve it and its original deadline. The remote laptop's latest returned
heartbeat reports no swap growth and ample memory. The coordinator yielded its
CFNADE96 workers after measuring active paging, preserving epochs33/31 with
all84 tensors rehashed. These are our own training states, not public pretrained
prediction features.

**The deadline remains November19,2026,14:33:28UTC for training and15:03:28UTC
for verified return. Do not start a new nine-hour clock.** Return partial exact
checkpoints if the original95-epoch phases do not finish within that window.

## First: return the completed temporal training outputs

The coordinator is still generating its duplicate model's training residuals.
The remote laptop already has the complete original63-output job. Return those
outputs first so the coordinator can retire that duplicate only after verifying
its full completed counterpart, freeing memory for further training.

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote_temporal_archive.py \
  --native-tree ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh \
  --output artifacts/upstream-remote9h-temporal-complete
```

This checks the exact original receipt, all63 native hashes, complete vector
shapes, source pin and zero qualifying targets. It includes only model outputs
and provenance, not raw datasets. Upload `temporal-native-complete.tar.gz` to a
new draft release asset, and immediately commit/push `transfer.json`, archive
tag/asset names and the native receipt to `research/remote9h-cf96-handoff`.
Do not put the large archive directly into Git. Preserve CFNADE48 and all old
native files. If these outputs are absent or changed, report it and continue
with the checkpoint lane below. Do not retrain the temporal model.

## Download and verify CFNADE96

Use a new worktree for this branch, or read its files without changing the
checkout used by active CFNADE48 supervisors. Keep its package and output
directories separate. Fetch the branch and read
`experiments/upstream-remote9h-cf96/package.json` for the exact archive size,
SHA256, source/data identities and captured epochs.

```sh
mkdir -p artifacts/upstream-remote9h-cf96
gh release download remote9h-cfnade96-20261119 --repo mquan/netflix_challenge \
  --pattern cfnade96-resume.tar.gz --dir artifacts/upstream-remote9h-cf96
```

Verify the archive SHA against tracked metadata before extracting into that
owned directory. Preserve the original archive. Then run:

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote_cf96.py verify-bundle \
  --bundle artifacts/upstream-remote9h-cf96/cfnade96-package \
  --data-root ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data
```

This verifies every package member, all declared actual data hashes, zero
qualifying targets, both owner identities, all84 tensors per phase, source
configuration and native original-source parity. Run it before training changes
the copied checkpoint pointers. The package includes owner files; do not modify
checkpoint metadata or tensors to force a restore.

## Run both additional phases

Set `P` to the absolute extracted `cfnade96-package` path. Create
`$P/native-source/data` as a symlink to the verified existing data directory.
Do not copy or publish raw datasets.

Use the copied binaries unchanged. The stage binary accepts `x` or `qual`;
it does not implement `--describe`. Its canonical configuration is320 hidden,
rank96,95 epochs, seed42, with the original source flags. Rebuild frozen source
only if the copied executable cannot run; preserve Cargo.lock, rerun parity and
record the actual new binary hash and commands.

Set `CF_CHECKPOINT_DIR=$P/checkpoints`, `CF_CHECKPOINT_SOURCE` and
`CF_CHECKPOINT_DATA` from tracked metadata, `RAYON_NUM_THREADS=2`, and all
BLAS/OMP/Accelerate thread limits1. Unset `CF_EXIT_AFTER_CHECKPOINT`,
`CF_PROOF_OUTPUT` and `CF_PROOF_INTERRUPT`.

Launch two owned supervisors under `caffeinate`/tmux using the existing
`scripts/remote12h_supervisor.py`, with cwd `$P/native-source`:

```text
--name remote9h-cf96-x    --out-dir experiments/upstream-remote9h-cf96/run/x
--name remote9h-cf96-qual --out-dir experiments/upstream-remote9h-cf96/run/qual
--deadline 2026-11-19T14:33:28Z --cwd $P/native-source
-- $P/bin/cfnade-stage x|qual
```

Before launch, move only each owned copied latest snapshot into an owned
baseline directory. Require the native `Exact CFNADE checkpoint restored` log
with the captured split/epoch. The regenerated latest-epoch NPY must match the
original snapshot SHA in `manifest.phases[split].prediction_sha256` exactly.
If either restore differs, stop only these owned CF96 supervisors, preserve the
evidence and report failure. Do not admit a mismatched trajectory.

Push the successful restore proof immediately to `research/remote9h-cf96-handoff`:
package/source/data IDs, actual native binary hash, both captured epochs and
regenerated hashes, actual commands, supervisor/native PIDs observed live,
UTC time and current resource observations. Do not control coordinator jobs.

## Preserve CFNADE48 priority and return progress

Keep the original CFNADE48 supervisors running. Push CF96 checkpoints/epochs,
actual live processes, memory/swap/disk observations and concise status every
30minutes on this add-on branch. If adding CF96 causes sustained swap growth
or doubles CF48 epoch time across two samples, stop only owned CF96 supervisors
and preserve their latest complete states. Do not reduce any model capacity,
epoch count, precision or logical training recipe.

At the shared training deadline, stop owned CF96 supervisors and preserve the
latest complete checkpoints and all epoch predictions. Verify owner/identity,
state-manifest MD5 and all84 tensor MD5s per phase; include RNG/timestep/SWA
state. Return a resumable archive through a draft release asset, with exact
archive/member hashes, actual elapsed work, commands and source provenance.
Push metadata before15:03:28UTC. Include the owner files in the return package.

If both phases finish95 epochs, verify all95 snapshots and terminal predictions
for both splits, finite float32 values, correct row counts and terminal/epoch95
equality. Use the supplied `cfnade-export-raw` only after both95-epoch states
exist, with the same identities, to return the already proved same-trajectory
raw96r variant. Export terminal vectors losslessly as NPZ `prediction`, with
native NPY hashes, transfer hashes, row identities and a truthful resumed-run
receipt. Partial states are not completed95-epoch models or goal success.

Qualifying outcomes stay zero. The coordinator performs known-ProbeX candidate
selection, production blending and the frozen recovered-group audit.
