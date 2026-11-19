# Remote laptop: nine-hour execution handoff

Work branch: `research/remote9h-handoff`. Return commits to this branch.
The coordinator continues on `research/full-upstream-reproduction`; do not
merge unrelated history or overwrite the coordinator's frozen model scripts.

This assignment is ready to execute, not another planning task. Use the existing
M3 Max /96 GiB laptop and its own previously completed native artifacts. Record
your actual UTC start; stop training at start+8h30m and reserve the final30m for
verification and return. Use actual process and log observations for status.

## 1. Recover our already-completed temporal qualifying outputs first

This is the highest-priority task. The laptop completed both original25-epoch
phases of `tsvdx5-150o` on November18. Only ProbeX was returned before. Its
qualifying predictions may still exist under
`~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/preds_new/`.
Our current laptop has independently reproduced the first20 qualifying snapshot
hashes exactly, but is still finishing that phase. Recovering the completed
outputs could unlock the qualifying blend immediately.

Preserve existing work and check for live jobs before changing a checkout.
Prefer a fresh worktree of this branch at `~/workspace/netflix_remote9h`, using
the existing Python environment and the existing native tree explicitly:

```sh
git fetch origin research/remote9h-handoff
git worktree add ~/workspace/netflix_remote9h -b research/remote9h-execution origin/research/remote9h-handoff
cd ~/workspace/netflix_remote9h
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote9h.py export-temporal \
  --native-tree ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh \
  --output experiments/upstream-remote9h/temporal-qual
```

Adjust the existing checkout/environment path only if needed. The exporter
requires the exact original receipt SHA
`12421a92ea6495ece7347ecf8e7d794c52d52c8e1f14e7fbc5914e455708dac7`, the pinned
source, unchanged tracked source code, all63 original output hashes, zero native
qualifying targets, and finite float32 vectors with2,817,131 rows. It exports
terminal +ep06/16/24 qualifying vectors losslessly, with both native and transfer
hashes and qualifying row-identity hashes. Do not use released prediction files
or recovered qualifying labels as substitutes.

**Commit and push that export immediately**, before starting the long run:

```sh
git add experiments/upstream-remote9h/temporal-qual
git commit -m 'Recover own completed temporal qualifying predictions'
git push origin HEAD:research/remote9h-handoff
```

If the native outputs are absent or hashes differ, preserve the evidence and
push a short missing-output report immediately; proceed to task2. Do not spend
hours retraining the temporal model or trying public prediction files.

## 2. Resume our checkpointed CFNADE48 trajectory

The warm-start package contains our two phase checkpoints, all84 parameter,
Adam/SWA/RNG tensors per phase, every earlier native epoch prediction, frozen
source, source/data identity, and arm64 stage/parity executables. This is our own
training, not downloaded public pretrained features. The original preset stays
160 hidden /48 rank /95 epochs per phase, with the original source parameters.

Download the transfer-only **draft** release asset using the owner's GitHub
credentials. Its exact size/SHA, starting epochs and checkpoint identities are
in `experiments/upstream-remote9h/cfnade48-package.json`:

```sh
mkdir -p artifacts/upstream-remote9h
gh release download remote9h-cfnade48-20261119 --repo mquan/netflix_challenge \
  --pattern cfnade48-resume.tar.gz --dir artifacts/upstream-remote9h
```

Verify the archive SHA against the tracked metadata before extracting it, then
extract into that owned directory. Keep the original archive immutable. Run:

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote9h.py verify-bundle \
  --bundle artifacts/upstream-remote9h/cfnade48-package \
  --data-root ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data
```

This verifies every package member, all36 actual data-array hashes, zero
qualifying targets, original runtime config and actual native160/48 parity.
Run this preflight before training changes the copied checkpoint pointers.
Create `native-source/data` inside the extracted package as a symlink to that
verified existing data directory. Do not copy or publish the raw dataset.

Launch **two direct stage processes**, `bin/cfnade48-stage x` and
`bin/cfnade48-stage qual`, with cwd set to the package's `native-source`.
Use `scripts/remote12h_supervisor.py` separately for each direct native process,
with descriptive names `remote9h-cf48-x` /`remote9h-cf48-qual` and the common
start+8h30m UTC deadline. Run under `caffeinate`; use tmux if available.
The original old12h deadline must not be reused.

Set these environment values from the extracted manifest, using absolute paths:

- `CF_CHECKPOINT_DIR`: the package's `checkpoints` directory.
- `CF_CHECKPOINT_SOURCE`: its `source_id`.
- `CF_CHECKPOINT_DATA`: its `data_id`.
- `RAYON_NUM_THREADS=2`; BLAS/OMP/Accelerate thread limits1.
- Clear `CF_EXIT_AFTER_CHECKPOINT`, `CF_PROOF_OUTPUT`, `CF_PROOF_INTERRUPT`.

Use the copied executable first. If it cannot run on this Mac, rebuild the
unchanged included source into a separate target directory with Cargo.lock
preserved, rerun the native parity binary, and record the actual new binary
hash and commands. Do not change math, precision, source/config identity,
rank, epoch count or checkpoint contents to force a resume.

### Required resume proof and immediate progress return

Before launching, move only each owned copied latest-epoch prediction to an
owned baseline filename. The original checkpoint restore regenerates that epoch
prediction before advancing. Require `Exact CFNADE checkpoint restored` with
the expected training split/epoch, and verify the regenerated NPY hash exactly
against `manifest.phases[split].prediction_sha256`. If either differs, stop only
the owned supervisors/native processes, preserve everything and report failure.
Do not admit a mismatched resumed trajectory.

Push the successful resume proof promptly: package SHA, source/data/config
identities, both starting epochs, regenerated prediction hashes, actual binary
hashes/commands, supervisor/native PIDs observed live, and UTC time. The
coordinator can then decide when to retire its duplicate fallback workers.
**Do not signal or control the coordinator's jobs.**

Every30m push concise status, actual epochs and process/resource observations
under `experiments/upstream-remote9h/`. Original model capacity and all95 epochs
remain required even if the nine-hour window expires first.

## 3. Return complete results or exact resumable state

If both phases finish95 epochs: verify final checkpoint identity and all84 tensor
MD5s, all95 snapshot arrays plus the terminal vector per split, correct shapes,
finite float32 values, and terminal equality to epoch95. Record an explicit
resumed-training receipt with the original reference command, actual executed
commands/binary hashes, package/checkpoint origins and complete output hashes.
Do not fabricate a fresh uninterrupted `u.model_job` execution.

Export both terminal vectors losslessly as NPZ `prediction`, along with native
NPY hashes, transfer hashes, row-identity hashes, receipts and logs. Commit/push
the small outputs immediately; no local blend selection is needed. Qualifying
labels remain zero throughout; the coordinator performs frozen-vector auditing.

If unfinished at the training deadline: stop the owned supervised processes,
preserve the latest complete optimizer/SWA/RNG checkpoints and snapshots, and
return a verified archive through a draft release asset plus its tracked SHA.
Push a truthful partial-state report; partial models do not count as completed
95-epoch models or target success. Finish verification/return within nine hours.

Do not start CFNADE96 from scratch or duplicate coordinator AEX/temporal fits.
If both CF48 phases finish early, return results first, then report remaining
time before choosing additional original source work.
