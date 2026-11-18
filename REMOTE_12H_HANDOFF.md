# Remote 12-hour fresh-model assignment: handoff

Branch `research/fresh-models-12h`, plan [REMOTE_12H_PLAN.md](REMOTE_12H_PLAN.md),
detailed log [REMOTE_12H_PROGRESS.md](REMOTE_12H_PROGRESS.md). Host: Apple M3 Max,
14 cores, 96 GiB RAM, macOS 27.2.1, checkout `~/workspace/netflix_upstream_repro`.

## Outcome

| Job | Status | Wall time | Peak RAM | Deliverable |
| --- | --- | --- | --- | --- |
| `bknbrx-100` | **Completed** with a fresh native receipt | 9.1 min (03:03:00–03:12:05Z) | 9.98 GB process group, 0 swap | 3 ProbeX NPZ + manifest |
| `cfnade-96` | **Not launched**: the measured cost doesn't fit | — | — | Bounded profile (resource-limit finding) |

The clock started at 2026-11-18T03:01:05Z. Training ended at 03:33Z, far
ahead of the 12:31Z no-new-job cutoff and the 13:01Z hard stop. The rest of the
budget was left unused because the plan assigns nothing else. **No training
processes owned by this assignment are running.** The tmux session `remote12h`
has exited, and no `cfnade-new`, `bknbrx-new`, wrapper or supervisor process
remains (checked with `ps`).

## bknbrx-100 (primary)

- I ran the pinned source `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c` unmodified,
  with `validate_fresh()` passing beforehand. Native config: 5 epochs per phase,
  seed 42, Hogwild, 8 internal threads (not patched). Wrapper environment:
  `UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1`. Rust 1.93.0.
  Hogwild is nondeterministic, so bitwise reproduction isn't expected.
- Phase timings came from the timestamped native log. Trainx epochs took about
  54 s each, and probex reached 0.90551 at 03:07:43Z. The 5 fulltrain epochs
  ended at 03:12:03Z.
- I verified all 13 receipt output hashes (12 `.npy` plus `.out`). The native
  receipt SHA256 is `fc8b6d62c62fd7a11e5b32517d7940bf1fccf60dbc6436645a0314b8f86e940f`.
- Exports are raw finite float32 arrays of shape (1128164,), in native order and
  unclipped:

| NPZ (`experiments/upstream-remote-12h/outputs/`) | Source NPY SHA256 | NPZ SHA256 | Content SHA256 | Diagnostic ProbeX RMSE |
| --- | --- | --- | --- | --- |
| `bknbrx-100.probex.npz` (terminal) | `3e840eb4…30e7ee` | `a6a30251…83e15c` | `0800803c…eab255` | 0.905514 |
| `bknbrx-100_ep01.probex.npz` | `ba4370e3…3368ae` | `1c5778a2…40e9` | `5db9dd90…bf271` | 0.940372 |
| `bknbrx-100_ep05.probex.npz` | `3e840eb4…30e7ee` | `a6a30251…83e15c` | `0800803c…eab255` | 0.905514 |

The full hashes are in `experiments/upstream-remote-12h/bknbrx-100/transfer-manifest.json`.
The **terminal vector and ep05 are identical**, so there are only two unique
vectors to add (ep01 and ep05/terminal). Deduplicate by content hash.

- The alignment SHA256s of the native ProbeX arrays are recorded in the manifest.
  They are: ratings `5a554ca0…a7f2`, user_idxs `d961f596…7b06`, item_idxs
  `b6e42d9d…a856`, dates `c9555bb4…4f47`. They match `fresh-data-receipt.json`,
  which is embedded in the manifest.
- These files were copied to `experiments/upstream-remote-12h/bknbrx-100/`:
  `native-receipt.json`, `native.log`, `native.out`, `native-log-timestamps.txt`,
  `resources.jsonl` (60 s samples), `status.json`, `supervisor.log` and
  `wrapper.out`.
- An optional diagnostic ran after the exports were frozen, comparing them with
  the released fixtures (`fixture-comparison.json`). For ep05, fresh scores
  0.905514 and released scores 0.905493, with RMS difference 0.0101 and
  correlation 0.99985. For ep01, fresh scores 0.940372 and released scores
  0.940417. Released vectors were never used as training inputs.
- Model weights and fulltrain/qual outputs stay in
  `artifacts/upstream-repro/fresh/preds_new/` on this machine.

## cfnade-96 (conditional second): resource-limit finding

The profile is in `experiments/upstream-remote-12h/cfnade-96-profile/profile-summary.json`.
It is **profiling only**, with modified budgets: `--test`, 4 epochs, `swa_start` 3,
trainx/probex only and no `--save`. No predictions were written, and
`preds_new` was unchanged before and after.

- The binary was built from the pinned source with the wrapper's environment.
  The first profile attempt (03:12:43Z) failed with exit 127 because the binary
  didn't exist yet. Its record is kept in `cfnade-96-profile-attempt0-missing-binary/`.
- Epoch 1 took **942.8 s of training** (about 1095 s wall time including init
  and probex eval). It used about 99% of one core: `src/cfnade.rs` has no rayon
  parallelism, so thread settings don't help. Peak RSS was 4.07 GB with no swap.
  The supervisor stopped the profile at its 20-min deadline, and it exited
  cleanly on SIGTERM.
- Projection: training alone takes about 24.9 h per phase, or **≥49.8 h for
  both phases** (75 h with the plan's 1.5× + 30 min margin). Only 9.47 h of
  training budget remained, so the full recipe was not launched. Running it
  needs a dedicated host for about 2–3 days. The two phases are independent in
  their data, but the pinned recipe runs them one after the other in a single
  process. Splitting them across hosts would therefore mean a non-pinned
  invocation.

## Coordinator evaluation

I did **not** run it here. The control bank's inputs aren't on this machine:
the M1 receipts and outputs for rbmx2-150, the rbmx2-150__knnf chain, the
expansion parents and the saved snapshots. The plan says not to regenerate the
baseline. Receiving-side steps, per the plan:

1. Verify the NPZ, content and receipt hashes and the alignment hashes against
   `transfer-manifest.json`.
2. Reproduce the controls 0.8677119362283443, 0.8663200739003126 and
   0.866286461848646, each within 1e-9.
3. Run the arm "latest control + bknbrx-100 {ep01, ep05≡terminal}" (two unique
   columns), with five user folds (seed 20261116), lambda 1000 and output clip
   [1.05, 4.95]. Keep the context interactions for the original fourteen only.
   Report all fold deltas, including negative results.

## Restart requirements

- If bknbrx-100 is re-run, `upstream_repro.py --run-job bknbrx-100` verifies and
  skips the existing receipt.
- cfnade-96: run `UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_repro.py --run-job cfnade-96`
  under `scripts/remote12h_supervisor.py`, with a deadline of at least 3 days.
  The binary is already built. No partial cfnade-96 outputs exist, so the guard
  won't block a run. Native training can't resume, so an interrupted run means
  starting over.
- Tools: `scripts/remote12h_supervisor.py` (deadline/memory/disk-bounded
  process-group supervisor, tests in `tests/test_remote12h_supervisor.py`, 4
  OK) and `scripts/remote12h_export.py --job NAME --kept EE ...` (receipt
  verification, NPZ export and manifest).
- Before this assignment, the checkout was on `research/upstream-reproduction`
  @ db394c1, which is already on origin.
