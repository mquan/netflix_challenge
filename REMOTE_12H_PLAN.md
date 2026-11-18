# Fresh-model assignment for the second laptop: 12 hours

Prepared 2026-11-17. Assignment branch: `research/fresh-models-12h`.
Start the 12-hour clock when the agent begins work, including setup and export.
This is a new, bounded remote assignment. It supersedes the previous remote
revocation only for the work in this document. Do not restart the old expansion,
eight-model queue, independent-ensemble queue, or production queues.

## Objective and frozen control

Train complementary models ourselves with the pinned upstream implementations.
No downloaded predictions, pretrained weights, released epoch coefficients, or
recovered qualifying outcomes may enter training or the fresh blend.

The coordinator's current fresh-only control is **0.866286461848646** on
1,128,164 ProbeX rows: fifteen terminal model/chain outputs, 42 unique snapshots,
five components, five context main effects and 70 interactions. Preserve it.
Its evidence is `experiments/upstream-missing-screen/replay/evaluation.json`;
its restart instructions are `RESTART_HANDOFF.md`.

The remote computer does **not** need this entire bank to train and export new
models. Training and transfer are its primary deliverables. The coordinator
can later evaluate them using its existing verified bank. An export without a
fresh receipt is not a completed training result.

## Fixed assignment

1. **Primary: `bknbrx-100`.** Exact pinned neighborhood model, five epochs per
   native training phase, seed 42, eight-thread Hogwild implementation. Retain
   terminal ProbeX predictions and pipeline-kept epoch 1 and epoch 5 outputs.
   Hogwild is nondeterministic; bitwise reproduction is not required. Record
   the actual configuration and provenance. Do not change its internal thread
   count by patching model source.
2. **Conditional second: `cfnade-96`.** Exact pinned CF-NADE preset, 95 epochs
   per phase; retain terminal predictions and pipeline-kept epochs 10 and 95.
   Launch only after measuring cost and confirming it can finish within the
   remaining training budget. No capacity sweep, seed sweep or epoch selection
   based on ProbeX performance.
3. **If the second full job cannot fit:** deliver a bounded CF-NADE throughput
   and RAM profile instead. Inspect `src/bin/cfnade-new.rs`'s `--test` interface
   before constructing a command. Keep profiling outputs in a separate directory
   and label all modified budgets as profiling, not a completed pinned recipe.
   Do not overwrite canonical model outputs or use profile predictions in the
   primary fresh-bank comparison.

Released-vector diagnostics suggested bundle gains of about 0.00078 for
`bknbrx-100` and 0.00103 for CF-NADE against the earlier 0.86632007 control.
Those are candidate-selection evidence, **not promised fresh gains** and not
measurements against the latest control. Include negative fresh results.

## Time and resource budget

| Elapsed time | Work and decision |
| --- | --- |
| 0–1 h | Inspect host, existing artifacts/processes, setup, tests and job plans. Start primary as soon as ready. |
| 1–9.5 h | Primary training; then conditional CF-NADE training/profile. Record phase/epoch timings and memory. |
| 9.5–10 h | Drain training, finish validated exports. Do not start another native job. |
| 10–12 h | Verify transfers, commit/push results, evaluate only if fresh control is available, write restart handoff. |

These are allocation limits, not runtime predictions. Setup counts against the
window. Record absolute UTC start, training cutoff and final deadline before
training. The supervisor must stop its own training process group at hour 10
even if a native phase is unfinished; stop profiling after at most 20 minutes.
Give SIGTERM a short grace period, then stop any remaining owned children.
Never use broad process-name kills. Preserve logs/partial outputs and mark the
job interrupted. Native training is not assumed to support optimizer resume.
If implementing a supervisor, test deadline/process-tree cleanup with a toy
child before using it on real training.

Run one native model job at a time. Start with RAYON=8 and BLAS=1, recording
CPU count and physical RAM; reduce the outer RAYON limit on smaller hosts if
needed, but bknbrx's internal pinned pool remains eight threads. These Rust
models use the CPU; the GPU is not automatically used.

Measure the whole process tree's RAM, disk space, swap growth and native phase
timings at least every five minutes; publish a heartbeat at least every 30
minutes. If sustained memory pressure or swapping threatens the host, stop
only this assignment's job and record the resource limit. Leave at least 20%
of physical RAM available for the OS/other work, and check free disk before
launch. The fulltrain phase may have a higher peak than trainx.

Do not launch CF-NADE without a conservative estimate covering **both** native
phases, setup/build, validation and export. Use at least 1.5 times a measured
projection plus 30 minutes; unknown timing is not evidence it fits. Use a
separate bounded profile first if needed. If setup consumes over an hour,
prioritize the primary job and explicitly reduce scope in the status report.

## Checkout and preflight

Prefer the laptop's idle upstream-reproduction checkout, which already holds
the verified native data, toolchain and first eight jobs. A normal git branch
switch preserves its ignored artifacts. A new worktree does not contain those
artifacts automatically. Inspect local changes and active jobs first; do not
overwrite changes, switch an active worker's checkout, or run concurrent
wrappers against the same `fresh/` directory.

```sh
git fetch origin research/fresh-models-12h
git switch --track origin/research/fresh-models-12h
```

If the local branch already exists, switch to it and use an ordinary
fast-forward-only update. If it has diverged, preserve local work and report
the conflict rather than rebasing or force-pushing automatically.

Read `UPSTREAM_REPRO.md` for initial setup if the native tree is absent. Install
NumPy and tomli in the venv; use the isolated Rust toolchain where available.
Only bootstrap/build/prepare/setup data if actually missing. Do not download
the released reference bank merely to run this assignment.

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
.venv/bin/python scripts/upstream_repro.py --plan-job bknbrx-100
.venv/bin/python scripts/upstream_repro.py --plan-job cfnade-96
```

The wrapper resolves explicit and implicit native dependencies. Verify the
exact source commit `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`, unchanged
model sources/configuration, data receipt and zero-sentinel qualifying
outcomes with `upstream_repro.validate_fresh()` before starting. Qualifying
query identities may be implicit inputs; their outcomes must remain zero.

The following are native job commands, **not deadline-aware launchers**. Run
them only under the bounded supervisor/monitor described above, within tmux
and with macOS sleep prevention if applicable:

```sh
UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_repro.py --run-job bknbrx-100
# Conditional; never start automatically just because the first command exited:
UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_repro.py --run-job cfnade-96
```

Each native job runs trainx/ProbeX and fulltrain/qual phases. The job wrapper
writes its execution receipt only after the whole recipe completes. Reuse
completed jobs only after verifying every receipt output hash. If partial
unreceipted files exist, preserve and inspect them; do not bypass the guard or
delete them to force a retry. Saved epoch vectors alone are not optimizer
checkpoints or completed-job evidence.

## Transfer contract and progress publishing

Own only `REMOTE_12H_PROGRESS.md`, `REMOTE_12H_HANDOFF.md`,
`experiments/upstream-remote-12h/` and any new assignment-specific supervisor
and tests. Preserve the plan and unrelated tracked files. Do not run old
publishers: they target other branches. Commit/push at startup, on each job
completion, and at cutoff using named paths and ordinary pushes to this branch.
Do not stage `.venv`, whole artifact trees, raw data or qualifying outputs.

For each completed native job:

- Verify **all** native receipt output hashes, not only its terminal vector.
- Export terminal and declared kept ProbeX epoch vectors as separate compressed
  NPZ files (`prediction` key) under `experiments/upstream-remote-12h/outputs/`.
  Require finite float32 arrays with shape `(1128164,)`; preserve native order
  and raw values without adding export-time clipping.
- Copy the native execution receipt, full native log and resource profile.
  Include source NPY SHA256, NPZ SHA256, prediction-content SHA256 and owning
  native receipt SHA256 in a transfer manifest. The receipt must bind each
  exported source path; confirm pipeline behavior rather than guessing names.
- Include hashes of native ProbeX `ratings.npy`, `user_idxs.npy`, `item_idxs.npy`
  and `dates.npy` for alignment, plus `fresh-data-receipt.json`, source pin,
  host/OS/RAM/thread information and command/environment. Do not transfer labels
  themselves if matching hashes suffice. Model weights and fulltrain/qual
  predictions stay on the originating machine.
- Standalone ProbeX RMSE may be reported as a diagnostic, never as ensemble
  improvement. A fixture comparison is optional and must happen after freezing
  the fresh output; unavailable fixtures are not training failures.

Each compressed prediction is only a few MB. The data, native weights and
qualifying arrays are deliberately excluded from Git. Keep failure records
and interrupted profiles even when no completed model can be exported.

At the deadline, handoff must identify completed/partial jobs, output hashes,
actual time/RAM, remaining work, exact restart requirements and confirmation
that all owned training processes have exited. A successful bounded run can
finish with only the primary model or a resource-limit finding; never label
an uncompleted recipe complete.

## Coordinator evaluation after transfer

This evaluation is a separate receiving-side task; the remote worker should
not regenerate the entire baseline just to perform it. If all required fresh
bank artifacts happen to be present, it may evaluate within the reserved time.

1. Independently verify receipt/export hashes and exact metadata alignment.
2. Reproduce 0.8677119362283443 (fourteen-output control),
   0.8663200739003126 (snapshot/context control), and
   **0.866286461848646** (latest control), each within 1e-9 before accepting gains.
3. Predeclared arms: latest control; control + all unique primary terminal/kept
   epoch outputs; control + all unique CF-NADE terminal/kept epoch outputs if
   complete; control + both bundles if both complete. Deduplicate exact vectors
   by content hash. Do not choose an epoch from outer-fold scores.
4. Preserve five user folds (seed 20261116), lambda 1000, unpenalized intercept,
   native input clipping rules, output clip [1.05,4.95], and the original
   fourteen-only context interactions. Add new rating columns globally for the
   primary comparison; changed context interactions are a separate experiment.
5. Fit each fold's coefficients on its other four user folds. Report aggregate
   RMSE, all five paired fold changes, prediction hashes and runtime, including
   failures and negative results. Retain the frozen control regardless of outcome.

The current `upstream_selected_replay.py` is hardcoded to the earlier RBM
replay; do not merely point it at a different job. Extend evaluation in a new
assignment-specific script with control-parity and held-fold-label-isolation
checks if needed. No recovered qualifying evaluation is assigned here.

These are exploratory development comparisons on historically selected source
recipes, with candidate selection informed by released-vector diagnostics.
Neither this plan nor a ProbeX score establishes the 10% qualifying target.
