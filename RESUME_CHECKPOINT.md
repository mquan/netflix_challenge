> **Superseded 2026-10-31:** the current resume point is [STATUS.md](STATUS.md). This file is kept as history.

# Resumed at user request, 2026-10-30

The root-owned serial research chain is `SO1-resume-20261030`. Inspect
`experiments/SO1-resume-20261030.status.json` and
`artifacts/selection-objective/SO1-resume-20261030.out` before launching any GPU
work. It is detached under caffeinate and must not be duplicated. The original
Oct 19 launch/status records and other research worktree remain unchanged.

N1's saved epoch-13 checkpoint passed its original honest P2 gate:
bank10+A2h 0.8477472632 versus bank10+A2h+N1 0.8465344856, improvement
0.0012127776 against the 0.001 minimum. Standalone N1 remains worse than A2h.
This is reused P2 development evidence, not an ensemble promotion.

The chain executes a fixed 13-epoch N1 full-training refit, then the original
three-seed LightGBM/NN2 comparison on R9-LAT-A2h, then the frozen SO1 P2 pair.
If N1 improves all five full-bank folds, it prepares a separate SO1 full-plan
against that bank, preserving the old plan. Full SO1 training requires the
unchanged 0.0005 P2 gate. Any failed subprocess or validation stops the chain.
No score is promoted automatically. The last promoted score remains
0.8668706425 until the completed comparison is reviewed.

The process guard excludes only the exact Runlayer transcript-worker command;
all other Python executables still block. Six focused adapter tests and the
13 existing queue/gate/full-stage tests pass. Scientific trainers, original
plans and gate thresholds are unchanged. Frozen N1 sources, plan and evidence
are under `experiments/N1-resume-20261030-*`.

Do not restart this chain in place: inspect its recorded failure and outputs,
then prepare an explicit recovery with fresh names. The original worktree's
selection report is incomplete because it was paused; use the saved epoch-13
progress/checkpoint evidence, not an inferred selection-run completion.

## Historical pause, 2026-10-19

Work is on `codex/local-context`; no PR is requested. The 10% goal remains
unachieved. Promoted development RMSE is 0.8668706425; the estimated development
target is 0.86465. Qualifying labels are unavailable, so estimated attainment
must not be reported as verified hidden-test success.

Latest committed N1 evidence is `experiments/N1-epoch13-observation.json`:
selected epoch 13, P2 RMSE 0.8683384740210056. Epoch 14 was running at pause.
N1 belongs to the separate `.claude/worktrees/honest-residual-chains` worktree
and was left untouched under the existing read-only restriction.

The SO1 waiting runner PID 67792 was interrupted before either training arm
started. Process inspection confirmed both it and its caffeinate child 67793
exited. Its status records `failed` / `KeyboardInterrupt()` because the frozen
runner records interruption that way; this was an intentional user-requested
stop, not a training failure. Preserve the original launch and status records.
No SO1 control, treatment or gate artifacts were found at pause.

On resume:

1. Inspect live processes and the N1 chain status, log and root progress report.
   Do not restart N1 or infer termination from a quiet log.
2. Verify the original SO1 plan and launch bindings, and confirm that neither arm
   nor gate has run. The original queue name refuses existing records. Restart
   the unchanged queue with a fresh unused `--name` (for example
   `SO1-queue-resume1`) after those checks; retain the original frozen plan.
   The runner still waits for N1's full chain and two idle Python snapshots.
3. Evaluate the fixed SO1 gate before launching the prepared full-data comparison.
   Refresh the full-plan reference if N1 has promoted a new bank. See
   `SELECTION_OBJECTIVE_RESEARCH.md` for the gate and full-stage procedure.

Do not launch any work until the user resumes. Continue committing on this
branch without creating a PR.
