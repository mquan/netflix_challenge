# Local ensemble continuation experiment

Started 2026-11-15 on `research/independent-ensemble`, in the isolated
`/Users/quan/workspace/netflix_independent_ensemble` worktree.

## Question and scope

Can unfinished ensemble extensions improve the preserved, locally trained
79-column R9-LAT-A2h bank? No downloaded predictions, pretrained upstream
models, or recovered qualifying labels enter this experiment. The local bank
includes historical Probe-guided checkpoint selection; results are development
evidence, not a new untouched test or proof of a historical counterfactual.

## Frozen comparisons

The saved R9 average independently rescores to 0.8668706424233037 on all
1,408,395 original Probe rows. Recombining the separately saved tree and NN
columns scores 0.866870642487332; the maximum float32 rounding difference is
4.76837158203125e-7. All 79 prediction and 46 support/latent column hashes
match the original R9 report. A fresh tree/NN refit is the matched control for
new arms, so library/version changes cannot masquerade as method gains.

All arms retain the original five user folds and user-disjoint inner stopping
split. Tree and NN seeds are 2026, 7, 13. Tree budgets: 8,000 rounds at 0.02,
63 leaves, minimum leaf 200. NN budgets: 60 epochs, patience 5, two 128-wide
layers, dropout 0.2, learning rate 0.0003. Their average stays fixed at 50/50.

1. **Snapshots:** add A2h epochs 1–11 to the existing epoch-12 column. Fit
   blend weights/stopping inside training users, retaining all 12 snapshots
   rather than selecting a snapshot using outer held-out labels.
2. **Nonlinear residual calibration:** add square/cube bases of six fixed
   local predictors, centered at 3.6. These are label-free feature columns;
   calibration coefficients are learned by the fold-local blender. This is
   a nonlinear post-processing experiment, not a reproduction of residual kNN.
3. **Nested subset-network stacking:** use the existing ENN implementation:
   400 small networks on four-predictor subsets, inner-selected epochs and a
   context-binned ridge stacker. Run seeds 2026, 7, 13. Fixed equal averaging
   with the matched tree/NN avoids fitting weights on globally cross-fitted
   ensemble predictions. Outer held-out labels never fit the stacker.
4. **Combined:** snapshots plus nonlinear bases, then the same ENN procedure.
   This tests complementarity without selecting combinations after seeing
   outer scores. The existing R9 bank remains an input, not an editable asset.

Each arm reports overall/fold deltas, seed sensitivity where available, and
paired user-cluster uncertainty conditional on the existing bank. All results,
including negative results, are retained. A small positive gain is recorded;
there is no 0.001 or 0.0003 gate that hides it. No automatic 10% claim follows
from a development score or an assumed Probe-to-qualifying offset.

## Execution and limits

Run `./.venv/bin/python scripts/local_ensemble_run.py --prepare`, then
`./.venv/bin/python scripts/local_ensemble_run.py --run` from this worktree.
The prepared plan, input/code hashes, status, logs and results live under
`artifacts/independent-ensemble/`. Training is sequential, CPU-only, at reduced
priority, with two parallel tree fold workers and two numerical/Torch threads. There is no
base-model retraining in this first experiment. A successful development arm
requires a separate production/refit feasibility audit before a qualifying run.

`status.json` and `pipeline.log` are the live status. The runner checkpoints
each completed stage; a resumed run verifies the original plan and code hashes.
It refuses changed inputs, changed code, conflicting output files and duplicate
runners. Other worktrees' jobs and files are not modified.

## Launch checkpoint

The runner started at 2026-11-16 05:08 UTC (November 15 Pacific), PID 29626,
with 14 predeclared stages. The initial stage is the fresh baseline tree fit.
See `experiments/LOCAL-ENSEMBLE-20261116-launch.json` for the launch receipt.
Two new end-to-end/edge-case checks and 13 existing Round 9 checks passed.
A separate historical latent-export test fails because its reader accepts
checkpoint versions 1/2 while the current C++ writer emits version 3. That
exporter is not invoked by this run: the original 46 support/latent columns
are preserved and hash-verified. The failure is recorded in the validation
receipt rather than represented as a passing full-suite check.

## Parallel execution revision

The user requested a shorter turnaround. Before any full-data fold result was
reported, the initial one-worker run was interrupted and its logs preserved.
The same 14-stage protocol now uses two concurrent tree folds. A separate
low-memory ridge screen evaluates the baseline, snapshots, calibration and
combined inputs on outer user fold 0, with the same fixed penalty as R9. It
shares one streamed Gram accumulation across the four conditions, matches the
historical ridge control, and fits only on the other four user folds.

This screen produces an early diagnostic rather than a full tree/neural
comparison. Its results do not select or cancel the predeclared full-trial
arms. Three tests now cover empty-bin fallback, one-versus-two-worker outer
label poisoning, and streamed-versus-direct ridge fitting.

The five-fold ridge diagnostic completed in 27.54 seconds: baseline
0.8685312898, snapshots 0.8684460182, nonlinear calibration 0.8684287526,
and combined 0.8683478614. All three treatments improve every fold; combined
gains 0.0001834284. This establishes a small development gain over the matched
ridge only. The stronger 0.8668706424 R9 tree/neural blend remains unbeaten by
this diagnostic, and the original full-treatment comparisons continue.

## Second computer

[REMOTE_ENSEMBLE.md](REMOTE_ENSEMBLE.md) provides clean-checkout setup, verified
input restoration, automatic CPU/CUDA/Apple-GPU selection, five-fold CPU
parallelism, resuming and result export. The separate worker overlaps one CPU
tree job with one neural job and trains its own matched control. CPU and Apple
GPU outer-label isolation were tested locally; CUDA is not available on the
development Mac. The original Mac run continues with its frozen CPU plan.

## Confirmed distributed assignment

The user confirmed that the second computer is running the default 10-stage
job, with an estimated 5–8 hours total. The Mac therefore completes only its
six stages: fresh control tree/NN, snapshots tree/NN, and calibration tree/NN.
A separate assignment monitor checks every completion receipt and report hash
before ending the remaining local queue. Combined treatment and both ENN
comparisons are delegated to the remote machine. This allocation is based on
job ownership, not observed treatment quality. The full original plan remains
preserved. Remote results are compared to that machine's own fresh control.
