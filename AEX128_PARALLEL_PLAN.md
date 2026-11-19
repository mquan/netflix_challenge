# Bring forward the original AEX128 fit

The final DNN endpoint has completed, leaving capacity on this ten-core machine.
Run the original `aex-128` job in a separate unmodified native checkout while the
primary scheduler continues attention models. This brings forward one of the
remaining fixed six-model extension inputs without changing its source recipe.

`scripts/upstream_aex_backfill.py` reuses the existing native-tree preparation
routine with a distinct source, prediction, receipt, build and lock directory.
Only hash-verified completed primary parents are hardlinked; no active primary
output scan is modified. Native data aliases are identical to the primary tree,
and qualifying training targets remain zero. The source preset is unchanged:
128 factors, 20 epochs per phase, seed 42, regularization 0.002 and dropout 0.1.
Both original native phases must finish successfully. All receipt output hashes
and paired finite float32 arrays are checked before completion is published.

The worker runs at lower priority. Admission reserves 4 GiB under a 23-GiB total
RSS limit. A sustained total above 24 GiB, own RSS above 5 GiB or free disk below
22 GiB stops only this owned job and preserves its partial outputs. Existing
native recipes, iteration counts and live source files are unchanged.

`scripts/upstream_import_aex.py` waits for the DNN import and primary scheduler
handoff to publish matching completed evidence. It verifies the new scheduler's
PID and binds both predecessor results. It then reuses the unchanged transactional
native import implementation, targeting the new AEX tree and checking the
successor command identity. The AEX source lock must be free and its native job
complete before preflight. Source outputs are linked only while the primary
scheduler is idle with no live wrapper group. Existing valid receipts survive;
conflicts fail closed and unpublished links roll back. Scheduler resumption runs
in the original finally block. No active fit is terminated to force reuse.

Tests cover both predecessor gates, their hash binding, stale successor PIDs and
real successor-only STOP/CONT on dedicated test processes. The original five
transactional import/recovery tests are also retained. Launch and execution
plans bind the new adapters and inherited implementation hashes.

This produces one additional own source model. It does not change the existing
85-column qualifying attempt, make partial predictions eligible, or establish
a development or qualifying score improvement.
