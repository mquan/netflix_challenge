# Continue the source DAG within the current laptop's capacity

The original scheduler remains unchanged. Its existing storage hold list misses
the `tsvdx4p-*` heap models; the largest of these exceed this laptop's memory.
It could also duplicate the original CFNADE-48/96/96r work already assigned to
independent native workers.

`scripts/upstream_primary_handoff.py` waits for the auxiliary artifact import to
finish and for a hash-verified completed primary job. It suspends only the
scheduler and checks that no immediate child or previous wrapper process group
is live. A racing wrapper causes immediate resumption and retry. At a confirmed
idle boundary it terminates that scheduler, resumes it so its signal handler can
exit, and verifies exit plus release of the exclusive scheduler lock. It then
starts `scripts/upstream_primary_guarded.py`, recording the predecessor and new
PID in the existing launch record. It never changes the active prediction tree.

The successor imports the unchanged original scheduler and extends its holds to
every `tsvdx4p-*` job and the three delegated CFNADE presets. Dependencies of held
jobs are held too. The original ranks, epoch counts, recipes, receipt checks,
priority list and handling of partial outputs remain in place. These holds are
pending work, not reproduced model results. The mapped 1000-factor queue and
independent CFNADE workers remain responsible for their original source models;
larger factor models still need verified memory-compatible execution.

The handoff plan binds both new scripts, the original scheduler and the existing
import controller by SHA-256. Four focused tests cover all parallel rank holds,
external CF presets, actual idle-process termination, actual live-child refusal
and resumption, and wrong-PID rejection. No approval or extra laptop is needed.

Reports: `experiments/upstream-primary-handoff/`. A published launch result proves
only a scheduler transition; verify the actual successor process and native job.
This work makes no new model-score or qualifying-success claim.
