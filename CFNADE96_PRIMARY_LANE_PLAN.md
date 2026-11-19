# Stable local CFNADE96 lane

The second CF96 admission raced the primary DAG's next native allocation.
Both phases and the new production fit passed a transient headroom check;
the next primary model then grew to a14.6GB footprint. Total owned footprint
reached26.9GiB, with16.3GiB in the compressor. Both checkpointed CF96 workers
were yielded; their original33/31 states and all84 tensors remain unchanged.

`scripts/upstream_cf96_primary_lane.py` now waits for a real primary-job idle
boundary. It requires the current wrapper to exit and its complete native
receipt to validate. It briefly freezes only the scheduler, checks again for
live children and rechecks the current job name while frozen. The original
scheduler writes the next name before Popen, so this closes the next-launch
race. If a child or changed job is found, the scheduler is resumed untouched.

Only a verified idle scheduler is terminated. Its original runner lock is held
by the lane while it takes two physical-memory admission samples and runs the
unchanged CF96 controller. Current native jobs are not stopped, no model source
is edited, and the320/96/95 preset, phases, seeding and checkpoint identities
remain exact. The source/data/owner fields and all84 tensors per phase are
rehashed before launch. The shared primary lock prevents another primary
controller from allocating a new model during CF96 training.

After CF96 exits, the lock is released and the unchanged primary continuation
script restarts from actual native receipts. The same cleanup/restart runs on
ordinary exceptions or termination, after stopping only the owned checkpointed
CF96 controller if necessary. A forced process kill is outside Python cleanup;
the recorded handoff and source commands remain available for recovery.

Two actual-process tests prove that busy owners are resumed with their child
still running, changed job conditions cannot trigger takeover, and an idle
owner can exit without affecting another process. Startup evidence verifies
the live lane, current primary/native processes and preserved states. The lane
is waiting; it has not yet taken over the primary scheduler or restarted CF96.
