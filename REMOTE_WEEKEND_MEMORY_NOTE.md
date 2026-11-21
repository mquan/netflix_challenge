# Weekend runner: resolve memory admission without restarting it

The execution branch's pinned checkpoint3e8ad78 at2026-11-21T00:45:25Z
reports that the original400pf run-job owner73979 is still waiting for
capacity. There is no native `fit` process in that checkpoint; its fallback
PID1 is macOS launchd, not a training job. The compressor sample is
12,642,205,696 bytes, about11.77GiB, above the frozen10GiB admission cap.
Disk217.7GiB and low research-worker RSS do not override that global gate.
This is an external checkpoint, not a current live remote-process poll.

Continue the existing weekend assignment on the remote laptop. Inspect the
same driver/run-job handles and current global memory before doing anything.
Identify which other processes own the compressed footprint; RSS alone can
miss it. `vm_stat`, `ps -axo pid,rss,command` and `vmmap -summary <pid>` provide
the relevant observations. Do not treat an old checkpoint or quiet log as
proof that the driver stopped.

Release only unused/disposable work within the existing authorization.
Preserve ongoing jobs, unsaved application work, the prepared original
source/data, every original output and the SAME waiting run-job owner and
its first swap baseline. Do not restart the driver or preparation, reset a
baseline, raise the compressor cap, change recipes or launch a duplicate.
When three original admission samples pass, the existing helper should
dispatch the full400pf producer through its original watchdog/bootstrap.

After admission, verify and push the actual native PID/start/state/elapsed
CPU and first epoch evidence. If the original conditions cannot be met,
report the actual memory owners and exact remaining gate while preserving
the same waiting process; do not claim training has started. Keep the
Monday06:00Pacific training and08:00Pacific return deadlines and all
original full-producer verification/export/upload requirements.
