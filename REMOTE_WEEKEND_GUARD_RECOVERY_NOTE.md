# Weekend resource-guard incident: preserve and recover the original processes

Pinned execution commit `ee38c3ec91e8160250d159c423e1581640c96700`
contains later guard/failure receipts for all three original producers. These
supersede its earlier 08:13 UTC progress snapshot. The change to parallel
execution is recorded there as an owner instruction; preserve that decision
without treating it as permission to weaken the unchanged resource guards.

| Producer | Original native PID | Original tree | Guard time UTC | Compressor GiB |
| --- | ---: | --- | --- | ---: |
| `s3-rbm400pf` | 24357 | `upstream-remote-weekend-r2/source` | 08:14:34 | 12.91 |
| `s3-rbm800p` | 68411 | `upstream-remote-weekend-r3/source` | 08:14:33 | 12.67 |
| `s3-tx300r` | 68498 | `upstream-remote-weekend-r4/source` | 08:14:39 | 13.07 |

Each failure receipt reports `actual_child_RAM_preserved: true`. That is
evidence of a guard stop, not evidence that numerical training exited or
that the processes are currently still present. Revalidate on the remote
host. No complete producer, export, archive or upload is established.

The compressor exceeded the frozen 12 GiB continuous ceiling while the
parallel native processes were starting. Reported swap did not grow. This
establishes the triggering guard; it does not isolate the system-wide
compressed-memory ownership or prove which allocation caused the increase.
The normal status files were not updated after the failures and must not be
used alone to claim ongoing training.

## Instruction for the remote agent

Continue in the existing execution checkout. Fetch the coordinator branch
and read this note without switching or resetting a live/dirty checkout.
Preserve all original outputs, source trees, logs, receipts and three native
processes. Do not rerun `prepare` or `run-job`, launch replacement models,
send a bare `SIGCONT`, reset swap baselines or raise resource caps.

1. Capture current `ps` evidence for all three native PIDs, original owners
   73979/67961/67991 and watchdogs 24356/68410/68497. Match the original
   PID, start time, process group and actual native command to the original
   ownership/launch receipts. Confirm stopped RAM and identify all live
   owners before any handoff. If a handle is absent or differs, preserve
   that evidence and do not claim same-process recovery.
2. Inspect global compressor/swap and actual memory owners. Release only
   unused/disposable work already authorized; preserve unsaved application
   work and every training artifact. Keep the original per-job swap
   baselines and the 10 GiB admission / 12 GiB continuous compressor caps,
   64 GiB per-job / 80 GiB research memory limits and original disk floors.
3. Implement and test an isolated same-process adoption owner, using the
   existing coordinator recovery controllers as patterns. Do not edit the
   frozen original training helper or scientific recipes. Bind original
   source/data/binary/preparation/launch/controller hashes and actual stopped
   identity. Require the previous owner and watchdog to be terminal before
   adoption. The new owner must journal its own identity and start a verified
   watchdog BEFORE continuing the exact preserved native process. Keep
   resource admission stable for three samples, preserve RAM on guard or
   owner loss, and never automatically restart or repeatedly continue a
   stopped process. Review the reservation arithmetic for already allocated
   stopped RAM; do not omit pending allocation or silently relax admission.
4. Recover the original 400pf process first to preserve its five completed
   first-phase epochs. Additional original processes may run concurrently
   only when their reviewed admission and all unchanged continuous guards
   permit it. Otherwise preserve their RAM and report the exact gate.
5. Retain the Monday 06:00 Pacific training deadline and 08:00 Pacific
   return deadline. A recovered orphan has no ordinary parent wait status:
   record that limitation honestly. Require both complete original epoch
   traces and every original output, then independent full export/array
   verification before declaring a complete eligible producer. Any compatible
   completion receipt must explicitly link the new adoption evidence and
   original launch; never invent a successful exit code or rewrite failures.
6. Push a new small execution checkpoint promptly with current actual
   process states, the triggering guard, original baseline bindings, and
   either reviewed same-process continuation evidence or its precise
   remaining gate. Keep original partial evidence and no-retry receipts.

Coordinator evidence:
`experiments/upstream-remote-weekend/coordinator-parallel-guard-observation.json`.
The coordinator has no direct process-control session on this laptop;
remote live validation and recovery must execute there.
