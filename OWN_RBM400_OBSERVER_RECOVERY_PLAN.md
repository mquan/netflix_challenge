# RBM400 input-verification observer recovery

Status: observer candidate tested; production recovery not yet implemented or
launched. Original producer remains in its data-only input admission phase.
No model training, MPS preflight success or qualifying improvement is claimed.

## Observed bottleneck

The original GPU400 owner31460/watch31465 dispatched worker74972. The worker
has only written `data-started.json`; the GPU400 artifact directory is empty.
At observation it had spent over27minutes wall time but only8.3seconds CPU.
Original input verification calls WorkerGuard for each1MiB hashed. The guard
uses a3second accounting cache whose timestamp precedes the resource scan.
The original observer invokes `vmmap -summary` for each research peer, taking
about0.7seconds per PID. A scan of more than20 peers exceeds that cache window,
so the next callback immediately repeats the scan.

Original files and plans remain unchanged. This is not evidence of slow model
training: training has not started.

## Candidate observer evidence

`scripts/upstream_research_footprints_native.py` uses Darwin
`proc_pid_rusage(pid, RUSAGE_INFO_V0, ...)`. The installed SDK defines a96byte
structure, with resident size at64, physical footprint at72 and start identity
at80. Counters are unsigned64bit bytes. Errors, disappearing processes,
changed peer sets or changed process identities fail closed.

The observer retains the original research scope and additionally counts its
descendants, even after exec changes their command. It excludes only its own
exact, already terminal `ps` subprocess. It samples physical footprint twice
and accounts RSS independently using the maximum of both kernel observations
and both process-list observations.

Nine CPU tests exercise counter domains, PID reuse, command changes, peer
appearance/disappearance, sampling errors, required worker absence, original
scope plus descendants and64bit values. Six actual comparisons match the
existing vmmap physical footprint within its printed rounding: private32MiB
fixture active/stopped and read-only observations of the data worker and old
paused tree. Only the private fixture received signals and it exited normally.
The full24peer native scan took0.078805seconds. Evidence:
`experiments/upstream-rbm400-native-observer-candidate/verification.json`.

No current controller imports this candidate. These observations validate the
resource observer candidate, not a GPU numerical kernel or trained model.

## Required production recovery

1. Create separately versioned execution and worker modules and new control,
   study and artifact namespaces. Retain the original attempt and all its
   metadata. Record this explicitly as restarting only input hashing/admission;
   no learned, optimizer or RNG state exists to discard.
2. Copy the exact original first-scientific baseline bytes and bind its digest.
   Never obtain a new swap baseline. Keep every original phase, full15+15
   epochs,500user MPS stress, dimensions, seed, input set, runtime, source
   checks, caps, reservations, lanes and stable readmission rules unchanged.
3. Pin the original source closure plus the new observer/controller/worker and
   recovery evidence. Compare scientific worker function bodies with the
   original; only the execution import may change. Preserve original numerical
   modules byte for byte. Keep live source files frozen.
4. Review concrete code, source/recipe equivalence, baseline inheritance and
   exact-process retirement before any production signal. Retire only this
   verified metadata-only worker/owner/watch. Confirm terminality and release
   of both original lanes before admitting the successor. Do not manufacture
   an old successful completion or alter old failure/history records.
5. The existing waiting local16 consumer pins old producer paths and proofs.
   Build a separately versioned consumer adapter/queue with the new producer
   provenance; retain all original16-family numerical and label-free gates.
   Never route new output through forged old plan/completion hashes.
6. Actual fresh MPS stress, both full fits, saved-model query replay and paired
   verification remain mandatory before the known-only calibration can run.
   Freeze all candidate vectors and complete independent preaudit before any
   recovered-label audit. Neither recovered group currently meets0.85725.

The independent reviewer accepted metadata-only supersession in principle,
subject to concrete implementation and review. No production shutdown or
successor launch is authorized by a successful observer unit test alone.
