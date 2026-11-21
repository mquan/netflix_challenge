# Read-only observer child retirement

The frozen native-observer producer can pause a worker process group while its
`ps` accounting subprocess is live. That subprocess also stops. macOS denies
physical-footprint inspection of this platform binary, so counting descendants
then prevents the original supervisor from collecting its readmission samples.
At the observed boundary, worker7939 and its read-only ps child9811 were both
stopped. Model state remains in the exact original worker.

This separately versioned companion changes no frozen producer source. It may
terminate only a stopped immediate child of the current exact source-bound,
paused dedicated worker, with command exactly
`ps -axo pid=,ppid=,stat=,rss=,lstart=,command=` and kernel executable `/bin/ps`.
It pins the OS binary, producer plan, owner, original baseline and its own
source/tests. Every event verifies the current worker dispatch gate and live
owner/watchdog, plus PID/start/PPID/PGID/command/executable identity twice.

The only signal is `os.kill(child_pid, SIGKILL)`, never a parent or process
group signal and never SIGCONT. Record an immutable intent before the signal
and terminal evidence afterward. Discard only the OS process-list attempt;
never claim that listing succeeded or drain its output. Termination is bounded
to2seconds; at most256events per companion and32per hour, with a5second delay.
Unexpected identities, source changes, ambiguities, rate limits or accounting
errors stop the companion and preserve the worker.

The existing owner alone must obtain its original3 valid samples and readmit
the same parent. The parent's existing WorkerGuard handles the failed ps call
as an accounting error, self-suspends and resamples on subsequent original
readmission. No scientific frame, model, optimizer, RNG, input hash, baseline,
budget or numerical output is discarded or restarted. No resource threshold is
relaxed. Actual data/MPS/model/paired-score proofs remain mandatory.

Tests include a private actual platform ps blocked on unread output exceeding
64KiB, child-only termination, wrong identity/executable rejection, bounded
termination failure and injected observer failure returning through the same
guard frame with its original baseline. No production processes are signalled
by tests. Actual companion actions require a clean source-bound review.
