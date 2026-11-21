# Native observer scope correction: integration pending

The first native observer candidate broadened the original resource scope to
all descendants. Actual operation exposed two problems not covered by its
initial tests: transient OS processes repeatedly disappear during observation,
and paused platform `ps` children cannot be inspected by either native counters
or vmmap. Fail-closed responses cause repeated same-worker suspensions.

The separately reviewed child-only reaper has retired one exact stopped OS
observer without signalling the parent or group. It prevents that particular
deadlock; it does not fix the repeated transient-process accounting errors.
Current worker7939 remains the same process and has made input-validation CPU
progress, with resource suspensions. No training or score completion is claimed.

The next candidate restores the EXACT original vmmap resource predicate:
required PID, repository path, target/release binary or upstream_/verify_upstream_
script. Every declared scientific worker remains counted; plain OS accounting
subprocesses were outside this original scope. Memory thresholds, physical/RSS
domains and original first-scientific baseline remain unchanged. Two native and
process-list identity samples and independent conservative counter maxima remain.
At most3 bounded metadata-observation attempts handle races; persistent errors
still fail closed. No numerical calculation or model fit is retried.

Four tests pass, and20 actual scans all include the live data worker with zero
failures and maximum0.07804seconds. Independent review is clean. Source and
evidence: `scripts/upstream_research_footprints_native_original_scope.py` and
`experiments/upstream-rbm400-original-scope-candidate/`.

No live controller imports the correction. A new explicitly versioned producer,
consumer and exact metadata-only retirement must be implemented and reviewed.
Retain both previous attempts, full closures/recipes and the byte-identical
ORIGINAL baseline; never fake an old completion. Before any retirement, freeze
and recheck exact live identities and that onlydata-started exists with empty
artifacts and idle consumer. If learned state has appeared, do not restart or
discard it. Include reaper14532 in terminality checks. Full inputs, actual MPS
stress,15+15epochs, saved-model replay and original own16 protocol remain required.
