# Preserve the live CPU fit while releasing the GPU epoch lock

The full-context tree supervisor held the shared source-extension epoch lock
throughout its CPU-only fit, leaving the raw MPS model waiting. Its actual
worker60801 owns the LightGBM fitting state and runs in an independent session.
A separate supervisor62158 verified the live parent/child identities, original
manifest and all prepared inputs, and checked that the worker had inherited no
controller lock descriptor. It then terminated only the supervising parent60506
and adopted the same running worker, without sending the worker any signal.
No fitting process, numerical input, thread count, RNG, model or budget changed.

The replacement owns the original model run lock and applies the same24GiB
combined-RSS/8GiB disk guards. It waits for actual worker termination and all
five original fold/model receipts, then runs the original actual saved-tree
query replay and independent score verification. Missing receipts or changed
bindings remain failures; a vanished process alone is not success.

The handoff receipt confirms unchanged worker60801 and original manifest SHA
173c0dc03b66f5e196279fbfc037369c52f35a016012be249d1f203675ba31f5.
The original raw controller58086 resumed actual trainx epoch3 after the shared
GPU lock was released. Its original per-epoch GPU locking,50/50 budgets, full
optimizer/RNG checkpoints and fixed snapshot/replay requirements are unchanged.
The queued epoch5/10 neural comparison still uses that shared GPU lock, so raw
and neural MPS fitting remain serialized while CPU tree fitting can overlap.

Evidence: experiments/upstream-item154-tree-supervisor/{plan,handoff}.json and
its live status.json. Code: scripts/upstream_item154_tree_supervisor.py.
This is a scheduling change, not a new accuracy result or official Test claim.
