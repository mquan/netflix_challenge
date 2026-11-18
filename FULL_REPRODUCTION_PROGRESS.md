# Full published-pipeline reproduction: active

Goal: verified recovered qualifying RMSE below 0.85725 using freshly trained outputs. No other laptop is assumed available.

Current state: **training**; current native job: `mf-60`.
Completed native jobs on this machine: 13. Registry: 201 jobs / 188 final input columns.

The full exact source DAG is the work list. Completed remote ProbeX exports do not stand in for missing native fulltrain/qual or parent-training artifacts. Resource-limited jobs remain in the inventory for the memory-compatible backend; no smaller factor count is substituted.

State: `experiments/full-upstream-reproduction/status.json`. Inventory and unresolved historical aliases: `inventory.json`. Logs/receipts: `artifacts/upstream-repro/fresh-receipts/`.

The best existing fresh-only development checkpoint is 0.8621943362. It does not establish the qualifying target. The goal remains active until production predictions are frozen and both recovered groups independently pass.
