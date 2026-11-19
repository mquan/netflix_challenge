# Fixed six-model source extension

This is a separate development experiment after the verified 85-column voting blend (0.8604038556527669 ProbeX RMSE). It adds the six published source recipes already scheduled in the primary native DAG: `cfnade-12`, `attn-32`, `attn-32-t`, `aex-16`, `aex-64`, and `aex-128`. All six remain the main candidate; family-specific ridge screens are diagnostics, not a feature-subset selection procedure.

`upstream_source_extension.py --watch` is running and waits for all six complete, verified native two-phase receipts. Fast polling checks receipt identity and declared file existence. Before any array enters fitting, preparation verifies the full native receipt hashes, both ProbeX and qual shapes/types/finiteness, the pinned source, zero qualifying labels, known ProbeX row identities, the frozen parent cache, and its independent evaluation binding. Partial epoch exports cannot enter. The original 85-column production recipe and its watcher are unchanged.

Preparation runs the fixed five-user-fold ridge screen, retaining control/CFNADE/attention/autoencoder/all-six predictions and per-fold scores. Compensated summation independently checks every screen score. The parent ridge control must match before a result is accepted. The cache has 91 rating predictors, 167 ridge columns, and 227 nonlinear features; the original parent feature order, centering, final ridge intercept and all nested-fold configuration remain unchanged. No recovered qualifying outcomes enter this work.

The preparation watcher still prepares data only. A separate `upstream_extension_run_current.py` master is now queued. It first waits for the current 85-column attempt's frozen, independently verified qualifying audit. If both recovered groups already pass the strict target, it skips further work pending goal review. Otherwise it waits for all six fresh models and their prepared cache, obtains the exclusive development lock and memory headroom, fits tree and NN in separate processes with the unchanged nested user-fold configuration, replays every saved fold, verifies the fixed equal mean and independently scores the result. It uses qualifying results only to check whether the goal already passed, never to choose predictors, hyperparameters or fitting weights.

The fixed all-six development average must improve the strongest completed independently verified CF12 context or matching-tree fixed average before another qualifying fit is allowed. The context reference is currently 0.8600637147575518, replacing the older 0.8604038556 baseline. The controller checks evaluation/proof/vector hashes, independently scored RMSE, and identical label/user/fold files. It records the reference before fitting and rechecks it afterward so a concurrently completed tree study can strengthen the gate. Incomplete tree studies are ineligible. If the extension does not improve, the goal stays active and a later source-model extension remains necessary. Source-specific ridge bundles remain diagnostics and do not select a feature subset.

On 2026-11-19 the prior controller was verified idle with no children, no prepared extension cache and no fitting plan, then stopped before this controller change. Its source and launch metadata are preserved under `experiments/upstream-extension-run/before-current-best-gate*`; `gate-transition.json` records the transition. No native training process was signalled. The source preparation and qualifying queues retain their existing fitting locks and unchanged model recipes.

The initial change incorrectly edited the original controller, which is also
included in existing frozen development code hashes. A partial replay detected
that conflict. The original controller was restored byte-for-byte from its
archive; both context and matching-tree manifest validations pass again. The
strengthened gate now lives in `upstream_extension_run_current.py`, and
`upstream_extension_production_current.py` adds both new files to production
code bindings while reusing the unchanged production core. The second idle
controller was stopped without children before this correction. Actual launch
and correction evidence is in `frozen-controller-restoration.json`; the
original transition records remain historical evidence.

`upstream_extension_bank.py` appends the same six paired source vectors to the locally rebuilt paired 85-column bank, preserving all parent transforms and the final ridge intercept. The resulting matrices have167 design columns and227 nonlinear features (228 including the ridge input). Its source and prepared-file hashes bind both splits, all receipts and the parent bank. `upstream_extension_production.py` delegates known-ProbeX fitting, median nested iteration budgets, saved-model replay, fixed equal family average, frozen final hash and post-freeze recovered-label audit to the existing production core through explicit bank/report/cache/worker bindings. It mirrors actual production worker PIDs to the existing operational resource status path for the canonical scheduler.

New reports are under `experiments/upstream-{extension-run,extension-bank,extension-production}/`; new production artifacts are under `artifacts/upstream-extension-{bank,production}/` (the paired cache is specifically `artifacts/upstream-extension-bank/`, final models `artifacts/upstream-extension-production/`). The new master waits for the current attempt; it does not interrupt it. Partial fold/model artifacts are preserved on failure. The full upstream suite now has78 passing checks, including strict threshold/equality handling, audit/vector binding, false completion rejection, feature clipping/intercept preservation and production source binding.

The two canonical CFNADE-96 phases and same-trajectory raw-SGD counterpart are a further independent extension, not silently included in these six columns. They require completed canonical receipts and disclosed exact-optimization/checkpoint provenance.

Do not modify the queued extension script or its frozen parent code while the watcher is live. Reports: `experiments/upstream-source-extension/`; prepared data: `artifacts/upstream-source-extension/`. Missing native recipes are a verified wait, not completion of the 0.85725 qualifying target.
