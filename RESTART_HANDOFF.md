# Current handoff: fresh-only remote integration verified

Saved 2026-11-18T04:24:51.083822+00:00. Local branch: `research/fresh-remote-integration`.

Best fresh-only ProbeX development RMSE: **0.8636584760**, fixed equal average of three-seed tree and neural corrections over 64 fresh prediction columns. Previous best 0.8644717283; matched gain 0.0008132523, all five folds.

All local work is complete. No local training or sleep-prevention process remains. Do not restart completed queues. Read `REMOTE_INTEGRATION_PROGRESS.md`, `REMOTE_INTEGRATION_PLAN.md`, and `experiments/upstream-remote-integration/completion.json`.

## Verification and commands

- 43 checks passed. Both model families replay their saved predictions bit for bit. Independent compensated-sum scoring, fixed-average verification, metadata alignment, source receipt bindings and matched fold/cluster comparisons passed.
- Source implementation pin remains `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. Qualifying outcomes remain zero. Base recipes have historical development-selection exposure; these are not untouched/official qualifying results.
- Keep `scripts/upstream_nonlinear.py` and its original manifests unchanged: the new adapter hashes that engine and its own code. Changing either invalidates frozen-run resume/verification.
- Existing frozen controls: fourteen 0.8677119362283443; snapshot/context 0.8663200739003126; latest pre-remote ridge 0.866286461848646; pre-remote nonlinear average 0.8644717283393785; expanded ridge 0.8655052917304086.

```sh
cd /Users/quan/workspace/netflix_upstream_repro
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
.venv/bin/python scripts/upstream_remote_integration.py --prepare
.venv/bin/python scripts/upstream_remote_integration.py --verify-family tree
.venv/bin/python scripts/upstream_remote_integration.py --verify-family nn
.venv/bin/python scripts/upstream_remote_integration.py --verify-scores
```

The prepare command verifies existing immutable caches and does not retrain. New cached arrays/models and float32 predictions are under ignored `artifacts/upstream-remote-integration/`. Compressed predictions and metadata receipts/logs are tracked under `experiments/upstream-remote-integration/`. The original 62-column cached experiment remains at `artifacts/upstream-nonlinear/`. Keep both.

## Remote continuation

- Larger assignment: `research/remote-temporal-push` / `REMOTE_TEMPORAL_PLAN.md` (latest plan commit e3daec1). Start with tsvdx5-150o, then assess larger temporal models and compatible chains, with cfnade-12/bknbrx-50 fallbacks. Unknown runtimes require measurement; do not change pinned capacities to force fits.
- That plan uses the original window: no new full recipe after 2026-11-18 12:31:05Z, hard training cutoff 13:01:05Z, final handoff 15:01:05Z. Do not assume another 12 hours without the user extending it.
- Do not launch old expansion queues, the completed remote12h queue, or canonical cfnade-96 (roughly 50 hours projected). NLPP paropt reads all ProbeX labels and is excluded pending a separately designed nested recipe; published epoch coefficients are also excluded.
- Verify new receipt-bound transfers and metadata before adding them. Refit inside scoring-user exclusion through every learned layer. Compare each new bundle and the predefined combination against the current 0.8636584760 average, reporting all fold changes and negative outcomes.

The user wants independently trained results, not another blend of downloaded public predictions. Upstream source reuse is disclosed; no released prediction features or pretrained weights enter the independent fits. Original-Probe R9 scores around 0.8668 use a different split and cannot be ranked directly against ProbeX. No production/qualifying target claim follows from the development score.
