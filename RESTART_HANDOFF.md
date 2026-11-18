# Current handoff: full fresh remote batch verified

Saved 2026-11-18T15:55:52.842432+00:00. Branch: `research/fresh-remote-batch`.

Best fresh-only ProbeX development RMSE: **0.8621943362**, fixed equal average of three-seed tree/neural corrections over 83 fresh prediction columns. Previous best 0.8636584760; matched gain 0.0014641399, all five folds.

All local fitting/auditing is complete. No owned training or sleep-prevention process remains. Read REMOTE_BATCH_PROGRESS.md, REMOTE_BATCH_PLAN.md and experiments/upstream-remote-batch/completion.json. Do not rerun completed queues.

## Completed checks

- All 25 exports from remote commit f130172 are accounted for: 24 unique, two previously integrated, nineteen primary additions, three separate intermediate-parent ridge sensitivity inputs.
- Sixteen pinned commands/native receipts and native.out hashes, all export hashes/content, all metadata alignment hashes and raw standalone scores passed. Eighteen available released fixtures were independently checked; eight exact, six near-exact, four reported Hogwild differences. Released arrays never became fit features.
- All 49 tests pass; both fitted families replay every prediction bit for bit, independent compensated-sum scoring agrees within 1e-12 and the fixed average is exact. The retained ridge vector is bit-identical to its old control.
- Preserve scripts/upstream_nonlinear.py, upstream_remote_integration.py and upstream_remote_batch.py and their manifests. Changing frozen source/configuration invalidates resume verification.

```sh
cd /Users/quan/workspace/netflix_upstream_repro
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
.venv/bin/python scripts/upstream_remote_batch.py --prepare
.venv/bin/python scripts/upstream_remote_batch.py --verify-family tree
.venv/bin/python scripts/upstream_remote_batch.py --verify-family nn
.venv/bin/python scripts/upstream_remote_batch.py --verify-scores
```

Prepare verifies existing caches without fitting. Models, ridge weights/scalers and prepared data remain in artifacts/upstream-remote-batch/. Track evidence/prediction archive under experiments/upstream-remote-batch/. Earlier 62-/64-column caches remain intact. The failed pretraining report-format audit is preserved separately; it never trained nonlinear models.

## Frozen controls and next work

- Fourteen ridge 0.8677119362283443; snapshot/context ridge 0.8663200739003126; 62-column ridge 0.866286461848646; 62-column average 0.8644717283393785; 64-column ridge 0.8655052917304086; 64-column average 0.8636584760350996; 83-column ridge 0.8637056490665563.
- All incoming work is evaluated. The three intermediate parents were tested only in the separately declared ridge arms; they improve the main ridge by about 0.000012, and were not silently added to the nonlinear primary.
- The original remote availability ended at 2026-11-18 15:01:05Z (training cutoff 13:01:05Z). No new remote training is assigned. Preserve its native model/fulltrain/qual output files; they are not all in Git.
- Larger temporal and small CF-NADE feasibility remains an option for a newly authorized compute window. Read REMOTE_TEMPORAL_PLAN.md on research/remote-temporal-push, but do not reuse its expired hard-coded deadline. The remote large-model RAM claim in its handoff was based on the smaller host; do not mistake lower bounds for actual native peak RAM.
- A qualifying target check needs an explicitly frozen production meta-model and verified fulltrain/qual inputs gathered from both hosts. Current exports are ProbeX only. Do not apply the original-Probe calibration offset to ProbeX or claim the 10% target from this CV score.

The user wants our own training outputs, not blending downloaded predictions. Source implementations remain pinned at 49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c and are attributed. No downloaded prediction features, pretrained weights, published epoch coefficients or qualifying outcomes enter these fits. Source recipes have historical development-selection exposure; this is not untouched/official Test scoring. Original-Probe R9 scores use a different split.
