# Active two-computer research assignment

The user authorized continued execution and coordination through Git branches on November 16. Both computers already have active work. Preserve those runs and all negative results; do not launch duplicate training.

## Branch ownership and current runs

- **`research/independent-ensemble`**: the first Mac finishes its six assigned baseline/snapshot/calibration stages. Its existing assignment monitor ends the duplicate remaining queue afterward. The second computer's earlier 10-stage experiment is complete; combined features and ENN stacking did not improve its 0.8668672 baseline.
- **`research/upstream-reproduction`**: the second computer is already running `scripts/upstream_repro_queue.py` inside tmux, with RAYON 8 / BLAS 1, starting with `tsvdx4-60`. It owns `UPSTREAM_PROGRESS.md`, native execution receipts, comparisons and exported fresh vectors. The pinned source is `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`.
- **`research/upstream-verification`**: the first Mac's durable coordinator owns `COORDINATION_STATUS.md` and `experiments/reproduction-coordination/`. It fetches the training branch, independently checks fresh predictions, and runs the fixed fresh-only blend experiment when ready.

The initial preparation and compatibility changes are merged into the training branch once. Subsequent local verification reports go only to the verification branch, avoiding competing progress writers. Do not force-push, rewrite the other computer's history, or change its active model sources/configuration.

## Second computer: continue the existing queue

**Do not start another worker or interrupt a model to adopt these changes.** The live queue runs `git pull --rebase` as it publishes progress between jobs. It will pick up the updated `scripts/upstream_repro.py` and `scripts/upstream_exchange.py` at a normal job boundary.

The comparison subprocess now exports verified fresh predictions under `experiments/upstream-repro/fresh-artifacts/`. The existing queue already stages this directory's parent, so its regular commits publish the vectors automatically. Earlier completed jobs are backfilled from their original receipts; no retraining is necessary. The exporter does not modify model sources, hyperparameters, data, saved predictions, or the active queue process.

Each completed model exports a compressed ProbeX vector, SHA256 manifest, native receipt/log and a canonical fresh-vs-released comparison. Only the eight development vectors and known development labels/user indices cross machines: about 36 MB of predictions uncompressed. Full-training predictions, qualifying predictions, epochs and latent files remain on the training computer. Receipts document remote execution; they are not remote attestation. Numerical rescoring happens independently on the first Mac.

The eight jobs remain:

```text
tsvdx4-60
tsvdx4-60__knns
rbmx2-40
rbmx2-40__knn3
dnn-24
dnn-24__dnn-16p50
dnn-24__dnn-16p50__knn3
dnn-24__knn3
```

The 188-column reference is unchanged. Two extra DNN parent fixtures are available for comparison and are fetched when needed. The pinned store has **no `rbmx2-40` base fixture**. Its fresh score and fresh base-to-chain comparison are still useful. The canonical transfer manifest records the absent reference as `null`, with `reference_available=false`. For compatibility with the old renderer already loaded in the active Python process, its legacy comparison JSON may use numeric NaN display sentinels; the progress table will show `nan` for those unavailable reference fields. This does not represent a measured reference score. Portable manifests normalize those sentinels to null.

On a real failure, preserve all partial outputs and logs. Publish the failure and investigate; do not delete files or force a retry. The existing receipt mechanism resumes completed jobs. This bounded assignment does not authorize a speculative parameter sweep or the full 201-job pipeline.

## First Mac: durable completion and verification

From `research/upstream-verification`, the active coordinator runs:

```sh
.venv/bin/python -u scripts/upstream_exchange.py --coordinator --publish \
  --local-root /Users/quan/workspace/netflix_independent_ensemble
```

It checks every two minutes, writes an untracked heartbeat, and commits only changed status/results. It first verifies the local experiment's frozen input/code hashes and six completed stage receipts, then publishes that experiment's summary. It fetches the training branch and verifies transferred artifact hashes, dtype/shape, native output receipt, and exact development label ordering against the independently reproduced reference. It recomputes fresh scores and fresh/reference prediction differences where a fixture exists. Active remote training is distinguished from waiting for its first artifact push.

Once the existing local run is complete **and** all eight verified fresh vectors have arrived, it automatically runs the committed plan:

- Three base models versus all eight base/chain models.
- Temporal base versus its chain; RBM base versus its chain; DNN base versus all four DNN outputs.
- Five fixed folds grouped by native upstream user index, seed `20261116`, ridge lambda `1000`, unpenalized intercept, input clip `[0,6]`, output clip `[1.05,4.95]`.
- No tuning, best-arm selection, downloaded prediction features, or qualifying outcomes.

Each scored user's labels are excluded from its fold's meta-model fit. These remain **development comparisons**, because the published base recipes were historically development-selected. They are not untouched or official test scores. The 0.856167 released-prediction reference remains distinct from fresh-training results.

The plan is frozen in `experiments/reproduction-coordination/meta-plan.json`. Local out-of-fold vectors stay under ignored `artifacts/upstream-coordinator/predictions`; their hashes, scores and paired fold deltas are published. The coordinator stops after both the existing local experiment and fresh-only evaluation complete. Integrity errors are published and preserve the artifacts.

No remote shell or direct active-agent notification channel is available here. Git branch pushes coordinate the existing queues; the training queue's normal job-boundary fetch picks up the handoff.
