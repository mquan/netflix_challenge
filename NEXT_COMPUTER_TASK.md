# Current assignment: preserve the active upstream queue

**The second computer has already started the eight-job upstream replay on `research/upstream-reproduction`. Do not launch another worker or restart an active model.** Its progress is in `UPSTREAM_PROGRESS.md`; the first confirmed active job is `tsvdx4-60`.

The user authorized autonomous execution and branch coordination. The first Mac is adding compatible per-model artifact export to the existing `--compare-model` command. The already-running queue fetches updates at job boundaries and stages `experiments/upstream-repro`, so it will publish the fresh prediction artifacts along with its normal receipts. Earlier completed outputs are backfilled; no retraining is needed.

Keep the current queue running. Follow the current `COORDINATION.md` on `research/upstream-reproduction` for details. A missing `rbmx2-40` released base fixture is explicitly unavailable, not an experiment failure; portable reports use null. The old in-memory progress renderer may display `nan` for that absent reference, and its fresh score remains valid.

The first Mac finishes its assigned snapshot/calibration stages and monitors the training branch from **`research/upstream-verification`**. It independently verifies transferred vectors and runs the committed fresh-only blend plan after its existing work is complete and all eight vectors arrive.

If a real training or integrity error occurs, preserve all partial outputs/logs, publish the failure, and investigate. Do not delete outputs, duplicate training, or force-push. The completed original 10-stage ensemble experiment remains intact.

Training: https://github.com/mquan/netflix_challenge/blob/research/upstream-reproduction/UPSTREAM_PROGRESS.md

Local verification: https://github.com/mquan/netflix_challenge/blob/research/upstream-verification/COORDINATION_STATUS.md
