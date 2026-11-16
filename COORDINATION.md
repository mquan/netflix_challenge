# Unattended two-computer assignment

The user authorized continued research and coordination through Git branches on November 16. Preserve all finished experiments, including negative results. No approval is needed to execute the bounded runs below or publish their progress to the designated branches.

## Branch ownership

- **`research/independent-ensemble`**: existing local experiment and completed second-computer experiment. Only handoff documentation is being added here. Existing frozen training scripts/inputs must stay unchanged.
- **`research/upstream-reproduction`**: preparation code, this coordination document, the fixed fresh-only blend plan, local experiment summaries, and independent verification reports. Owned by the first Mac.
- **`research/upstream-worker`**: fresh upstream training, eight compressed ProbeX prediction artifacts, execution receipts/logs, and stage progress. Owned by the second computer. Create this branch from `origin/research/upstream-reproduction`; the first Mac only fetches it.

No role force-pushes, rebases, merges, or automatically changes the other role's branch. Push rejection preserves all local results for inspection. Do not delete partial model outputs to force a retry.

## Second computer: start the worker

The previous 10-stage ensemble run is complete and found no gain from the combined features or ENN stacking. Its outputs must remain intact. Use a separate worktree from `~/workspace/netflix-independent`:

```sh
git fetch origin
git worktree add -b research/upstream-worker ../netflix-upstream-worker origin/research/upstream-reproduction
cd ../netflix-upstream-worker
python3 -m venv .venv
.venv/bin/python -m pip install numpy tomli
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
tmux new-session -d -s upstream-repro 'env UPSTREAM_REPRO_THREADS=8 .venv/bin/python -u scripts/upstream_exchange.py --worker --publish > upstream-worker.log 2>&1'
```

The worker automatically prepares the pinned upstream source, installs a checksum-verified official Rust installer into this experiment if Cargo is missing, builds the native tools, ingests the original archive with qualifying outcomes replaced by zeros, and verifies both development-reference implementations. It reuses the old clone's raw archive if present; otherwise upstream's downloader obtains it. Rust installation does not modify shell profiles. A missing platform build prerequisite is reported as a failed stage rather than silently changing the recipes.

It then trains these exact jobs sequentially, including their missing parents:

1. `tsvdx4-60`
2. `tsvdx4-60__knns`
3. `rbmx2-40`
4. `rbmx2-40__knn3`
5. `dnn-24`
6. `dnn-24__knn3`
7. `dnn-24__dnn-16p50`
8. `dnn-24__dnn-16p50__knn3`

Each completed job publishes its fresh ProbeX vector as a compressed `.npz` (roughly a few MB), a SHA256 manifest, the native execution receipt/log, and a fresh-vs-released score comparison. Only eight development vectors and known development labels/user indices cross machines. Full-training predictions, qualifying predictions, epochs and latent files stay on the worker. In total the development predictions are about 36 MB uncompressed. Git is the transfer mechanism so the first Mac can independently rescore real predictions, rather than trust a status summary.

The released final ensemble has 188 inputs. Two extra DNN parent fixtures are downloaded for the eight-job comparison. The pinned store has **no `rbmx2-40` base fixture**; report that comparison unavailable. Its fresh score and the fresh base-to-chain contrast remain valid development diagnostics. Do not substitute another model's reference.

On interruption, inspect the logs and rerun the **same worker command in the same branch/worktree**. Completed job receipts are verified and reused. Partial/unreceipted model outputs cause a stop and require inspection; preserve them. A role lock prevents duplicate workers. If Git rejects a push, keep the local commits and investigate instead of forcing it.

Progress:

```sh
cat UPSTREAM_WORKER_STATUS.md
tail -n 30 upstream-worker.log
ls artifacts/upstream-repro/fresh-receipts/
```

The second computer's agent should handle platform/setup failures within this assignment, retaining the pinned model sources and recipes. Push an explicit failure report if external help is required. Do not launch speculative parameter sweeps or the full 201-job graph.

## First Mac: completion and independent verification

The current six assigned baseline/snapshot/calibration stages finish first. The existing assignment monitor then ends the duplicate queue. A separate durable coordinator checks every two minutes:

```sh
.venv/bin/python -u scripts/upstream_exchange.py --coordinator --publish \
  --local-root /Users/quan/workspace/netflix_independent_ensemble
```

The coordinator owns `COORDINATION_STATUS.md` and `experiments/reproduction-coordination/`. It verifies and publishes the finished local experiment's six stage receipts and summary, fetches the worker branch, checks transferred artifact hashes and row ordering against the independently reproduced ProbeX labels, and recomputes each fresh model's score. Where a released fixture exists, it also measures the fresh/reference prediction difference. Receipts document remote execution but do not independently attest that execution; numerical rescoring is local and independent.

Once all eight fresh vectors arrive, it runs the predeclared comparisons automatically:

- Three base models versus all eight base/chain models.
- Temporal base versus its chain; RBM base versus its chain; DNN base versus all four DNN base/chain outputs.
- Five fixed user folds, seed `20261116`, ridge lambda `1000`, unpenalized intercept, input clip `[0,6]`, output clip `[1.05,4.95]`. No tuning or best-arm selection.

Only fresh vectors enter these fits. Released fixtures are used for numerical comparisons, never as blend features. Each scored user's labels are excluded from its fold's meta-model fit. These are still **development comparisons**, because the published base recipes were historically development-selected. They do not establish an untouched or official test score.

The fixed plan is committed before fresh results in `experiments/reproduction-coordination/meta-plan.json`. Local out-of-fold vectors remain under ignored `artifacts/upstream-coordinator/predictions`; their hashes, scores and paired fold deltas are published. The coordinator exits once both the local six-stage experiment and fresh-only evaluation are complete. While waiting, it remains active and writes an untracked heartbeat; it commits only when tracked results/status change.

No remote shell or active-agent notification channel is available here. A pushed handoff does not itself launch the second computer. Its session needs to fetch/read this assignment; the coordinator will truthfully report waiting until the worker branch appears.
