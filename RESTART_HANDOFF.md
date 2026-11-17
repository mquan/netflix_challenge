# Restart handoff: independent upstream reproduction

Saved 2026-11-17T09:29:47.007474-08:00. Completed baseline/expansion and gap audit at commit `e3709327d67aaa98a022783208c9e1cb3b1a5051`.

## Resume here

- Working tree: `/Users/quan/workspace/netflix_upstream_repro`
- Branch: `research/upstream-verification`
- Main plan: [UPSTREAM_TARGET_PLAN.md](UPSTREAM_TARGET_PLAN.md)
- Snapshot/component bank and single-layer context blend are now complete: best fresh user-fold development RMSE **0.8663200739003126**. Read `SNAPSHOT_BLEND_PROGRESS.md` and `experiments/upstream-snapshot-blend/{manifest,evaluation}.json`. Next: diagnostic missing-predictor screening and full native dependency/cost resolution before selecting training bundles.
- No research jobs are running. Nothing needs to be resumed at the OS/process level. Do not rerun the completed training queues.

Suggested prompt for the next session:

> Read `/Users/quan/workspace/netflix_upstream_repro/RESTART_HANDOFF.md` and `UPSTREAM_TARGET_PLAN.md`. Continue diagnostic missing-predictor screening against the completed fresh snapshot/context control, then resolve promising native dependency bundles and costs. Preserve the verified fourteen-model control and publish progress to `research/upstream-verification`.

## What is complete

1. Eight fresh upstream outputs trained on the M3 Max, then independently checked on the M1. The fresh-only ridge improved from 0.876411386 for three bases to 0.869549365 with residual chains, on all five folds. Four DNN outputs matched released vectors exactly.
2. Six more outputs completed; all remaining work was transferred to the M1 because the other computer is busy. The fourteen-output bank scores **0.8677119362283443**, improving by **0.0018374287854254** over eight, in all five folds. All jobs and the automatic verification/evaluation have exited successfully.
3. The temporal neighbor mismatch was explained numerically by support provenance: including qualifying query identities (with all qualifying outcomes zero) reduced prediction RMS discrepancy from 0.017859 to 1.49e-6. Original fourteen-bank predictions remain unchanged; the diagnostic is a separate variant.
4. The historical RBM mismatch remains unresolved: released config targets `rbmx2-b40`, pinned recipe uses `rbmx2-40`; historical parent vector/config is unavailable. Do not assume the name discrepancy proves the complete cause.
5. Gap audit completed and controls reproduced. Same final-layer user-fold scores:

| Bank | RMSE |
| --- | ---: |
| Fresh fourteen | 0.867711936 |
| Released 137-column base bank | 0.860485311 |
| Released base + GBM outputs | 0.857614872 |
| Released base + MLP outputs | 0.857242909 |
| Released base + FWLS outputs | 0.856610757 |
| Full released 188-column stack | 0.856323241 |

The full released stack's fitted score remains **0.856167156**. We have NOT achieved this through our own training. Its final-layer fitting/CV difference is only about 0.000156. Most of the gap lies in the missing model bank and blending layers. Released intermediate columns are already learned; this audit is conditional final-layer CV, not end-to-end held-out evaluation. FWLS outputs can depend on GBM/MLP outputs, so trace full dependencies before estimating a reduced replay's cost.

## Completed since restart

The original fourteen control reproduced at 0.8677119362283469. All 45 saved epochs and five components were verified against native receipts; three exact terminal-epoch duplicates were removed. The combined 61-column fresh bank scores 0.8672877930850104. A single-layer context-weighted fit over this bank scores **0.8663200739003126**, improving in all five folds. Context uses trainx identities/dates and query identities/dates only, with fixed normalization and no outcome-derived transforms. Five context main effects and 70 interactions with the original fourteen predictions yield 137 design columns including the intercept. No additional base training or corrected-support substitution occurred. All 24 upstream tests pass. See `SNAPSHOT_BLEND_PROGRESS.md` for details and reproduction commands.

## Original experiment sequence

Steps 1–4 are complete through the first context regression. Optional GBM/MLP fits remain; step 5 is next.

1. Read the plan and audit, then reproduce the fresh-fourteen control before adding features.
2. Use the already saved 45 epoch predictions (20 temporal-60, 20 temporal-400, 5 DNN-64) and five component outputs. Freeze content hashes; remove exact duplicates. Preserve clipping rules: rating predictions clip to [0,6] at ridge input, but component features may need to remain unclipped.
3. Compare snapshots, components and their combination, refitting coefficients inside the existing outer user folds. Do not reuse published epoch-blend coefficients. Keep any corrected-support prediction as a separately declared arm, not a silent baseline replacement.
4. Fit a context-dependent feature-weighted regression using user/movie/time features derived from known training ratings and query metadata. No ProbeX or qualifying outcomes in feature generation; target-dependent transformations belong inside training folds. Evaluate with the fixed outer folds. Test GBM/MLP stack layers if warranted, with proper isolation across layers.
5. Screen released missing predictors diagnostically, resolve their native dependency bundles, rank gains against actual training/RAM cost, and retrain selected bundles. Avoid another untargeted capacity sweep or blindly launching the full 201-job graph.

## Locations and commands

```sh
cd /Users/quan/workspace/netflix_upstream_repro
git status --short
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
```

The current full implementation suite passes 24 tests. The newer read-only gap audit verified frozen input hashes and asserted both prior numerical controls.

- `scripts/upstream_repro.py`: native source/build/data wrapper and receipt verification.
- `scripts/upstream_exchange.py`: original fresh-only control, fold assignment and ridge cross-fitting. `crossfit` clips all inputs to [0,6]; **do not blindly reuse it for unbounded component features** without explicitly handling the clip mask.
- `scripts/upstream_expansion.py`: completed expansion and artifact collector. `collect_expansion()` revalidates the original eight and six new outputs and returns canonical labels, native user indices and fourteen fresh arrays.
- `scripts/upstream_gap_audit.py`: the completed reference-layer audit, with input-specific clipping.
- `experiments/upstream-expansion/evaluation.json`: final fourteen-bank results and out-of-fold hashes.
- `experiments/upstream-expansion/gap-audit.json`: layer-ablation measurements and limitations.
- `experiments/reproduction-coordination/fresh-only-blends.json`: original eight-bank result and frozen input artifact hashes.
- `experiments/upstream-expansion/m1/`: all six expansion prediction artifacts, native receipts, scores, and support diagnostic.
- `artifacts/upstream-coordinator/incoming/`: eight original fresh prediction artifacts received from M3, validated by frozen hashes.
- `artifacts/upstream-repro/fresh/preds_new/`: native locally trained predictions, epochs and components. Native arrays are in author ProbeX row order, 1,128,164 rows.
- `artifacts/upstream-repro/fresh/data/`: exact native trainx/probex/fulltrain/qual datasets. Qualifying outcomes are zero sentinels.
- `artifacts/upstream-repro/reference/`: downloaded development reference fixtures only; keep these out of fresh model features.
- `artifacts/upstream-repro/toolchain/`: isolated Rust/Cargo installation. `upstream_repro.environment()` restores its environment and Accelerate link flags.

Epoch filenames: `tsvdx4-60_ep*.probex.npy`, `tsvdx4-400lm_ep*.probex.npy`, `dnn-64_ep*.probex.npy`. Components: `tsvdx4-400lm-{bias,mf,nsvd1}.probex.npy`, `dnn-64-{base,net}.probex.npy`.

The `.venv` symlink points at the original project's venv; it is intentionally untracked. Ignored artifacts/toolchains/native training files stay on disk across a normal restart; they are not all backed up in Git. Avoid deleting the artifact trees.

## Constraints and coordination

- Preserve pinned upstream model sources at `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`; record any diagnostic helper separately.
- Use five fixed user folds, seed 20261116, ridge lambda 1000, unpenalized intercept and output clip [1.05,4.95] for control parity. Any change needs a separately recorded experiment.
- These are development comparisons on historically selected base recipes, not official/untouched test scores. Original local-bank scores around 0.8668 used a different Probe split and cannot be ranked directly against ProbeX.
- User authorizes autonomous work and progress commits/pushes. Commit with `mquannie@gmail.com`. No force-pushes or history rewrites for coordination.
- `research/upstream-verification` owns local results and next work. `research/upstream-reproduction` contains source replay/handoff material; the M3 assignment is revoked in `experiments/upstream-expansion/execution.json`. Do not launch the other computer's old worker.
- The original user cwd `/Users/quan/workspace/netflix_prize` has unrelated work; do not overwrite its files. Use this worktree.
- Branch-push conflict retry is allowed only when the remote commit is an ancestor of local HEAD. Never resolve divergence by force-pushing.
