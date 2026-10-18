# Honest-residual chains

Started 2026-10-18 on branch `research/honest-residual-chains` (worktree
`.claude/worktrees/honest-residual-chains`, from 4db5fc4). Runs in parallel with
the local rating-path branch (`research/local-context`, RESEARCH_RESET.md and
LOCAL_PATHS.md). This branch does not edit those documents.

Control: **R9-LAT-A2h, 0.8668706 development CV**. Target ≈0.8646, gap ≈0.0022.
All scores are development estimates, not test scores.

## Idea

The other branch adds new evidence for each query (cross-user rating paths).
This branch instead corrects the errors of the strongest existing models,
A2h (transformer) and B4 (Bayes), with second-stage learners trained on their
**out-of-sample** residuals. Every strong model × second-stage learner becomes a
new column under one fixed protocol, so it can be built on round after round.

Why:
- Our best single columns (B4 0.8897, A2h 0.8901 probe) trail Pragmatic Theory's
  best chained single predictor (≈0.879 probe-equivalent, GAP_AUDIT §0.1). Its
  chain is base → post-processing → residual kNN.
- Residual-kNN columns exist only on SGD factor models; none on A2h or B4.
- Honest residual corrections transfer: the HOLDOUT J3 pilot took J3 from
  0.8979 to 0.8934 from 1.4M P2 rows. It failed in the blend because J3 was weak.
- In-sample residuals are the known trap (BigChaos §3.3).

## Results so far

**HR-pre, user-level transfer (dead).** Per-user mean of out-of-fold P2 residuals
(ridge on the eleven honest rehearsal columns) used to shift R9-LAT-A2h probe
predictions: correlation 0.0105, RMSE −0.000047 (folds −.000042 −.000068
−.000020 −.000056 −.000048). User bias is already absorbed; second-stage learners
need item or neighbourhood structure, not per-user offsets.

**HR0, can masked A2h score its own training rows honestly? No.**

| A2h checkpoint on the same 1,407,566 P2 rows (masked) | P2 RMSE |
|---|---:|
| A2h-refit (trained on these ratings) | 0.83497 |
| A2h-select (never saw them) | 0.86740 |

Optimism 0.032, far above the 0.010 kill line. Masking the input does not stop
memorisation because each rating was a training target. Honest residuals for all
training rows would need K-fold A2h retrains (≈25 h MPS). Output:
`artifacts/residual-chains/HR0-A2h-refit.p2-insample.f32` (sha256 69f2dbe8…).

**HR1, do the other branch's path features add beyond honest P2 predictions?**
`scripts/hr_path_residual_test.py` reads LP2's saved `sample.npz` (read-only) and
reruns its exact ridge protocol (user folds 2–4 fit, 1 select, 0 evaluate)
with honest P2 columns added to the 27-feature control. 39,887 evaluation rows.

| Control | Control RMSE | + path features | Δ | 95% paired MSE interval |
|---|---:|---:|---:|---|
| LP2 pilot control (27 features) | 0.943222 | 0.937536 | −0.005686 | [−0.01310, −0.00829] |
| + A2h-select P2 column | 0.862041 | 0.862177 | **+0.000136** | [−0.00045, +0.00092] |
| + all 11 honest P2 columns | 0.845307 | 0.845633 | **+0.000326** | [+0.00003, +0.00107] |

A2h-select alone scores 0.864407 on these rows. Once A2h is in the control,
the linear path summaries add nothing. This is a linear test of summary features.
It does not test the planned structural encoder, which could still find
cross-link signal the histogram cannot express. Record: `experiments/HR1-path-features-vs-honest-bank.json`.

## Plan

| Stage | What | Gate |
|---|---|---|
| **HR2a** (running) | LightGBM on A2h-select's honest P2 residuals plus meta features (`train_only_residual.py`), applied to the A2h-refit probe column | Standalone A2h 0.8901 → ≤0.887, then blend |
| HR2b | Same with B4 (`A-B4-union.p2` → `B4-full-r128.s0160`) and with latent (K1 PCA) coordinates as inputs | same |
| Blend | Add each corrected column to the frozen R9-LAT-A2h bank: LightGBM-s3 + NN2 50/50 | ≥0.0003 and all five folds to continue; <0.0001 for both → stop P2-only path |
| HR3 | If HR2 pays: K-fold A2h retrains for honest residuals on all training rows, enabling residual kNN and residual factor models | — |

Probe labels never enter training. Settings are chosen on P2 user folds and
refit at fixed values, so the chain can be retrained in Stage C. One MPS process
at a time; CPU jobs niced while the other branch runs blends.

## Shared learning with the local-path branch (revisit at each checkpoint)

| When | Check |
|---|---|
| Now (done) | HR1: path summaries vs honest A2h on P2 → no gain. Tell the other branch before it scales the encoder; its comparison should include A2h-select's P2 column as a control input, not only pooled + kNN. |
| Their structural encoder result | Rerun HR1 with the encoder output in place of the histogram. If it adds beyond A2h, its output is a stage-2 input for HR2 residual learners. |
| Their residual variant | Point them at HR0: training-row predictions from a model trained on those rows are 0.032 optimistic for A2h. Use A2h-select P2 predictions, or HR3's K-fold residuals once they exist. |
| HR3 K-fold residuals exist | Offer them as residual targets on ~100M rows instead of 1.4M P2 rows. Coordinate MPS time first. |
| Either branch has a column | Score both on the same frozen R9-LAT-A2h bank, together and separately, to check whether gains add or overlap. |
| Any time | Their retrieval index (2.5 s build, ≈1,240 queries/s) is reusable for neighbourhood features in stage-2 learners. HR-pre says to skip per-user offsets. P2 is now selected on by both branches; promotion rests on the five probe folds. |
