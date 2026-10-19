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

**HR2a, honest-residual LightGBM on A2h.** `train_only_residual.py` fitted on
A2h-select's P2 residuals with meta features (952 rounds, chosen on P2 user
fold 0: 0.86684 → 0.86341), applied to the A2h-refit probe column:
**0.89014 → 0.88751** (−0.0026; folds .88885 .88778 .88343 .88845 .88904).
This misses the 0.887 standalone gate by 0.0005. Its inputs are features the blender already
sees, so its only possible edge is being fitted on 1.4M extra (P2) rows. The blend
test decides: `scripts/hr_blend_chain.py --name HR2a-blend` (frozen R9-LAT-A2h
config + this column; LightGBM-s3, NN2, 50/50; two processes at nice 10).
Status: `experiments/HR2a-blend.status.json`. Launched 2026-10-18 10:28 PDT.

**HR2a blend: not promoted.** Finished 12:11 PDT. R9-LAT-A2h + HR2a column:
**0.8668444 vs 0.8668706 (−0.0000262)**; folds +.000011 +.000144 −.000046
−.000085 −.000154. Two folds worse, and the gain is below the 0.0001 line. The extra 1.4M P2 rows do
not give the blender anything it lacks: a residual tree on the same meta
features learns what LightGBM/NN2 already learn on probe rows.
HR2b (same correction on B4) launched 16:51 PDT as the second arm of the stop rule.

**HR2b, honest-residual LightGBM on B4.** 1,228 rounds (P2 fold 0: 0.86387 →
0.85944). Applied to B4-full sweep 160: **0.88966 → 0.88485** (−0.0048; folds
.88588 .88552 .88096 .88617 .88575). This passes the standalone gate, and the
correction is nearly twice A2h's. The B4 chain's posterior mean has more
correctable structure on recent ratings than the transformer does. Blend test `HR2b-blend` launched
16:54 PDT (three processes, nice 10).

**HR2b blend: not promoted.** Finished 18:13 PDT. **0.8668187 vs 0.8668706
(−0.0000519)**; folds +.000080 +.000105 −.000134 −.000091 −.000220.

## Verdict on the P2-only residual route: stop

Both arms are below 0.0001 and both fail folds 0 and 1, the same two folds B5 failed.
A residual tree fitted on 1.4M P2 rows improves each base model on its own
(A2h −0.0026, B4 −0.0048), but the blender already has the same meta
features and learns the same corrections from probe rows. Extra rows from a
nearby distribution do not add information. K-fold A2h retrains (HR3) are **not**
justified by this evidence: the open question is new information, not more
honest rows for the same features.

Next on this branch: representation stacking. Feed the blender A2h's
internal 64-d query code at each probe row (plus B4 posterior factors), following
up R9-LAT's −0.00023 from 22 K1 latent features.

## Plan

| Stage | What | Gate |
|---|---|---|
| **HR2a** (done: −0.000026, not promoted; route stopped) | LightGBM on A2h-select's honest P2 residuals plus meta features (`train_only_residual.py`), applied to the A2h-refit probe column | Standalone A2h 0.8901 → ≤0.887, then blend |
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

## HR4: representation stacking (A2h)

`scripts/hr_a2h_codes.py` reads A2h-refit and exports 18 label-free features per
probe row (`artifacts/residual-chains/a2h-codes/`):
- The session-adjusted user vector (64-d), projected on the top 8 principal
  directions of the movies' liking vectors Σ(s−3)·output[i,s]. The basis is
  fitted on movie parameters only.
- The target movie's liking vector in the same basis.
- The session shift's norm, and its cosine with the transformer code.

The captured vectors reproduce A2h-refit's saved probe column exactly (max
error 0.0). This mirrors R9-LAT's K1 recipe (−0.00023, all folds).
Blend test `HR4-blend` = R9-LAT-A2h + these 18 support features, launched
2026-10-18 evening (three processes, nice 10). Same gates: ≥0.0003 and all five folds.

**HR4 blend: not promoted, worse.** Finished 21:55 PDT. **0.8669397 vs 0.8668706
(+0.0000690)**; folds +.000022 +.000135 +.000026 +.000134 +.000028, so all five
folds are worse. A2h's own column already carries what its internal code says about the
target rating. The 18 extra tree features only add split noise. K1's −0.00023 came from a
model whose column was weak, so its latents carried information its prediction
did not. A2h is the bank's strongest column, so that does not repeat. Representation
stacking of A2h is stopped. B4 posterior factors are not worth testing on this evidence
(B4 is also a strong column).

## Branch status after HR2a, HR2b and HR4

Every way of reusing the existing strong models has hit the same wall:
P2-trained residual corrections, per-user offsets, and internal codes. The blender
already extracts what A2h and B4 know. The remaining 0.0022 has to come from
**new evidence or a new model family** that is not a function of the current
columns. That is the other branch's bet (local paths/structural encoder), or a
new base model trained on different signal.
