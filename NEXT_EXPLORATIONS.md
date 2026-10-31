# Next explorations beyond SO1 and N1

Written 2026-10-30 on `codex/local-context`. This is a plan, not a score claim.
The promoted reference is still R9-LAT-A2h at **0.8668706425** dev CV, against
a calibrated target of **0.86465** (gap 0.0022206425; see
[PLAN_TO_TARGET.md](PLAN_TO_TARGET.md)). SO1 (auxiliary choice prediction,
[SELECTION_OBJECTIVE_RESEARCH.md](SELECTION_OBJECTIVE_RESEARCH.md)) and N1
(choice-history input tokens) belong to other sessions. Nothing here edits
their plans, runners or artifacts, or competes with them for the GPU.

## Where the remaining error is

- Round 11's error analysis of the R9-LAT-A2h blend found that 63% of probe
  rows fall on a day with no training rating from that user. Those rows carry
  67% of the squared error (RMSE 0.893), and error grows with the time since
  the user's last training day. No per-bin recalibration is left (<0.00003).
- HR0–HR4 (`HONEST_RESIDUALS.md` in the `honest-residual-chains` worktree) showed that the
  blender already extracts what A2h and B4 know. Residual trees, code
  features and path summaries add ≤0.0001.
- The test offset (−0.0074, measured twice) comes entirely from retraining on
  train+probe. That retrain is the one step that sees the user's most recent,
  qualifying-like behaviour.

So the gap needs either new evidence about the future, or a better way to
absorb probe-like data in the final retrain. The bets below are ranked by how
directly they target that.

## E1. Make the final retrain (Stage C) use probe data better (CPU, start now)

**Hypothesis.** Probe rows are each user's most recent ratings, from the same
regime as qualifying. Plain union training gives them the same weight as
ratings from years earlier. If the Stage C retrain weights them more, it may
widen the retraining gain beyond the measured 0.0074.

**Harness.** This uses [PHASE0_REHEARSAL.md](PHASE0_REHEARSAL.md) unchanged.
P2 (training ratings dated nearest each user's probe dates) stands in for the
probe, and the original probe stands in for qualifying. Number **b** (the
blender fit on P2, applied to columns retrained with P2) is the Stage C
analogue. The treatment changes only how P2 rows enter that retrain:

| Arm | Retrain on T + P2 |
|---|---|
| b0 (control) | unchanged recipe; must reproduce the recorded b columns |
| b-w2 | P2 rows get loss weight 2 |
| b-w3 | P2 rows get loss weight 3 |
| b-ft | b0, then a short fine-tune on P2 plus same-user recent rows at reduced learning rate (fixed epoch count chosen before scoring) |

Every arm keeps the P2-selected epochs and the blender fitted in **a**. The
original probe scores **b** only; it selects nothing. Weights {2, 3} and the
fine-tune length are fixed before running, and there is no sweep after reading
the probe.

**Implementation note.** `src/joint.hpp` (K1) has no per-row weights today.
Duplicating P2 rows in the training view would also double-count them in
implicit feedback and support counts, so it is not a valid shortcut. Add an
explicit loss weight that leaves implicit sets and counts unchanged. Add a test
that weight 1 reproduces b0 byte for byte. Start with K1 and its two kNN
columns (CPU), then the conditional RBM chain.

**Gate.** At least **0.0005 extra retraining gain** (b0 − b-w) on the
9-column core-bank blend, with the direction holding on all five user folds of
the original probe. Pass → repeat on the Bayes columns, then A2h (GPU, after
SO1). Fail → close E1; the plain union retrain stands.

**Caveat.** P2 is less "future" than the real probe (PHASE0 already notes this),
so the gain may be conservative or biased. Any result is an estimate of the
hidden-test effect, not a verified test score.

### E1 stage 1: K1 family (fixed before any score, 2026-10-30)

- **Code.** `netflix joint --row-weights FILE` (float32, one weight per
  training row) scales each row's error in every SGD update. Implicit sets,
  support counts, the reported train RMSE and the model fingerprint are
  unchanged. `tests/test_joint_row_weights.py` covers these: all-ones weights
  reproduce the unweighted model and predictions byte for byte (one thread),
  non-unit weights change the fit, `eval-joint` accepts the weighted model,
  and short, long, negative, NaN, >16 and missing files fail. The existing
  10 `test_joint.py` tests pass on the same binary. The binary is built
  separately as `build/netflix-rowweights`; the shared `build/netflix` is
  left alone while other sessions' frozen runs may call it.
- **Weights.** `scripts/stage_c_row_weights.py` marks the 1,407,566 P2 rows
  in `data/processed/netflix.bin`. Each must match exactly one training row on
  (user, item, day, rating), and the other 97,664,546 rows must equal the
  holdout's training rows in order. Files are `artifacts/rehearsal/e1/p2-w{1,2,3}.f32`
  with hash reports.
- **Runs.** `scripts/e1_chain.sh build/netflix-rowweights w2 w3 w1` repeats
  P0-B-K1 (256 factors, lr .008, reg .02, decay .97, 41 epochs fixed,
  5 threads) and both P0-B-kNN commands, adding only `--row-weights`. w1 is
  run last as a same-binary control; with 5 Hogwild threads it is not
  bit-identical to the recorded B-K1 run, so the w1 − B-K1 difference
  measures run-to-run noise. Each arm takes about 67 minutes.
- **Score.** Per arm, copy `artifacts/rehearsal/spec-core.json` and swap only
  the three b columns (K1, kNN-256td + its support, kNN-64) for that arm's.
  Run `scripts/rehearsal_blend.py` with its default recipe (seed 2026, `--skip-cv`),
  and only when SO1 is not about to start an arm. The fit and a columns are
  untouched, so the blender is still fit on P2 alone.
- **Decision.** Compare b(w2) and b(w3) with b(w1). Pass: one of them beats w1 by
  ≥0.0005, the gain exceeds |b(w1) − b(B-K1)|, and all five user folds of the
  original probe improve. The better of w2/w3 is reported; no other weight
  is tried after reading these. Pass → extend to the RBM chain, then Bayes.
  Fail → close E1. The fine-tune arm (b-ft) is deferred until stage 1 passes.

### Fixed E1 scoring implementation, 2026-10-30

The root-owned scoring step is `scripts/e1_fixed_score.py`, with the immutable
`experiments/E1-fixed-scoring-plan-20261030.json`. It fits the unchanged
rehearsal blender once on P2, then scores original/w1/w2/w3 b banks with that
same fitted predictor. Only the three K1-family predictions and kNN support
change. The original b score must reproduce within 1e-6; all nine training
records must be successful and match the declared commands and binary hash.
It reports the predeclared 0.0005/noise/five-fold gate and user-cluster paired
uncertainty. No dev-CV c run or new hyperparameter selection is introduced.

`scripts/e1_fixed_score_wait.sh` is a shell-only waiter for those nine records.
It launches the one-thread scorer only while the verified N1 training PID is
still running the expected command. Status is
`experiments/E1-fixed-scoring-20261030.state`; results go to
`artifacts/rehearsal/e1/fixed-scoring-20261030/comparison.json`. Do not launch a
duplicate scorer. Four focused tests cover unchanged P2/a banks, undeclared
arm rejection, all-fold gating, run-noise rejection and the minimum gain.

An additional integration fixture exercises the actual data loader, metadata,
contextual ridge and LightGBM on 100 users / 200 queries. Changing evaluation
labels changes the score but leaves all four arms' prediction bytes identical;
only one P2 fit is performed per evaluation. A failed producer receipt rejects
before output creation or fitting. Evidence is
`experiments/E1-fixed-scoring-integration-20261030.json`; the frozen real-data
scorer and plan remain unchanged. These are fixture values, not Netflix gains.

**Execution checkpoint, 2026-10-31 05:04 UTC.** All nine K1/kNN training
receipts for w1/w2/w3 report return code zero. The shell waiter launched the
unchanged one-thread scorer after verifying N1's expected training process
was live. PID 42412 was confirmed running the declared `score` command. This
is the fixed four-bank comparison, not a new parameter search; no gain or
gate outcome is claimed before `comparison.json` is complete. Memory pressure
was checked before the transition (26% system-wide free memory).

**Independent control verification.** On all 1,408,395 original-probe rows,
the saved float32 original-bank predictions score 0.875952449707 versus the
historical 0.875952449804 (difference -9.62e-11). The new same-binary w1 bank
scores 0.875952731319, a noise delta of +0.000000281612. Prediction hashes,
data/plan bindings, finiteness and all five fold scores are recorded in
`experiments/E1-fixed-control-verification-20261030.json`. No weighted-arm
gate result or promoted gain is claimed by this control-only check.

### E1 result: closed (2026-10-31)

`artifacts/rehearsal/e1/fixed-scoring-20261030/comparison.json`: the blender
is fit once on P2 and scores every b bank. The original bank reproduces
0.8759524498 exactly.

| Bank | b RMSE | Δ vs w1 | Folds better |
|---|---:|---:|---:|
| original (recorded B-K1) | 0.8759524498 | −0.0000003 | — |
| w1 (same binary, unweighted) | 0.8759527312 | — | — |
| w2 | 0.8760935147 | **+0.0001408** | 0 / 5 |
| w3 | 0.8764472883 | **+0.0004946** | 0 / 5 |

Run-to-run noise is 3e-7. Up-weighting P2 rows makes the Stage C bank
*worse*, monotonically in the weight, on every fold. The paired user-cluster
95% intervals on ΔMSE exclude zero for both arms. Standalone, no epoch of the
w2 K1 run beats the unweighted final (best 0.89504 vs 0.89422), so a
different epoch could not rescue it, and that bound needs no probe selection.

The pre-registered early stop applied. The Bayes w1 control was never launched
(`E1-bayes-control-queue-20261030.state.json`: `skipped_no_k1_gain`). The
Bayes w2 arm (this session's job) was stopped at sweep 15 of 160, before any
kept sample, and its receipt records return code 143. No Bayes score exists or
is claimed. The `--row-weights` code stays, tested, for any future use.

**Reading.** Plain union training already absorbs the probe-like rows about as
well as this model family can. Extra weight overfits those 1.4M rows instead
of transferring recency. The Stage C offset is not a lever. The gap has to come from
new modelling signal, which agrees with round 12's conclusion.

### E1 stage 1 observation during the w2 run (not a gate result)

The K1 trainer logs standalone original-probe RMSE every epoch. Compared with
the recorded unweighted P0-B-K1 run, w2 is better early (−0.0055 at epoch 1,
−0.0008 at 21) but crosses over at about epoch 27 and is +0.0010 at epoch 36.
P2 rows, at double weight, appear to be overfit under the fixed 41-epoch
schedule that was selected unweighted. This does not change the gate. If
stage 1 fails, the follow-up would have to select the weighted arm's epoch on
a training-only inner split (a P2 carved from T − P2), never on the original
probe. Choosing epoch ~25 from this log would be probe selection.

### E1-Bayes: the same weighting in Bayes timeSVD++ (fixed before any score, 2026-10-30)

- **Code.** `netflix bayes --row-weights FILE` gives row p noise precision
  alpha·w_p. Every Gibbs sufficient statistic is weighted (row-group A/B,
  block row mass s0 and s1, s2, se, sqe, and the w0 draw). The noise-precision
  Gamma uses the weighted SSE in its rate and the row count in its shape.
  Residuals and the reported train RMSE are unweighted.
  `tests/test_bayes_row_weights.py` covers these: all-ones weights reproduce
  the unweighted chain byte for byte (one thread), residuals stay exact under
  weights (1 and 3 threads), rows weighted 8 are fitted at least 20% closer,
  and malformed files (including zero weights) fail. The 5 existing
  `test_bayes.py` tests pass on the same binary, `build/netflix-bayesweights`.
- **Runs.** `scripts/e1_bayes_chain.sh build/netflix-bayesweights w2`, then
  `… w1` as the same-binary noise control once the K1 chain frees CPU. Each
  repeats the B4-full-r128 command exactly (rank 128, 160 sweeps, burn-in 16,
  seed 13, item-bin 2, drift 2, 4 threads), plus `--row-weights`. About 10 hours each.
  Only weight 2 is tested on Bayes; it was fixed before any K1 result.
- **Score.** `artifacts/rehearsal/r12/spec-core-B4.json` with only the b B4
  column swapped for the arm's final posterior mean (160 sweeps, as the
  recorded s0160 column). Same blend recipe as stage 1.
- **Decision.** Same gate as stage 1: w2 beats w1 by ≥0.0005 b-RMSE, the gain
  exceeds |b(w1) − b(recorded B4)|, and all five folds improve.
- **Early stop.** If K1 stage 1 shows no gain at all (both b(w2) and
  b(w3) ≥ b(w1)), stop the Bayes arms to free CPU and record that they were stopped.

**Control queue, 2026-10-30.** The previously planned w1 control now has an
actual shell-only waiter: `scripts/e1_bayes_control_wait.sh`, bound by
`experiments/E1-bayes-control-queue-20261030.plan.json`. It waits for the fixed
K1 comparison and released K1 CPU threads, then launches the unchanged
four-thread w1 command only if at least one K1 treatment improves w1 at all,
matching the existing early-stop condition. This is not a relaxed promotion
gate; Bayes still needs its own 0.0005/noise/five-fold comparison. The queue
skips an existing control, rejects changed inputs or mismatched score-plan
evidence, and never stops another session's jobs. If K1 shows no gain it
records that no control was launched; the existing w2 job is left to its owner.
Four shell fixture tests cover launch-on-some-gain, no-gain skip, mismatched
plan rejection, existing-output preservation and duplicate-queue rejection.
Live status is `experiments/E1-bayes-control-queue-20261030.state.json`.

## E2. Movie-side context inside A2 (CF-UIcA-style) — GPU, after SO1

**Hypothesis.** A2h encodes only the user's own history. Let the query token
cross-attend to a sampled set of the target movie's other raters (their frozen
A2h user codes, ratings and rating dates). That adds a learned, query-time,
two-sided context, which is MOONSHOTS bet 2. It is still unmeasured.

**Why the earlier negatives don't settle it.** The A1 item twin was a separate
item-only model. Round 10's kNN neighbour features were fixed blender inputs,
not a learned context. Neither tests joint conditioning.

**Recipe.** Warm-start from the A2h-select checkpoint. Sample 64 co-raters
uniformly from the movie's training raters, using the current A2h codes kept
frozen for the pilot. Always exclude the target (user, movie) rating from the
movie context at training and inference. Matched control: same warm start,
schedule and seed, with the cross-attention block present but its context
masked.

**Gate.** At least **0.0005** in the R9-LAT-A2h blend (round4_boost three
seeds + NN2 50/50), all folds better. Main risks: target leakage through
co-rater codes trained on the target row, and sampling bias toward popular
raters. Both need tests before the run.

## E3. Ordinal loss on A2h's existing distribution head — GPU, after SO1

**Hypothesis, corrected by source audit 2026-10-30.** A2h already has a five-way
distribution head: its saved selection report says `head=distribution` and
`loss=mse`. `SequenceModel.decode` emits five logits; the inherited `expected`
method takes the softmax-weighted star expectation, and `loss_value` minimizes
MSE on that expectation. The proposed experiment is therefore a loss change,
not a new head. Cross-entropy and an ordinal term could constrain aspects of
the distribution that expected-rating MSE does not; whether that improves
RMSE or ensemble diversity remains unmeasured on A2h.

The existing `ordinal` loss is cumulative binary log loss, not the squared
ranked-probability score. Fix the exact loss formula, coefficients and matched
P2 gate before running this bet. Keep the expected rating as the prediction.
Distribution support features are a separate ablation: earlier R8-A3-dist
features regressed, so they must not be bundled into an alleged head gain.

**Recipe.** Two-epoch warm-start fine-tune from the A2h-select checkpoint,
reusing SO1's matched-arm harness: same seed, masks, batch order and epochs.
Control is MSE-only. Implement in a separate trainer after the GPU queue is
finished; do not edit the frozen SO1 trainer or its dependencies.

**Gate.** At least **0.0003** in the blend, all folds better.

## E4. Second A2h seed (filler)

The A2b twin added 0.0003 and later A2 variants added 0.0001–0.0004. A second
A2h seed probably adds 0.0001–0.0002. Run it only if the GPU would otherwise
sit idle. It does not close the gap on its own.

## Not worth reopening

The record already closes each of these: Bayes session/day variants
(round 11), B5 item-frequency and kNN neighbour features (round 10),
ENNBlend / third blender (round 8), LC6 snapshot/epoch diversity, linear path
summaries (HR1), A2 code features in the blender (HR4), more MT/RBM seeds
(rounds 6–7), and a bigger transformer (A2c).

## Order

1. E1 now, on CPU. Its retrains are C++ (`netflix`), so they leave the GPU
   free. But SO1's queue waits for two process snapshots with **no other
   Python executable**, so the Python blend step (`scripts/rehearsal_blend.py`)
   would delay SO1's start. Run blends only while SO1 is not about to launch.
   Use `nice` and few threads, and check `ps` before each launch.
2. After SO1's gate: if it fails, E2 takes the GPU. If it passes, run E3 first,
   since it reuses the same harness and objective-change setup.
3. E4 only as idle-GPU filler.
