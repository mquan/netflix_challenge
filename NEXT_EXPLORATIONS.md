# Next explorations beyond SO1 and N1

Written 2026-10-30 on `codex/local-context`. This is a plan, not a score claim.
The promoted reference is still R9-LAT-A2h at **0.8668706425** dev CV, against
a calibrated target of **0.86465** (gap 0.0022206425; see
[PLAN_TO_TARGET.md](PLAN_TO_TARGET.md)). SO1 (auxiliary choice prediction,
[SELECTION_OBJECTIVE_RESEARCH.md](SELECTION_OBJECTIVE_RESEARCH.md)) and N1
(choice-history input tokens) belong to other sessions. Nothing here edits
their plans, runners or artifacts, or competes with them for the GPU.

## Shared experiment ledger and P2 command

| Bet | Fixed decision | Current evidence/state |
|---|---|---|
| E1 Stage C weighting | >=0.0005, run-noise check, all five folds | closed: w2/w3 worse on every fold |
| N1 choice-history input | original N1 P2 gate, then full-bank all-fold comparison | full result 0.86670063: 0.000170 overall gain, fold 1 worse; promotion and substantial-continuation gates fail |
| SO1 auxiliary objective | frozen SO1 plan/gates | closed: P2 gain 0.0000275 < 0.0005; gate independently reproduced; queue terminal |
| E2 two-sided context | original E2 blend gate; recipe not yet frozen | CPU components verified; deferred after headroom review |
| E3 CE loss | >=0.0003 under registered matched protocol | code ready, GPU deferred |
| E5 ordinal Bayes | >=0.0003 P2 matched-pair gain | closed: +0.000135 gain, below threshold; shared gate independently reproduced |
| A7 sibling pooling | >=0.0005 after Stage C, all five folds | failed: +0.001779 worse on all folds |
| H8 choice affinity | P2 gate then >=0.0003 full-bank matched-control gain, all folds | full result 0.86648484; gain 0.00022028 vs matched control, 0.00038580 vs R9, all folds better; fixed full-bank minimum fails; not promoted |
| H9 choice co-occurrence SVD | >=0.0003 nested P2 gain beyond H8 | closed: 0.000048 gain, interval includes zero |

`scripts/p2_candidate_gate.py --plan PLAN --output RESULT` is the shared
scoring adapter for completed matched P2 columns. It directly reuses
`selection_objective_gate.compare`: bank+control+duplicate-control versus
bank+control+treatment, penalties {1,10,100,1000} selected on user fold 1,
fit on folds 2–4 then 1–4, evaluate fold 0. It adds no model or penalty sweep.
The result distinguishes this P2 fold-0 decision from full-bank promotion.
Any experiment's additional all-fold requirements still apply separately.
Existing frozen runners are unchanged and are not rewired automatically.

A plan declares `experiment`, `protocol` = `SO1-P2-fold0-matched-pair-v1`,
`minimum_gain`, `protocol_reference`, `data`, `bank`, `frozen` hashes and
`arms.control`/`arms.treatment`, each with `prediction` and `receipt` paths.
Bind the data, bank, protocol reference and five scorer source files named
by the module's `SOURCES` constant. The protocol reference is the experiment's
already registered recipe/gate; creating a scoring manifest after training
does not authorize changing that protocol.

Each owner validates its trainer-specific recipe and writes a completion
receipt containing `state: complete`, `smoke_only: false`, `experiment`,
`arm`, `data_sha256`, `prediction` and `prediction_sha256`. The adapter checks
these fields and hashes, plus finite arrays and row shapes, before calling
the comparator, then rehashes inputs before writing a new output. A receipt
is an owner assertion, not a substitute for trainer-specific provenance audit.
No running or smoke output may be scored as a completed candidate.

Three fixture tests verify exact comparator parity, held-label independence
of fitted predictions/penalties, and rejection of incomplete receipts,
changed payloads/frozen inputs and output overwrite. No real experiment was
scored by this new adapter during its implementation.

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
w2 K1 run beats the unweighted final (best 0.89504 vs 0.89422). This offers no
standalone evidence for an epoch rescue; it does not measure the blend value
of untested epochs. The fixed recipe is closed without selecting another epoch.

Independent verification rehashed all 73 bound inputs/artifacts and rescored
all four saved float32 prediction files across every row and user fold. Both
treatments still worsen every fold; the independent deltas agree with the
original report within 1e-7. The archived comparison and verification are
`experiments/E1-fixed-scoring-comparison-20261030.json` and
`experiments/E1-fixed-scoring-verification-20261030.json`.

The pre-registered early stop applied. The Bayes w1 control was never launched
(`E1-bayes-control-queue-20261030.state.json`: `skipped_no_k1_gain`). The
Bayes w2 arm (this session's job) was stopped at sweep 15 of 160, before any
kept sample, and its receipt records return code 143. No Bayes score exists or
is claimed. The `--row-weights` code stays, tested, for any future use.

**Reading.** The tested weights 2/3 under the fixed 41-epoch K1 recipe fail to
increase the measured Stage C benefit. This closes the declared weighting
experiment and its conditional scale-up; it does not establish an upper bound
on every retraining objective or on a completed weighted Bayes model. The
measured offset and estimated target gap remain unchanged. New modelling
signal remains the next planned direction.

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

### E2 CPU data preparation (2026-10-30; no model launched)

`scripts/e2_corater_data.py` adds a persisted training-only movie-to-rater
index and uniform sampling of up to 64 distinct other raters. The query user
is excluded before sampling, including when the query is a masked training
row. Selection depends only on seed and user/movie/date identifiers; ratings,
date proximity, overlap and popularity scores never determine sampling.
Only training-row ratings/dates are exported. An explicit mask distinguishes
padding from valid dense user ID zero. Duplicate user/movie training ratings
are rejected because exclusion assumes one rating per pair.

Five focused tests cover query-label independence with nonempty context,
exact target-user exclusion, distinct/uniform sampling, empty contexts, valid
user zero, duplicate rejection, payload/dataset changes and no overwrite.
The fixed data-only plan is `experiments/E2-data-plan-20261030.json`: build
indexes for A2h's hash-matched P2 and full-training datasets, then sample
10,000 label-independent queries per view to measure coverage and throughput.
All work is one-thread CPU while N1's training process is live; no GPU job,
model training, P2 rating score or original-probe score is part of this step.

This prepares a component of E2, not the full experiment. The exact frozen
co-rater code must still be defined: A2h's normal codes are query-dependent,
so a generic user-code cache cannot be assumed equivalent. Freeze the code
producer, matched architecture, training budget and pilot gate before any
learning run, and preserve the existing GPU queue's priority.

**Measured data feasibility.** Both indexes are complete: P2 has 97,664,546
training rows and the full view has 99,072,112. Every query in each fixed
10,000-row sample has other-rater context. Mean sample sizes are 63.981 and
63.9861 respectively; sampled retrieval/packing rates are about 17,401 and
10,970 queries/second, excluding index loading/hashing and any neural work.
Independent verification maps every sampled source row back through the user
CSR and confirms zero owner leaks, duplicate sampled raters, source-value
mismatches or held-out query targets present in training across 20,000 queries
and 1,279,671 context edges. All seven frozen inputs and all index/sample
payload hashes were rechecked. The four view manifests are archived under
`experiments/E2-{P2,full}-{index,sample}-20261030.json`; the independent record
is `experiments/E2-data-verification-20261030.json`.

This establishes the sampler's feasibility and exclusion contracts on those
samples. It does not establish an E2 rating gain, neural throughput, a complete
co-rater code cache, or readiness to launch the deferred GPU experiment.

### E2 exact-code feasibility audit (prepared before code generation)

`scripts/e2_query_codes.py` measures actual query-conditioned A2h codes, rather
than treating an unvalidated generic USER-slot readout as equivalent. Each code
is for one observed co-rater/movie/day pair. Its own rating value is masked
from all explicit sequence, histogram and session channels; permitted implicit
identities/dates remain. The captured session-adjusted vector reconstructs the
frozen five-logit decoder within 1e-6. The original checkpoint is read-only.

The data-only plan is `experiments/E2-query-code-plan-20261030.json`: select up
to three sampled source pairs in each of five history-count strata, with seed
20261031, then encode those pairs on one CPU thread. Inference uses padding to
one row to bound CPU workload; fixture outputs match padding 64 within 1e-6.
This is not an MPS equivalence or throughput claim and does not alter any
queued trainer. Three tests cover held-rating/query-label exclusion, decoder
reconstruction, frozen parameters, visible-rating sensitivity and padding.
The audit does not train a model or choose any rating hyperparameter. Its
purpose is to measure the exact code producer before proposing a full cache
or an explicitly different representation; no generic-cache approximation
may be reported as an exact-query-code test.

**Audit outcome.** All 15 actual P2 co-rater pairs were encoded successfully,
covering training histories of 9--2,446 ratings and widths up to 1,536. Captured
vectors reconstruct the decoder exactly (maximum logit error zero). Changing
each held rating and all P2 query labels leaves every input tensor and saved
code byte-identical; a real short-history padding-64 comparison also agrees
exactly. Parameters remain unchanged. Four fixture tests pass, including a
sole-rating user whose visible explicit context becomes empty.

The three longest-history CPU forward/check timings are 26.7--43.6 ms each,
plus 2.2--2.8 ms input preparation. These are bounded one-thread CPU timings,
not MPS estimates. The plan, exact audit and independent verification are
`experiments/E2-query-code-{plan,audit,verification}-20261030.json`.

The 10,000-query P2 sample requests 635,196 distinct source-row codes. One
64-float32 vector per code is about 155 MiB for that sample; the all-training-row
upper bound is 23.29 GiB for P2 and 23.62 GiB for the full view. This is not a
measured required full cache. Cardinality/storage arithmetic is recorded in
`experiments/E2-cache-size-audit-20261030.json`. Exact code production is now
defined and exercised, but whole-cache generation, MPS throughput, matched
neural architecture and the final learning plan remain pending. No E2 rating
score or model improvement has been measured.

### E2 batched producer and joint-context model prototype

`scripts/e2_joint_context.py` adds batched exact peer-code production, including
independently masked rows for repeated global users. It also defines an A2h
subclass whose rating-query code attends to frozen peer vectors, observed
peer stars (an eight-dimensional embedding), and three fixed date-gap features.
The head uses A2h's rank/head count and dropout settings. A zero-initialized
output projection preserves the warm rating prediction before optimization.
Empty peer sets produce exactly zero correction, with a safe attention mask.

The matched control executes the same shapes and dropout calls but zeros peer
code/rating/date values while retaining peer masks. Frozen cache inputs are
detached. Context attachment rejects a target user in its own peer set and
missing/nonfinite cache values; it never reads target rating labels. Four CPU
tests cover separate/batched code parity, repeated-user exclusion semantics,
warm prediction parity, empty contexts, matched training RNG, nonzero learning
gradients, cache immutability, sampler-to-model integration and rejection paths.

`experiments/E2-joint-plan-20261030.json` freezes a bounded real-data forward
audit before execution: batches of four peer codes, parity with the existing
15 single-code records, then the first query of the frozen P2 sample and all
its up-to-64 sampled peers. This has no real-data optimization, rating score,
or GPU launch. This prototype is not the complete E2 training/cache runner;
the full learning recipe and whole-cache generation remain pending.

**Prototype verification.** Four CPU fixture tests pass. Three optimizer steps
on fixture targets produce nonzero peer-head gradients and distinct control/
treatment predictions while consuming identical CPU dropout RNG; frozen cache
inputs receive no gradients. These are fixture learning checks, not Netflix
scores. On the real P2 data, batched vectors match all 15 previously audited
single-query vectors within 4.77e-7, and decoder reconstruction remains exact.
The first frozen sample query (row 25) was forwarded with all 64 sampled peers;
the zero-initialized head preserves the original A2h logits exactly. Changing
every P2 query label changes no attached input tensor. Both the frozen producer
and new model parameters remain unchanged in this inference audit.

The head adds 21,616 parameters at A2h's rank 64. Evidence is
`experiments/E2-joint-audit-20261030.json`. This verifies the CPU model/data
boundary; it does not provide a complete training runner, whole-cache coverage,
MPS memory/throughput evidence, a selected training budget or an E2 RMSE gain.

### E2 resumable code cache (CPU verification only)

`scripts/e2_code_cache.py` now exports the exact producer into a memory-mapped
float32 table, either for all training source rows or an explicitly bound,
sorted sparse row list. Every completed chunk is flushed, hashed and recorded
atomically. Resume verifies all committed chunks and rewrites any uncommitted
tail. Plans bind the dataset, selected checkpoint/report, producer sources,
row list, inference settings and Torch version. The read-only loader requires
a complete, matching cache, verifies payload integrity and rejects uncached
rows; missing rows never become zero context.

Four CPU tests cover interrupted output, resumed/direct code parity, changed
plans and row maps, committed payload corruption, dense/sparse lookup and the
complete plan-to-export path on a fixture. The fixed real audit plan is
`experiments/E2-cache-audit-plan-20261030.json`. Its first call committed eight
of the existing 15 audited source rows; the loader rejected that partial cache.
The resumed call completed all 15, matching their prior single-query vectors
within 4.77e-7. Evidence: `experiments/E2-cache-audit-20261030.json`.

No training or rating score was computed. A whole-cache run and its MPS
throughput/memory feasibility remain unmeasured. The matched E2 learning
runner, fixed training budget and pilot gate still need implementation and
freezing before a real learning launch. This exporter does not own scheduling;
GPU execution remains deferred behind the existing queue.

### E2 matched P2 training runner (fixture verified, no real launch)

`scripts/e2_train.py` accepts a frozen plan and an explicit control/treatment
arm. It requires a complete cache from the exact warm backbone and training
context, verifies the sampler index, and binds its sources and input manifests.
It reuses A2h's existing mask/sequence preparation unchanged; target dates are
recovered by local user/movie identity, without consuming RNG or target stars.
The two arms use the same optimizer, masks, batch ordering, sampled peers and
dropout calls, differing only in the prototype's peer-value switch.

The runner trains the plan's fixed epoch count, then emits one final prediction
file and checkpoint. It neither scores P2 nor selects an epoch. A batch-capped
run is marked smoke-only. Three CPU fixture tests verify original batch/RNG
parity, matched control/treatment RNG consumption, distinct arm predictions,
exact repeated-treatment model/prediction reproducibility, label-independent
inference, query-order parity and early rejection of changed/missing inputs.
All three pass; no real Netflix optimization was run.

This implements the selection-context runner, not a frozen learning experiment.
The actual budget, shared gate binding, whole-cache feasibility, GPU scheduling
and any full-data follow-up remain pending. In accordance with STRATEGY.md,
E2's next learning decision waits for the headroom map and the existing GPU
queue; this preparation does not launch a new one-off bet.

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

### E3 fixed protocol and implementation (2026-10-31, before any score)

- **Loss.** L = MSE(expected rating) + **λ·CE**(five logits, observed star),
  with **λ = 0.1** fixed. One arm only, no λ search. The prediction stays the
  softmax-weighted expectation. No distribution support features.
- **Trainer.** `scripts/e3_distribution_loss.py` calls SO1's
  `selection_objective.run` unchanged, with auxiliary weight 0, and swaps only
  the rating loss inside its own process. It records `ce_weight` and its own
  SHA-256 in the report's `arguments`. No frozen SO1 file is edited.
- **Matched control = SO1's control arm.** `tests/test_e3_distribution_loss.py`
  shows λ=0 reproduces SO1's weight-0 run exactly on a CPU fixture: identical
  epoch-2 prediction bytes and every model tensor. So `SO1-control` (A2h-select
  warm start, P2 dataset, seed 20261019, lr 0.0003, 2 epochs) is the control,
  and E3 runs that command with `--ce-weight 0.1` and a new output prefix. If
  SO1's control is ever rerun or changed, rerun both arms. The tests also
  cover λ>0 changing the fit, the exact combined-loss value, restoring the
  original loss function, and rejecting a nonzero auxiliary weight and a
  negative or NaN λ.
- **Score.** Epoch 2 fixed. Same scoring as SO1's primary comparison: bank11 +
  control + duplicate control versus bank11 + control + E3, ridge folds as in
  `SO1-plan.json`. The E3 gate plan will bind `e3_distribution_loss.py`'s hash
  in addition to SO1's frozen sources.
- **Gate.** ≥0.0003 P2 RMSE gain, all five folds. Pass → full-T pair from the
  A2h-refit checkpoint, as SO1's full stage, then the full-bank comparison
  (promotion requires all five dev folds).
- **Launch condition.** Only after the other session's GPU queue (SO1,
  including any full-data stage) is finished, the GPU is idle, and no other
  Python training process is live. Never in parallel with their jobs.

### E3 five-fold gate binding (2026-10-31, before E3 training or scores)

The shared P2 command measures only outer fold 0; its result cannot prove
E3's registered all-five-fold requirement. `scripts/e3_all_fold_gate.py`
adds that check without changing the shared scorer or SO1's frozen files.
For each outer fold f, select the ridge penalty from {1,10,100,1000} on
(f+1) modulo 5 using the other three folds, then refit on all four non-held
folds and predict f. The matched designs remain bank11+control+control
versus bank11+control+E3. Fold 0 exactly matches the shared gate.

E3 proceeds only if its original shared fold-0 gate passes **and** this
additional cross-fitted P2 comparison gains at least 0.0003 overall with all
five folds improving. This is fixed before any E3 score; no choice between
the two results is allowed. Bind this module's hash in E3's future scoring
plan, and audit trainer completion before calling it. Three fixture tests
verify exact shared-fold parity, outer-label exclusion for each fold, and
rejection of a duplicate candidate or an incomplete fold set. No real E3
training or scoring has run.

## E5. Ordinal (interval-censored) Bayes timeSVD++ — CPU (fixed 2026-10-31, before any score)

**Hypothesis.** The one CPU family that kept paying was the conditional RBM,
which models the rating *distribution*. The Bayes model, the strongest CPU
learner, uses a Gaussian likelihood on star values. Treat a rating r as a
rounded latent: z = f + N(0, 1/α) lies in (r−½, r+½], open-ended at 1 and 5.
Gibbs data augmentation (Albert–Chib) resamples z from its truncated normal
every sweep, and the unchanged factorization-machine Gibbs updates then run
on z. Prediction is the posterior mean of E[r] = 1 + Σ_t P(z > t),
t ∈ {1.5, 2.5, 3.5, 4.5}. Ratings at 1 and 5 no longer pull f toward the
boundary, and the predicted mean is calibrated near the ends. That gives a
different error structure from every Gaussian column in the bank.

**Code.** `netflix bayes --likelihood ordinal` (default `gaussian`), in the
separate `build/netflix-ordinal`.
- A single-thread Gaussian run is byte-identical to the previous binary,
  and ordinal mode draws RNG only in its own latent step.
- `netflix check-truncated-normal` checks the sampler against analytic means:
  central, one-sided and far-tail intervals, plus quantile round trips and the
  expectation identities.
- `tests/test_bayes_ordinal.py` covers: the self-check; explicit `gaussian`
  equal to the default; exact residuals under ordinal (1 and 3 threads);
  ordinal learning on rank-2 data, within 0.05 of Gaussian, predictions in
  [1, 5]; and rejection of an unknown likelihood and of ordinal resume.
- The joint and Bayes suites (plain and row-weight) pass on the same binary.

**Pilot.** `scripts/e5_chain.sh build/netflix-ordinal {gaussian,ordinal}`,
two parallel processes with 4 threads each. Both use the B2 recipe (rank 64,
128 sweeps, burn-in 16, seed 2026) on the P2 holdout
`artifacts/validation/train-matched-20260927.bin`, and predict P2. Output goes
to `artifacts/e5/`, logs to `experiments/E5-p2-r64-*`. About 4 hours.

**Gate (P2 only; the original probe is not read).** Use SO1's
`selection_objective_gate.compare` on SO1's bank11 P2 columns, which already
include B4 and A2h. Fit ridge on user folds 2–4, select the penalty on fold 1
from {1, 10, 100, 1000}, refit on 1–4 and evaluate on fold 0. The primary
comparison is bank11 + Gaussian control + duplicate control versus bank11 +
Gaussian control + ordinal. **Pass: ≥0.0003 P2 RMSE gain.** Standalone P2
RMSE of both arms is reported, not gated. Pass → one full-probe ordinal run at
the B4 recipe (rank 128, 160 sweeps, item-bin 2, drift 2, seed 13), added to
the promoted R9-LAT-A2h bank under the standard dev-CV recipe: ≥0.0005 and
all five folds to promote. Fail → close E5 without a rank or recipe sweep.

### E5 result: closed (2026-10-31)

`artifacts/e5/E5-gate.json` (`scripts/e5_gate.py`, SO1's `compare` unchanged, P2 only).

| | P2 RMSE / Δ |
|---|---:|
| Gaussian control, standalone (rank 64, B2 recipe) | 0.868034 |
| Ordinal, standalone | 0.867065 (**−0.00097**) |
| **Primary:** bank11 + control + ordinal vs + duplicate control | **−0.000135**, 95% ΔMSE interval [−0.00035, −0.00011] |
| Replacement diagnostic: bank11 + ordinal vs bank11 + control | −0.000087 |

Gate (≤ −0.0003) **failed**. The interval-censored likelihood is a better
standalone Bayes model, but its information is almost entirely already in
bank11 (B4 and A2h). Closed without a rank or recipe sweep, as pre-registered.
`--likelihood ordinal` stays in the code, tested. This is consistent with the
headroom map: model-quality changes inside existing families no longer carry
new information.

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

## A7 sibling Stage C rehearsal — independent fixed-gate result

The unchanged owner scorer (`scripts/a7_score.py`, protocol in HEADROOM.md)
was run after all five A7v2 steps returned zero, on the independently verified
chronological dataset. The pre-score input binding and result are archived as
`experiments/A7v2-independent-score-{plan,result,evidence}-20261031.json`.
All bound scoring inputs were rehashed unchanged after scoring.

K1-family half-B RMSE: dev 0.8912278762; dev+pool 0.8889590137;
Stage C 0.8860581428; Stage C+pool 0.8878368913. The fixed deployable
pooling rule **worsens Stage C by 0.0017787484**, with every user fold worse
(+0.00150 to +0.00199). Its user-cluster 95% delta-MSE interval is entirely
positive [0.0028491, 0.0034615]. The predeclared >=0.0005/all-fold extension
gate fails; this recipe does not advance to Bayes/RBM on these results.

The B-label-refitted diagnostic gains only 0.0000638 and has a mixed fold
direction; it is not a deployable selection or an alternative passing arm.
Retraining alone gains 0.0051697334 in this half-sibling/K1-family rehearsal.
That is a family-specific, half-labelled-pool measurement, not an automatic
replacement for the twice-measured full-bank offset of approximately 0.0074.
No full-bank dev-CV or qualifying prediction was promoted, and this negative
does not rule out every possible interaction or calibration recipe.

### E5 shared-gate reproduction

The generic shared adapter reproduces the owner's `E5-gate.json` primary
statistics exactly: 0.8477520529 -> 0.8476166706 on P2 fold 0, a gain of
0.0001353823, below the registered 0.0003 threshold. Both commands, binary and
Bayes source hashes, dataset, 128 terminal sweeps/112 kept samples, final
prediction shapes/finiteness and standalone logged RMSE were validated before
writing completion receipts. Raw blend input files were preserved; standalone
score validation uses the trainer's existing [1,5] clipping convention.

Evidence: `experiments/E5-shared-gate-{result,evidence}-20261031.json` and
`E5-shared-{control,treatment}-receipt-20261031.json`. The first scoring attempt
correctly rejected a changed live NEXT_EXPLORATIONS.md. Plan v2 replaces that
reference with its **byte-identical original content** recovered from git
6c97938; all gate parameters, data, bank, model outputs and source hashes are
unchanged. Both plans remain archived. Future scoring plans should bind an
immutable protocol snapshot rather than a shared working document.

## H8 independent nested validation (fixed before its real score)

Claimed by Codex after the original H8 diagnostic completed. The original
H8 sources/output remain unchanged. `scripts/choice_outer_validation.py`
reuses the original twelve choice features, windows, 32-dimensional
projection, seed, tree parameters, 2,000-round cap and 100-round stopping
patience, with two CPU threads. Plan: `experiments/H8-nested-plan-20261031.json`.

The change is proper inner/outer stacking. Round selection uses training
folds 2–4 with internally out-of-fold ridge bases, and fold 1 with a base fit
only on 2–4. After selecting rounds for treatment and matched base-only tree,
final trees fit on 1–4 using ridge bases cross-fitted within 1–4. Fold 0 gets
its ridge prediction from 1–4 and is evaluated once; its labels never enter
any base fit, residual training target or iteration selection. Ridge penalty
remains 10. This is an independent nested check with final refit, not a
one-variable numerical reproduction of the original H8 pipeline.

Primary decision remains the H8 >=0.0003 gain versus the raw ridge. The
matched base-only-tree comparison is also reported. There is no feature,
window, seed or parameter sweep after this score, no original-probe score,
and no automatic full-bank promotion. Three fixture tests verify complete
fold-0 label invariance (predictions and selected rounds), exclusion of fold-1
labels from inner training bases, own-fold exclusion and label-free choice
features. All three pass. The real CPU check is running while N1's training
process is live, under the existing CPU/GPU priority rules.

### H8 nested result

The fixed check completed successfully. P2 fold-0 RMSE is 0.8465348720 for
the bank11+N1 ridge, 0.8464506905 for its matched base-only residual tree,
and 0.8461614543 with the choice features. Gain versus ridge is
**0.0003734177**, passing the original H8 threshold; gain versus the matched
tree is **0.0002892362**, with user-cluster delta-MSE interval
[−0.00064277, −0.00033636]. Round counts were selected on fold 1 only
(control 39, treatment 114), then fixed for the final 1–4 refits.

The signal persists under outer-fold isolation. This is a small CPU complement
to N1, not evidence of a 0.0022 gain or a full-bank promotion. Only fold 0 was
evaluated; the full-bank/all-fold gate remains necessary once N1's full-data
column is available. No original-probe score was read by this check.

Reports: `experiments/H8-nested-{result,verification}-20261031.json`.
A fresh rescore from the saved fold-0 predictions reproduces both comparison
records exactly; all 20 frozen inputs and the prediction payload were
rehashed unchanged. No recipe or threshold changed after the result.

## H8 fixed CPU column preparation (after nested P2 pass)

Claimed by Codex. `scripts/choice_affinity_column.py` fits a P2-only model
bundle and compiles a separate full-view feature cache. Its plans are
`experiments/H8-column-{fit,query-features}-plan-20261031.json`, frozen before
execution. P2 fit uses the original bank11+N1 columns, five-user-fold ridge
bases at penalty 10 and the unchanged H8 feature/tree recipe. Final tree
rounds are fixed from the nested result: base-only control 39, choice treatment
114. Both trees fit all P2 training labels without another score or search;
a full-P2 ridge supplies the weights for query inference.

Full-view features use `B-K1.bin`, the Phase 0 K1 trained on original training
ratings. The producer receives a newly allocated query array containing only
user, movie and date, with no rating field. No original-probe score is part of
this preparation. Applying the bundle still requires the corresponding full
N1 column, which is not yet available. A future full-bank/all-fold comparison
must measure the transfer; independently refitted factor embeddings can
change feature distributions, so the P2 gain is not assumed to transfer.

Two fixture tests cover exact saved-model prediction parity, schema/payload
rejection, fixed full training and label-independent query feature generation.
No GPU job or existing trainer is changed. This prepares the CPU complement
while preserving N1/SO1 queue priority.

### H8 column preparation complete

Both fixed CPU jobs completed. The bundle fits 1,407,566 P2 rows with control
39 and treatment 114 rounds. The separate full-view cache contains twelve
finite features for all 1,408,395 original-probe query identities/dates.
All frozen inputs and payload hashes were reverified; both serialized trees
load with the expected feature counts (1 and 13), exact fixed round counts,
and thirteen finite ridge coefficients.

Archived manifests are `experiments/H8-column-{fit,query-features}-manifest-20261031.json`;
verification is `experiments/H8-column-preparation-verification-20261031.json`.
The next boundary is the matching twelve-column query bank, including N1's
full-data epoch-13 prediction. No query predictions, original-probe scores or
full-bank promotion were produced by this preparation.

## H8 full-bank comparison protocol (before any original-probe H8 score)

`experiments/H8-full-comparison-protocol-20261031.json` binds the prepared
bundle/cache, eleven existing query columns in their P2 semantic order,
parent N1/SO1 plans and blender sources. The twelfth query column must be
N1's completed epoch-13 full-data prediction, validated against its original
plan/report; no partial or earlier-epoch substitute is allowed.

At launch, bind the published retained full-bank inputs/reference before any
H8 comparison, and include N1's raw column if absent. Compare that same bank
plus the base-only H8 column twice against the bank plus base-only and choice
H8 columns. Reuse the fixed three-seed `round4_boost` + NN2 50/50 recipe.
Require at least 0.0003 full-bank gain over matched control (the existing
SO1 full-bank minimum), every user fold better, and every fold better than
the retained reference. No post-probe setting, seed or threshold change.

The plan declares recipe and inference mapping; it does not install another
scheduler or launch jobs. Full comparison waits for N1/SO1 to finish and
release GPU capacity. CPU inference follows GOAL.md's active-stage guard.
All original-probe development and hidden-test limitations still apply.

## H9 choice co-occurrence embeddings (fixed CPU protocol, before any score)

Claimed by Codex on the strategy's choice-history line. Hypothesis: the small
H8 signal is constrained by supervised K1 embeddings; unlabelled-only choice
co-occurrence may expose complementary current tastes without another GPU
model. This tests a representation of existing choices, not an oracle label.

Build a binary user/movie presence matrix from implicit pairs absent from
training. Weight columns by log(1+users/(1+document frequency)), then normalize
each user row to unit L2 length. Compute a fixed randomized rank-32 SVD
(oversample 8, five reorthogonalized power passes, seed 20261031). Use
V*sqrt(singular value) and unit-normalized item vectors with H8's target-excluded
3-day/30-day/all-history pooling. Add nine affinity features to H8's twelve
features; reuse existing counts rather than duplicate them.

Use the same nested stacking, tree settings, 2,000-round cap and 100-round
stopping patience as validated H8. Control is H8 itself, not a weaker ridge.
Its saved fold-0 predictions must reproduce within 1e-8. Fold 0 labels enter
no ridge fit, residual training target or selection. Gate: >=0.0003 P2 RMSE
gain over matched H8; fail closes H9 without a rank/weight/window/seed sweep.
A pass still needs a separately frozen full-bank/all-fold test.

Plan: `experiments/H9-choice-cooccurrence-plan-20261031.json`; implementation:
`scripts/choice_cooccurrence.py`. Three fixture tests cover rating independence,
binary presence/IDF normalization, exact low-rank geometry, control replay
and outer-label independence. CPU only (two tree threads, BLAS one thread),
while the verified N1 training process remains live. No original-probe score
or existing trainer modification is part of this run.

### H9 result: closed

The fixed run completed on CPU. Its matrix contains 5,633,092 distinct
unlabelled user/movie pairs. H8 control RMSE reproduces exactly at
0.8461614543; H8 plus SVD affinities scores 0.8461137958. The gain is only
**0.0000476584**, below the fixed 0.0003 gate, with user-cluster 95%
delta-MSE interval [−0.000165064, +0.000003762], which includes zero.
Iteration selection used fold 1 (H8 114, treatment 89); fold 0 was evaluated
once. No rank, weighting, window or seed sweep follows.

Evidence: `experiments/H9-choice-cooccurrence-{result,verification}-20261031.json`.
The saved H8 control predictions match the earlier nested H8 payload exactly;
a fresh rescore reproduces the comparison, and all 23 frozen inputs and both
payload hashes were verified. This closes the tested SVD recipe, with no
full-bank or hidden-test claim. N1's existing GPU queue and the prepared H8
column remain the active path.

## H8 full-query columns emitted after N1 completion (2026-10-31)

N1's fixed full-data refit exited zero after all 13 epochs, selecting epoch 13.
Its final 1,408,395-row prediction SHA-256 is
`a4dc251df10084b417862e7ed87dd92f4a4fd438a2c8fdb66ec50039dc00022f`.
The completed report is archived in
`experiments/N1-completed-refit-report-20261031.json`. Exact parent arguments,
source hash, data hash and recorded prediction hash were independently checked.
This is artifact completion, not evidence of full-bank improvement.

While the owner's three tree-blend workers were verified active, the existing
H8 inference API emitted both full-query columns using the fixed twelve-column
bank, fitted bundle and feature cache. No query labels were loaded or scored.
Both outputs contain 1,408,395 finite float32 predictions, and all 25 protocol
inputs were hash-verified before and after inference. Evidence and output hashes
are in `experiments/H8-query-column-verification-20261031.json`; payloads are in
`artifacts/headroom/H8-query-columns-20261031/`.

H8 remains unscored. The pre-registered full-bank comparison still waits for
N1/SO1 queue completion and GPU release, then binds the retained reference before
any H8 score. Its recipe, matched control and all-five-fold promotion gate stand.

## N1 full-bank comparison independently verified (2026-10-31)

The fixed 13-epoch N1 refit and registered three-seed tree/NN2 comparison
completed. The 50/50 average scores **0.8667006277**, versus retained R9-LAT-A2h
**0.8668706424**: gain **0.0001700148**. Fold deltas (candidate minus retained)
are −0.000226401, **+0.0000377574**, −0.000210640, −0.000274501,
−0.000175834. The all-five-fold promotion gate fails because fold 1 worsens;
the substantial-continuation rule also fails. No promotion or reference refresh.

Independent scoring from the saved predictions exactly reproduces the owner's
comparison, including the user-cluster MSE interval. The saved average agrees
with the registered 50/50 tree/NN average to 2.39e-7. All four prediction hashes
were verified before and after scoring. Evidence:
`experiments/N1-full-independent-verification-20261031.json`.

The promoted best remains 0.8668706425 and the estimated development target gap
remains 0.0022206425. SO1's original control/treatment queue continues unchanged.
The already registered H8 comparison remains pending queue release, with its
N1-containing matched bank and unscored columns preserved. This result provides
no hidden-test verification and does not authorize a new N1 parameter sweep.

## SO1 closes; H8 full comparison bound (2026-10-31)

SO1's unchanged two-epoch pair completed. Its primary P2 matched-bank score
is control 0.8470131554 versus treatment 0.8469856351, a gain of 0.0000275202,
below its frozen 0.0005 P2 gate. The user-cluster MSE interval includes zero.
`experiments/SO1-independent-gate-20261031.json` independently reruns the
original frozen gate and reproduces the owner's complete JSON exactly.
The controller records `complete`, its trainer and controller PIDs are absent,
and the process snapshot shows no other Python trainer. No conditional full
SO1 stage is authorized by its failed gate. No SO1 file was edited.

`experiments/H8-full-20261031-plan.json` now binds the still-retained R9
reference (0.8668706424), its complete input bank plus raw N1 in both arms,
and all H8 inputs **before any H8 original-probe score**. Its 187 frozen
files and the bound plan are checked before each stage. Control adds the
base-only H8 column twice; treatment adds base-only plus choice. The four
sequential stages use the registered three-seed tree and NN2 recipes, with
three tree workers and two NN CPU threads. `H8-full-20261031-run.sh` records
separate stage logs and stops on any failure. The runner does not promote a
result; saved payload verification and the registered all-five-fold gate
remain required after completion. The GPU queue is released; these H8 blend
scripts use CPU, although the registered queue-release condition was observed.

## E3 fixed launch plan prepared (2026-10-31; no real E3 score)

`experiments/E3-plan-20261031.json` binds the exact SO1 control command,
completed control report and epoch-2 payload, and 30 frozen files. The
control's arguments, two-epoch budget, warm/data/source hashes, library
versions and saved prediction hash reproduce its frozen SO1 specification.
The E3 treatment changes only the separate trainer, output prefix and fixed
CE coefficient 0.1; its expected report arguments include the trainer hash.
Both the shared fold-0 gain and separate five-fold requirement are fixed at
0.0003. This E3 threshold is distinct from SO1's 0.0005 P2 threshold (the
initial SO1 closing note misstated that threshold; corrected above).

No E3 trainer or scorer was launched. E3's registered launch condition also
requires no other Python training to be live, so it awaits H8's four CPU
stages. The treatment's completion provenance must be audited before either
E3 scorer runs. The full-T follow-up remains conditional on both P2 gates.

### H8 control tree complete and verified (2026-10-31)

The first registered stage completed all five folds at RMSE 0.8669775992.
`experiments/H8-control-tree-verification-20261031.json` verifies the exact
planned configuration, input bank, library version, finite saved predictions
and reproduced saved scores using the existing full-comparison verifier.
All 187 frozen inputs were unchanged before and after this check. The shell
runner checked its input hashes again and advanced to the neural control
stage. Both treatment stages remain pending. This control-only result does
not establish an H8 gain and does not change the retained reference.

### H8 matched control complete and verified (2026-10-31)

The neural control stage completed all five folds: NN RMSE 0.8669253416,
fixed 50/50 tree+NN control RMSE **0.8667051174**. Verification in
`experiments/H8-control-neural-verification-20261031.json` checks exact
configuration/library, equal tree and neural input banks, finite payloads,
reproduced scores, and the saved average against its two components within
1e-6. All 187 frozen files remain unchanged before and after verification.
The unchanged runner passed its next hash check and launched the three-worker
treatment-tree stage. Treatment neural remains pending; no H8 gain or
promotion decision is available from the completed control alone.

## S0 — Structural-plan week-0 scoping (fixed before scoping scores)

`experiments/S0-scoping-plan-20261031.json` inventories the retained bank's
79 prediction columns, ten fixed families and 46 fixed support columns.
It addresses STRUCTURAL_PLAN.md's diversity and quality-substitution checks.
The original-probe labels have not been loaded for S0 yet.

**Diversity.** Use nested random subsets of 1, 4, 16, 32, 64 and 79 columns
from three fixed permutations (numpy default_rng seeds 20261031, 20261101,
20261102), preserving retained-bank column order. The full 79-column arm is
shared, giving 16 prescribed arms. Fit the existing three-seed tree + NN2
50/50 blend on all five user folds, with its unchanged recipe and fixed
support/context. This measures the actual retained blender's count curve;
it is not an information-source ceiling because support stays present.

**Quality substitution.** Compute clipped standalone scores for every
column. Within each declared family choose the best and worst by that score
(ties by original bank index), and replace its best occurrence by a duplicate
of the weaker sibling, keeping all other occurrences and support fixed.
Run all ten substitutions, regardless of early results. Report standalone
and blend degradation, all-five-fold deltas, and their finite ratio. A ratio
is undefined for zero standalone difference. Substitution changes diversity
as well as quality; it cannot identify a causal slope for future models.

**Interpretation gate.** The all-79 scoping arm must first reproduce retained
R9 within 1e-7 in saved predictions and 1e-8 in RMSE before interpreting any
curve. Report all permutations and substitutions; no winning-subset search,
no model promotion, no stopping H8 based on S0. Use the observed ranges to
rank the structural lanes, without promising the 0.0041 target gap or
asserting a model-family upper bound. The next step is the standalone
inventory and frozen executable arm manifests.

**Resources.** At most two S0 arms live, three tree workers or two NN threads
per arm, total own CPU threads at most ten including H8. No cloud spend.
E3's existing no-other-training launch condition continues to apply.

### S0 manifests and standalone inventory prepared (2026-10-31)

`scripts/structural_scoping.py` passes three fixture tests for fixed nested
subsets/ties, rejection of malformed inputs, and round-trip preservation of
duplicate prediction occurrences and support order. Each occurrence receives
a distinct symlink in a separate per-arm bank; ordinary frozen union/support
loaders read those aliases without changing either blender. This prevents a
best-to-existing-sibling substitution from silently losing a column through
dictionary deduplication.

The fixed run produced 26 executable arm manifests and all 79 standalone
scores, archived in `experiments/S0-execution-20261031.json` and
`experiments/S0-standalone-20261031.json`. Best standalone columns in their
fixed families include B4 0.889660, A2h 0.890141, GP2 0.892294, and I2R
0.894582. These are reused original-probe standalone development scores;
no scoping blend score has been computed. The family substitutions follow
the registered deterministic best/worst rule, without inspecting a blend.

Only the all79 reference reproduction is prepared for initial training
(`experiments/S0-all79-20261031-run.sh`). It uses three tree workers then
two NN threads. The remaining 25 blend arms cannot be interpreted until
this control passes the predeclared saved-prediction/RMSE parity check.
H8 remains live and unchanged; the combined initial tree budget is six
CPU workers, below GOAL.md's limit. No GPU or cloud job is launched by S0.

### S0 coordination and narrative-document provenance (2026-10-31)

The other session completed a separate ridge-proxy analysis
(`scripts/scoping_bank.py`, `artifacts/scoping/S0-bank-scoping.json`). It
excludes support/metadata and measures family quality after removing all
siblings. S0's registered full tree+NN comparison retains support and tests
best-to-weaker substitution in the retained bank. These answer different
questions; the ridge slopes are not a replacement for full-blender parity
or the registered S0 results. No existing S0 recipe or arm is changed.

The other session also legitimately extended STRUCTURAL_PLAN.md. It is the
**only** frozen S0 execution input whose bytes now differ. The exact document
registered before S0 scores is recovered from commit e4ff55d into
`experiments/S0-structural-plan-snapshot-20261031.md`; its SHA-256 equals the
original expected hash. `S0-execution-document-rebind-20261031.json` changes
only that narrative input's path to this byte-identical archival snapshot.
Its scientific inputs, all 26 arm definitions and every training command
are exactly equal to the original execution plan.

The running S0 script and hash manifest are left untouched. After completing
the tree stage, its original check is expected to stop it before NN because
the live narrative document changed. Confirm that terminal state, preserve
the completed tree, and use the documented administrative continuation for
the unchanged neural command. Evidence is
`experiments/S0-document-rebind-verification-20261031.json`. This rebind does
not authorize changing a feature, model, hyperparameter or decision gate.
Shared STATUS.md now distinguishes the completed ridge proxy from the live
full-blender check and records both live jobs.

### H8 treatment tree complete and verified (2026-10-31)

The third registered stage completed all five folds at RMSE 0.8667581915.
`experiments/H8-treatment-tree-verification-20261031.json` verifies the exact
configuration, bank and library, finite saved predictions and reproduced
scores, with all 187 frozen inputs unchanged before and after checking.
The unmodified runner checked its inputs again and launched the fourth and
final stage, treatment NN2. Its completed 50/50 average, paired comparisons
against the matched control and retained R9, and the registered full-bank
gate remain pending. The tree-only result does not authorize promotion.

### S0 all79 tree exact parity; unchanged NN continuation (2026-10-31)

The original all79 tree stage finished all five folds at 0.8671122161 RMSE.
Its saved predictions are **exactly equal** to retained R9's tree payload
(maximum absolute difference 0.0). The existing full-comparison verifier also
checks its exact configuration, bank, library, finite predictions and saved
scores. `experiments/S0-all79-tree-verification-20261031.json` records this.

The original managed session exited 1 after the planned narrative-document
hash failure, before neural training. Its completed tree is preserved. The
new `S0-all79-NN-continuation-20261031.json` binds the byte-identical archived
document, all otherwise unchanged frozen inputs and the completed tree's
outputs, and contains the **exact** original NN command/configuration.
The companion shell/hash manifest will run only that neural stage. Full
reference parity is still unproven until its saved 50/50 average is checked;
no other S0 blend arm has been launched.

## H8 full comparison complete — useful signal, fixed gate fails (2026-10-31)

The unchanged shell runner completed all four stages and exited zero.
Independent saved-payload verification in
`experiments/H8-full-independent-comparison-20261031.json` confirms exact
planned configurations, banks and library versions for every stage, finite
outputs and reproduced scores, the fixed 50/50 averages within 1e-6, and the
matched arm design. All 187 frozen inputs were unchanged before and after
verification. No model, recipe, seed or threshold was adjusted after scoring.

| Comparison | Control RMSE | Treatment RMSE | Gain | All five folds better? |
|---|---:|---:|---:|---|
| matched control → treatment | 0.8667051174 | 0.8664848395 | 0.0002202779 | yes |
| retained R9 → treatment | 0.8668706424 | 0.8664848395 | 0.0003858030 | yes |
| retained R9 → matched control | 0.8668706424 | 0.8667051174 | 0.0001655251 | yes |

The matched-comparison user-cluster 95% MSE interval is
[-0.00044347, -0.00032010]. The signal is measurable, but its improvement
beyond the matched control is **below the registered 0.0003 minimum**.
That condition was explicitly part of H8's full-bank promotion rule, so
`passes_promotion_rule=false`. H8 is not promoted, no threshold is relaxed,
and the retained R9 score and winner-target gap remain unchanged. Preserve
the candidate as development evidence; no feature/round/seed continuation
is authorized by this failed gate. Hidden labels remain unavailable.

H8's trainers are terminal. S0's unchanged neural reference reproduction
continues, so E3's no-other-training launch condition is not yet satisfied.

### E3 exact launch script ready after H8 (2026-10-31)

`experiments/E3-select-20261031-run.sh` contains the exact command already
bound in E3-plan-20261031.json, preceded by checks of its 30 frozen inputs
and the plan itself. Bash syntax passes; all frozen inputs currently match.
No E3 process has started because S0's reference NN is still training.
After that process is terminal, inspect the whole process list and memory,
confirm the existing GPU queue is terminal and capacity is idle, and start
E3 before additional CPU scoping training. Its two-epoch budget, CE=0.1,
matched SO1 control and both predeclared P2 gates remain unchanged. The shell
runner performs no scoring or automatic promotion.

## S0 full reference parity verified; E3 launched (2026-10-31)

S0's unchanged NN continuation completed and exited zero. Independent
verification in `experiments/S0-full-parity-verification-20261031.json`
checks both full-blender configurations/banks/libraries and all saved scores.
Its tree, neural and 50/50 average prediction payloads each match retained
R9 **exactly** (maximum absolute differences all 0.0); saved average CV is
0.8668706424233039. The within-arm fixed-average error is below 1e-6, all
frozen inputs remain unchanged under the documented narrative-path rebind,
and all 26 planned commands are identical to the original execution plan.
The predeclared reference gate passes; the other 25 fixed S0 arms remain
unlaunched and are now interpretable once completed. No subset is promoted.

A fresh whole-process snapshot then showed no Python trainer live, the
N1/SO1 and H8 queues were terminal, and system memory was 92% free. E3's
frozen two-epoch P2 treatment launched only after that check, using its exact
registered command. All 30 frozen inputs and the plan passed the shell's
hash check. Live trainer PID 38985 and managed session 51948 are recorded in
`experiments/E3-select-20261031-launch.json`; output is
`experiments/E3-select-20261031.log`. No E3 score or recipe adjustment has
occurred. Its completed SO1 control and both predeclared P2 gates stand.

### S0 remaining 25 fixed arms bound into two CPU lanes (2026-10-31)

`experiments/S0-lane{0,1}-20261031-plan.json` partitions all 25 original
non-reference arms into 13 and 12 entries, preserving their relative order
within each lane. Their commands/configurations are exactly equal to the
original execution plan. Both lanes require the exact full-reference parity
receipt and bind its completed payloads, the archival-document execution
plan, every frozen source/input, their own lane plan and shell script.
Hash checks run before every stage; a failed command stops that lane.
The two shell scripts pass bash syntax checks and perform no adaptive search
or promotion. All predeclared arms remain in the analysis regardless of
intermediate scores.

Each lane runs at most one three-worker tree stage or two-thread NN stage.
With E3's one CPU thread and GPU treatment, the configured maximum is seven
CPU threads (below the shared ten-thread cap) and three simultaneous
training jobs. E3's no-other-training launch condition was satisfied before
it started; these subsequent scoping jobs use CPU only and preserve its GPU
priority. Memory/process checks remain required before each lane launch.

### S0 CPU lanes live alongside E3 (2026-10-31)

Both frozen S0 lane scripts are now live after the full reference-parity
gate. Lane 0 (runner 39539, managed session 96236) starts with the fixed
one-column arm; lane 1 (runner 39615, session 83214) starts with the fixed
four-column arm. Each passed every frozen hash check before training.
`experiments/S0-lanes-20261031-launch.json` records both handles and the
preflight resource checks: memory 54% free before lane 0, 45% before lane 1.
E3 was already actively training, having satisfied its launch condition
before these CPU jobs began. Maximum configured own CPU threads are seven;
S0 uses no GPU. Monitor memory as the two scripts advance through their
unchanged 13+12 arms. No partial diversity result is used to change the plan.

### E3 audited completion-to-gate command ready (2026-10-31)

`scripts/e3_candidate_gate.py --plan experiments/E3-plan-20261031.json
--output-dir artifacts/ordinal/E3-gate-20261031` audits both arms before
creating any scoring output. It checks exact frozen arguments (including
E3's CE coefficient and trainer hash), matched architecture and sources,
training libraries, data/warm hashes, the fixed full two-epoch budget, and
all recorded prediction/checkpoint hashes. It then writes owner-validated
receipts and a frozen shared-gate manifest, calls the unchanged shared P2
fold-0 scorer, and calls the already registered E3 five-fold check. Both
must pass; no metric, threshold, fold split or training budget is changed.
All-fold predictions are saved without float32 rounding, and hashes and
plans are checked again after scoring. No automatic full training or
promotion is performed.

Seven fixture tests pass across this command and the existing all-fold
helper. They cover the full audit-to-two-gate handoff, rejection of partial
training before any scoring output, architecture/source/warm/payload
mismatches, exact shared fold-0 parity and held-fold label exclusion. Only
fixtures ran; E3 remains in epoch 1 and no real E3 P2 score was read. The
command's source will be frozen into its scoring manifest before the real
gates, while the original E3 training plan and trainer remain unchanged.

### E3 fixed treatment complete; both gates fail (2026-10-31)

The unchanged two-epoch CE=0.1 treatment and frozen automatic audit handoff
both exited zero. The completion audit verified exact arguments, source and
warm/data hashes, libraries, full two-epoch budget, and every saved prediction
and checkpoint hash before scoring. Epoch-2 standalone P2 RMSE is
0.8649681712; this is not the incremental-value gate.

The shared fold-0 comparison is control 0.8470131554 versus treatment
0.8470144642: **0.0000013089 worse**. The separate nested five-fold comparison
is control 0.8477011478 versus treatment 0.8476883811: gain
**0.0000127667**, below the fixed 0.0003 minimum. Fold deltas (treatment minus
control) are [+0.0000013089, +0.0000001542, −0.0000171297, −0.0000063721,
−0.0000418458]; folds 0 and 1 worsen. **Both registered gates fail.**

`experiments/E3-comparison-20261031.json` archives the unchanged result;
`experiments/E3-independent-verification-20261031.json` verifies frozen inputs
and output hashes, and exactly reproduces both comparisons from the saved
float64 predictions, including shared fold-0 parity. Training and handoff
managed sessions are terminal at exit zero. Close this fixed bet with no
full-T pair, coefficient/epoch/seed sweep, or promotion. Retained R9 remains
0.8668706424 dev CV; the estimated winner-target gap remains about 0.0041.

S0 continues independently: the first four-column tree passed the unchanged
config/bank/library/saved-score audit at RMSE 0.8747112544, and its neural stage
is live. Its final 50/50 result and the other fixed arms remain pending. No
partial scoping result changes their frozen recipes or schedule.

### S0 complete-result reporter frozen (2026-10-31)

`scripts/structural_scoping_report.py` requires every registered arm's tree
and neural reports and payloads before loading labels or creating a summary.
It reconstructs the fixed arm definitions from the frozen standalone inventory,
reuses `selection_objective_full.verify_blend` for exact config/bank/library
and saved-score checks, requires all five stage folds and identical component
banks, and verifies each saved 50/50 average. All79 tree/NN/average parity is
checked again directly against retained R9.

The JSON report retains all arm comparisons and user-fold deltas, all three
permutations' diversity results (shared all79 counted explicitly as one run),
and every family's signed blend-degradation/standalone-degradation ratio,
with an undefined ratio for a zero denominator. There is no curve extrapolation,
winner selection, training, promotion, or future-gain claim. Frozen support and
context remain part of every arm, and substitution ratios are observed finite
interventions, not causal derivatives or upper bounds.

`experiments/S0-report-binding-20261031.json` binds the reporter and dependencies,
the unchanged execution, and the reference-parity proof before any nonreference
complete average. Five fixture tests pass. The real completed all79 arm also
passes the reporter audit with exact three-payload parity, and the actual
incomplete 26-arm experiment is rejected before a summary file is created;
`experiments/S0-report-preflight-20261031.json` records these checks. The
separate bash handoff waits for both existing lane runners to exit, then runs
this audit once. A failed or incomplete lane causes rejection, not a retry or
partial synthesis. The training scripts and frozen plans remain unchanged.

### S0 first complete nonreference arm audited (2026-10-31)

The fixed four-column arm (`diversity-s20261031-n4`) completed tree and neural
stages. The bound reporter's `audit_arm` reproduces both saved scores and
verifies exact stage configurations, bank inputs, library versions and all
five folds. Saved tree/NN/50–50 RMSEs are 0.8747112544 / 0.8746226105 /
**0.8743334426**; average/component maximum difference is 2.3842e-7, within
float32 rounding tolerance. All six output hashes are archived in
`experiments/S0-diversity-s20261031-n4-full-verification-20261031.json`.

This is 1/25 fixed nonreference arms, not a diversity-curve conclusion. Lane 1
advanced unchanged to the registered n32 tree stage; lane 0's n1 neural stage
continues. The final report still requires every arm, and no partial result
changes the schedule, recipes, subset selection or promotion rules.
