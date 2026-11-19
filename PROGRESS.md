---
status: fresh-only-reproduction-active
branch: research/full-upstream-reproduction
timestamp: 2026-11-19T18:26:00+00:00
best_score: 0.8593661638722666
score_metric: adaptive five-fold known-ProbeX RMSE using own freshly trained predictors
qualifying_status: audited_miss
qualifying_target_per_recovered_group: 0.85725
---

# Netflix Prize progress checkpoint

## Current fresh-only reproduction (2026-11-19 UTC)

New verified development best: **0.8593661638722666**, the fixed half mean of
own107+50 expanded tree and NN. All five tree folds, all15 saved-tree replays,
reused actual neural replays and separate compensated scoring passed. Four
folds improve and one worsens; lower aggregate RMSE is promoted. The new mean
requires a full-known tree refit and a separate frozen qualifying audit.
The larger all90 GPU neural worker24204 is now fitting.
See [RBM50_TREE_RESULT.md](RBM50_TREE_RESULT.md).

Earlier observation at18:26 UTC (superseded by the complete tree result above):

New independently verified own qualifying best: **0.8609266297566702 /
0.8619141823430468**, from the predeclared standalone own107+50 NN. Full14/14/13
fit, actual saved-model inference replay, frozen-vector identity and separate
canonical group scoring passed. Worse group improves9.5103221%; both still miss
0.85725. See [RBM50_QUALIFYING_RESULT.md](RBM50_QUALIFYING_RESULT.md).
The107+50 tree study has completed four of five folds; its final seed is
fitting in fold4. Promotion requires complete saved-model replay and independent
scoring. No partial-fold score is promoted.

Latest observation at18:26 UTC: both original latent models are complete20/20
and independently verified, making all90 original support columns ready. The
rank1000 second phase has logged20/20 but its process remains active; complete
outputs and independent verification are still required. CF96 saved states are
42/40 of95, with both native processes active. Original split3 dnn16t started
at18:18 UTC and is training, with the other two recipes serialized behind it.

The larger all90 hidden128/rank32 study is queued on the tested float32 MPS
backend behind the tree fit. A second predeclared batch adds all three complete
native rating pairs to make110 raw columns, retaining all90 supports. Its queue
currently waits only for rank1000's independent whole-model completion proof.
The synthetic MPS benchmark was3.36x faster than CPU with bitwise saved-model
replay; full-data accuracy and speed remain unmeasured.
See [LATENT90_MPS_CONTROL_PLAN.md](LATENT90_MPS_CONTROL_PLAN.md) and
[NEXT_WHOLE_NATIVE_BATCH.md](NEXT_WHOLE_NATIVE_BATCH.md).

Current verified development best remains **0.8594554093063749**, from107 own
raw predictors plus50 learned RBM user supports. All15 saved neural models
replay. Its full-known refit and qualifying audit are complete as reported above.

A matched known-row ridge diagnostic compares the two previously disclosed
cached and paired source realizations:0.8625801964 versus0.8625751216, a paired
improvement of0.0000050748. This small ridge effect does not explain the observed
development-to-qualifying gap in this diagnostic; it does not establish neural
parity or any new qualifying score. No qualifying labels were read.
See experiments/upstream-paired-realization-diagnostic/ridge.json.

Earlier observation:

Local update: own105 completed all five folds and all15 model replays. Its
NN0.8594783385 is slightly worse than NN103, so it is not promoted. The next
actual worker98536 is training on107 own predictors plus all50 exact learned
RBM user supports. Every support query row and every appended cache column was
checked bitwise. See [RBM50_CONTEXT_PLAN.md](RBM50_CONTEXT_PLAN.md) and
[SPLIT3_105_CONTEXT_RESULT.md](SPLIT3_105_CONTEXT_RESULT.md).
Original RBM50 continues its second20-epoch phase; Tx20lat remains queued.
Both CF96 phases continue from verified original checkpoints, with latest saved
epochs36/34, alongside the original1,000-factor temporal model. Original
23/23 temporal60 is complete. The returned CF48 result is already incorporated.

**New verified own qualifying best:0.8609916544142975 /0.8619938614020105**,
from the predeclared standalone full-fit NN103. Worse-group improvement is
9.5019568%; both still miss0.85725. Full model/scaler/source bindings, actual
saved-model replay, canonical labels, full group coverage and independent
compensated scores verify. The fixed tree91/NN103 mean also improves, to
0.8610694976822610 /0.8620220770129196. See
[CF48_103_QUALIFYING_RESULT.md](CF48_103_QUALIFYING_RESULT.md).
Own105 is complete without improvement; the raw107 plus50-support study is now training.

Earlier checkpoints below retain their original observations.

The completed own103 context iteration gives a new independently checked
known-ProbeX development best: **standalone NN0.8594692563654075**, versus
0.8599596977902322 previously. The fixed half mean with tree91 also improves,
to0.8595400254184109. All15 saved NN models replay across the original five
nested user folds. Full-known production is now running under controller74649;
the stronger NN-only candidate is separately declared before either new
qualifying outcome. See [CF48_103_PRODUCTION_PLAN.md](CF48_103_PRODUCTION_PLAN.md).
The qualifying target is still unmet and the best verified worse-group audit
remains0.8624021195492101.

The new context91 qualifying audit is complete and independently rechecked:
0.8615005553485519 /0.8624166617824088, both above0.85725 and slightly worse
than the preceding best. Keep0.8624021195492101 as the best worse-group score.
See [CONTEXT91_QUALIFYING_RESULT.md](CONTEXT91_QUALIFYING_RESULT.md).
The own103 context fit is complete; it includes
our now-complete CF48 epoch95 and has independently replayed ridge0.8626072996.
See [CF48_103_CONTEXT_PLAN.md](CF48_103_CONTEXT_PLAN.md).

The completed all91 context fit gives a new verified development best:
**0.8599596977902322**, versus0.8599745622436025 on the same known-ProbeX
rows/user folds. Actual saved-family replays, ten source fold/model bindings,
the fixed float32 mean and compensated scoring verify. Its qualifying production with exact full tree91 reuse and a new
contextNN91 full-known fit has completed, with the slightly worse result
recorded above. See
[CONTEXT91_PRODUCTION_PLAN.md](CONTEXT91_PRODUCTION_PLAN.md).

At approximately11:27UTC, our published CF48 epoch25 snapshot pair passes
captured-package/source/data checks. The fixed ready-eight plus snapshot25
arm improves the own93 ridge diagnostic from0.8631793373 to0.8626981139;
a separate five-user-fold solve and compensated score replay confirm it.
The complete95-epoch CF48 trajectory is still running remotely; this is a
snapshot diagnostic, not whole-job completion or a qualifying promotion.
Own102 data is prepared and verified, and controller49665 queues its context
model behind the running all91 fit, ahead of the stopped idle93 queue.
See [CF48_PREFIX_102_PLAN.md](CF48_PREFIX_102_PLAN.md).

At11:04UTC the first isolated split3 pilot completed its native data preparation:
98,720,013 training rows and1,760,494 validation rows. All1,128,164 known ProbeX
rows map uniquely into validation and were verified absent from generated
training. The original s3-attn16 two-phase fit is queued behind measured memory
admission under live controller40527. It subsequently started at11:08:33UTC as
native PID42926. See [SPLIT3_PILOT_PLAN.md](SPLIT3_PILOT_PLAN.md).
This is training progress, not a new score.

At approximately 10:55 UTC, a fixed eight-column addition using already completed
own native outputs improves the own93 ridge diagnostic from 0.8631793373 to
0.8631599063, independently replayed. This does not change the best nonlinear
development or qualifying score. A source coverage audit establishes that all
263 public split3 raw columns have declared native recipes, outside the current
new-split queue. See [READY_OUTPUT_COVERAGE.md](READY_OUTPUT_COVERAGE.md) for the
completed screen, exact coverage limits, and next isolated reproduction lane.

The best tree91/contextNN86 own-training qualifying audit **misses** the target:

| Recovered source group | Rows | Independently verified RMSE | Required |
| --- | ---: | ---: | ---: |
| source0 (`is_test=0`) | 1,408,789 | **0.8614776640541453** | <0.85725 |
| source1 (`is_test=1`) | 1,408,342 | **0.8624021195492101** | <0.85725 |

The worse group improves **9.4590950605%** over the Cinematch reference 0.9525,
short of 10%. The remaining gap is 0.0051521195 RMSE. These are adaptive
community-recovered outcomes, not official or untouched Netflix Test scores.
The published Quiz/Test sizes reverse the source labels, so source groups
are preserved without relabeling.

The earlier first85 result used our freshly trained 85-predictor bank and full-known-ProbeX
blend fitting. Public source code is reused; no downloaded pretrained prediction
features or qualifying outcomes entered fitting or selection. Both of its saved full
families replay bit-for-bit, their fixed float32 mean verifies, and independent
compensated scoring covers all 2,817,131 qualifying rows. The separate completion
check binds 333 evidence files, 29 own source jobs and the actual development
models behind the per-seed budgets. It explicitly records `goal_achieved=false`.

The preceding matching86 result improved both first85 group scores (0.8619187880 /
0.8628067765). Its three trees and three context neural models replay before
freezing; a separate checker verifies source/code bindings, model/replay hashes,
the bitwise float32 mean and compensated group scores. See
`FIRST86_QUALIFYING_RESULT.md`. No qualifying outcomes selected the newer tree91/contextNN86 recipe. Its
production is now complete with a fresh tree91 and explicitly reused, replayed
own full-fit NN86. A separate frozen-mean/score check verifies408 source inputs
and28 code files; see `TREE91_CONTEXT86_QUALIFYING_RESULT.md`.

Our best verified **development** result is now **0.8599745622436025**: a
fixed half mean of the completed tree91 vector and the unchanged contextNN86
vector, improving the previous0.8600300601883686. It uses the same1,128,164
known-ProbeX rows and462,858 users. All ten actual originating fold receipts
and models were rehashed, prior full saved-model replays verified, row/fold
identities matched and the stored float32 mean scored independently. This is
adaptive development evidence on different rows from the qualifying audit.

The full fixed91 plain-tree/plain-NN study also completed and replayed both
families. Its0.8601526901474831 mean did not improve the preceding best and is
not promoted. The new heterogeneous-family production recipe is documented in
`TREE91_CONTEXT86_PRODUCTION_PLAN.md`; it preserves separate91/86 feature and
ridge layouts, actual per-seed budgets and the fixed half mean.

| Work | Verified state |
| --- | --- |
| First paired 85 qualifying fit | Complete; both source groups miss the target |
| Matching86 production | Complete and replayed; frozen audit0.8615073690 /0.8624355593 misses target. The exact full-fit NN86 is available for the new combination |
| Fixed 91-predictor study | Complete: both families replayed; fixed mean 0.8601526901 is worse and not promoted |
| New tree91/contextNN86 production | Complete: fresh tree and exact reused own NN replay; frozen audit0.8614776641 /0.8624021195 still misses target; separate mean/score recheck passes |
| All91 context follow-up | Controller 10867 waits behind both production runs and two physical-memory admission samples; it now binds the stronger 0.8599745622 development reference |
| Temporal93 context follow-up | Fixed tsvdx4-60o + tsvdx4-q16tb addition prepared; independently replayed ridge0.8631793373 versus0.8633016697. Controller29784 queues after context91; diagnostic gain is not a global best or qualifying result |
| Original1,000-factor model | Native canonical 20-epoch job started with 16 logical chunks/four physical workers; no reduced capacity |
| Local temporal150 phase | Complete at08:48 UTC; all31 native NPY outputs match the original remote receipt exactly, including fulltrain residuals; all eight selected split vectors equal the recovered own pairs |
| CFNADE96 | Checkpoints33/31 preserved and all84 tensors rehashed. Auto-resume admitted during the primary job transition, then yielded again at09:41 UTC after the next native allocation raised total owned footprint to26.9 GiB; no automatic retry remains live |
| CFNADE48 | Remote live-process heartbeat 09:36 UTC (commitc63caea) reports 70/68 epochs and zero swap use; deadline 14:33 UTC, return 15:03 UTC |
| AEX64/AEX128 | Completed; own AEX128 safely imported at an idle scheduler boundary |
| Primary DAG | Original tsvdx4-60o completed; primary DAG advanced to tsvdx5-60, native PID16305, which remains active |

The original-schema phase bank is now complete through the explicit recovered
own-temporal route; all six cache files match the verified private bank exactly.
The unchanged matching86 queue restarted at07:50 UTC and has passed the fixed91
study gate and started full tree fitting at09:02 UTC. The newer tree91/contextNN86 candidate
now has a concrete controller. The matching86 fit started before replacement;
it remains intact, and the new controller waits for its complete original
contextNN86 rather than duplicating that exact neural fit. The original NN
plan, ridge, bank, budgets, models and semantic replay must all match. New
tree91 fitting then uses its separate 167-column ridge and 228 inputs.
The two neural-origin gate tests reject changed recipes, budgets, model sets,
plans and replay evidence. Both actual queue processes and launch code hashes
were verified in `experiments/upstream-tree91-context86-production/startup-verification.json`.

The paired production bank has now passed an independent full-row column
check against its actual declared own native pairs: original predictions,
clipping, float32 centering, context interactions, intercept, all130 voting
features and all six additions. Known labels and user identities match the
original development rows. Nine base predictor columns differ from the frozen
development realization, from four retrained source producers. This was
already disclosed in the original parent bank's `feature_recipe`; it is now
quantified in `experiments/upstream-extension-bank/paired-source-differences.json`.
The checker initially assumed identical source realizations, then was corrected
to validate the declared paired inputs while reporting those differences.
Production bank files and model math were not changed. The development gain
is recipe evidence; it does not imply the same score on the rebuilt bank or
on qualifying rows. Both current full-fit families learn from that rebuilt
known-ProbeX bank, with their original schemas and no qualifying-label fitting.
Two regression tests cover valid retrained inputs and corrupted column math.

Temporal completion evidence is in
`experiments/upstream-temporal-phase/completion-verification.json`, produced by
`scripts/verify_upstream_temporal_completion.py`. This is a completed original
fulltrain/qual phase, not a new full two-phase pipeline or a new score.

The fixed all91 context follow-up is documented in `ALL91_CONTEXT_PLAN.md`.
Its first idle controller 97839 was stopped only after confirming no children,
and its launch/status/code snapshot preserved. Updated controller 10867 waits
behind the new production queue and binds the stronger fixed-combination
reference; no live model or native training source was changed.
Its actual train/save/reload test reproduces predictions exactly. It retains
all six predeclared additions, reuses the fully verified matching trees and
starts no fit until its predecessor and physical-memory gates clear.

A second CF96 admission exposed a scheduler race: both CF96 workers and new
production admitted while the next primary native job was still loading. Its
later allocation raised owned physical footprint to26.9 GiB and compressor use
to16.3 GiB. At09:41 UTC only the checkpointed CF96 controller16335 and its owned
workers16378/16379 were yielded through the original cleanup path. All84 tensors
and exact state/source/data identities remain unchanged at33/31. The active
uncheckpointed native jobs and tree91 fit were untouched. Evidence is in
`experiments/upstream-memory-observation/cf96-second-resource-yield.json`.
The old one-shot waiter has exited. Controller21217 now waits for a verified
primary-job idle boundary, then takes over its launch lock for unchanged CF96
training and restarts the original primary controller afterward. Actual-process
tests cover busy-owner preservation and idle takeover. Neither the current
primary job nor its scheduler is stopped while this lane waits. See
`CFNADE96_PRIMARY_LANE_PLAN.md`.

The two completed temporal models are now prepared as a fixed93-predictor
context follow-up. Its separate ridge checker recomputed grouped sufficient
statistics and held-user predictions; compensated scoring confirms0.8631793373
versus0.8633016697, with maximum replay error below7e-15. A230-input rank16
context NN is queued as controller29784 after context91, preserving all optimizer,
fold and seed rules and reusing the actual verified tree91. A train/save/reload
check reproduces predictions exactly. No qualifying labels were read, no
nonlinear fit has started, and no new overall score is promoted. See
`TEMPORAL93_CONTEXT_PLAN.md` and `experiments/upstream-temporal93/`.

The additional remote branch `research/remote9h-cf96-handoff` contains the
verified CF96 checkpoint package and runbook. No remote CF96 restore proof has
returned yet. The runbook first requests all63 outputs of our already completed
remote temporal job. `scripts/verify_remote_temporal_archive.py` is ready to
independently verify that return against the original receipt and local input
identities in an isolated directory. Its tests reject archive traversal, links,
duplicates and incomplete payloads. It does not control any live job; the local
independent phase subsequently completed naturally at08:48 UTC. Its separate
completion verifier confirms all31 native NPY hashes match the original remote
receipt, including full training residuals; the eight selected vectors match
the already admitted own recovered pairs. The full63-output remote archive
remains useful for the complete original pipeline provenance and training
outputs, but is no longer needed to retire that local process.

The first separate completion checker encountered an OpenMP runtime clash when
loading LightGBM and PyTorch together. Its replacement replays each family in
an isolated process; all three tests and the full completion audit pass. No
production fitting artifact changed during that correction.

The website and plots show the new own-training qualifying audit separately
from historical public-bank results and known-ProbeX development. Historical
promotion revisions remain: LC6/HR2a are promoted candidates, while HR2b/N1/H8
advance the chronological original-Probe record. See
[PROMOTION_POLICY.md](PROMOTION_POLICY.md).

Current evidence:

- `experiments/upstream-memory-observation/cf96-resource-yield.json`: preserved33/31 checkpoints and measured active-paging evidence.
- `experiments/upstream-cf96-memory-resume/launch.json`: guarded original-controller resume.
- `experiments/upstream-cfnade96-sparse-resume/kernel-verification.json`:32 synthetic parity cases; full-data trajectory proof is still required.

- [FIRST85_QUALIFYING_RESULT.md](FIRST85_QUALIFYING_RESULT.md): complete scores, model replay, scope and continuation.
- `experiments/upstream-first85-completion-audit/verification.json`: independent completion audit, 333 evidence bindings and explicit target miss.
- `experiments/upstream-phase-production/recovered-audit/report.json`: original frozen-vector audit and immutable recovered-label provenance.
- `experiments/upstream-phase-production/independent-qualifying-verification.json`: independent compensated scoring.
- `experiments/upstream-phase-production/report.json`: model report frozen before label access.
- [REMOTE_RECOVERY_PROGRESS.md](REMOTE_RECOVERY_PROGRESS.md): recovery, migration and runtime correction.
- `experiments/upstream-cf12-matching-recovery/plan.json`: one-time dependency recovery.
- [CFNADE12_MATCHING_TREE_PROGRESS.md](CFNADE12_MATCHING_TREE_PROGRESS.md): verified development best.

## Historical public-bank result (2026-11-16)

The scores below use public pretrained-bank inputs. They do not satisfy the
active fresh-only reproduction goal above.

The entire-original-Probe best is now **0.864533752805203**, using the 466-base
local/public NN2 ensemble. Separately, the frozen 619-input contextual ridge
scores **0.8571903481** on recovered source1 and **0.8561542082** on source0:
both pass **0.85725**, and source1 improves over 0.9525 by **10.00626267%**.
Independent CSV scoring and bitwise inference replay verify the result.
The local submission gzip passed exact row/header/float round-trip checks.

Source group counts conflict with published Quiz/Test counts; both were
checked without relabeling. This is adaptive community evaluation, not an
untouched official Test score. CF predictors and Probe-fitted pretrained
blenders are distinguished: the final stack fits known Probe/ProbeX ratings,
while qualifying outcomes are excluded from all fitting/stopping. Its
upstream meta-model dependence prevents an independent development-CV claim.
Optional unfinished experiments were stopped after the target was independently
verified; checkpoints and logs remain. Evidence is in
[QUALIFYING_PROGRESS.md](QUALIFYING_PROGRESS.md) and [RESULTS.md](RESULTS.md).

## What made the breakthrough possible

The retrospective is [BREAKTHROUGH.md](BREAKTHROUGH.md). The successful run
shifted from training more individual models to integrating reusable prediction
pairs and auditing which public intermediate blends fit only Probe labels.
It builds on the earlier local modeling work and external pretrained research.

The decisive matched comparison retained 536 individual CF predictions beside
the compact 83-input blend bank. With the same development population,
contextual-ridge method, and selected alpha 0.001, recovered source1 RMSE fell
from **0.8575942364 to 0.8571903481**. The final winner is the contextual
ridge, not the additional 619-input neural candidate, which remains unaudited.

The recorded optimization goal took approximately 8h 38m; that excludes
earlier local and upstream training. Original-Probe CV, alternative ProbeX
scores, and qualifying audits remain separate metrics. The writeup records
the sequence of improvements, failed approaches, verification, and limits
on attributing the gain.

## Consolidated research branches (2026-10-11)

All fetched remote research branches are merged into `main`; branch refs are
retained. The best **promoted** development score is **R9-LAT-A2h 0.8668706425**:
A2h's 79 prediction columns, 22 training-derived K1 latent and bias features,
three-seed LightGBM, and a fixed 50/50 average with the NN2 neural residual
blend. All five held-out-user folds improve over R8-A2h-NN2 (0.8670984524).
The latest B5 Bayesian item-frequency column yields a lower mean (0.8668212)
but worsens two folds, so it was not promoted. Round 10 kNN neighbour feature
extraction has launched; no completed blend score is recorded for it.

This is original-Probe development CV, not a hidden-test score. Probe-guided
selection remains. [MERGE_RECORD.md](MERGE_RECORD.md) lists each merged branch,
its method, measured score and result status; [ROUND8.md](ROUND8.md),
[ROUND9.md](ROUND9.md), and [ROUND10.md](ROUND10.md) retain the run details.

## New best: fixed C10/C11 average

The strongest reproducible original-probe development score is now
**C12 0.8747780516**. It is a fixed equal average of C10 and C11's saved
out-of-fold columns. The saved float32 result independently scores
0.8747780515 and has SHA-256
`93129d524dcd2749d7fc7ccdaf32ee03277f7b2a942035805a0c3dd07eed6940`.
C12 is 0.0000313573 below historical R5 and 0.0197780516 above 0.855.
The R5 margin is tiny and probe-guided selection affects both scores.
`experiments/SESSION-C12-fixed-half.json` records both input hashes, five
fold scores and the output hash. C11 itself adds two training-only same-day
joint-residual features to C10 and scores 0.8748606495; four of five folds
improve. See `SESSION_META.md` and the C11 reports. This remains development
evidence, not an independent test or verified 0.855 solution.

A standard catch-up `git add`/`git commit` attempt for C5–C12 again failed
creating `.git/index.lock` with `Operation not permitted`. All source,
manifests, reports and ignored prediction binaries remain in this workspace.

## New checkpoint, training-day context

C10 is the best reproducible original-probe development CV at
**0.8749217647**, improving C9 by 0.0001408716 in all five user folds.
Only six training-only same-day context features were added to C9's matched
tree blend; all common input hashes match. Independent float32 scoring gives
0.8749217647 and saved prediction SHA-256
`189cbe747233fd31e149f000094de30645c000f9a6a9b5b2de3d75fb33a210f6`.
The reports are `experiments/SESSION-C9-training-day-meta.json` and
`experiments/SESSION-C10-blend.json`; see `SESSION_META.md` for the feature
definition and probe-label-isolation test. C10 is 0.0001123558 above
historical R5 and 0.0199217647 above 0.855.

## Adaptive factor pilot and R5 bank audit

A sampled per-user ridge correction on frozen J3 item factors was measured
against C9. Penalty 100 gives only 0.00000129 full-probe improvement when
applied to 1/16 of users, and its standalone sampled RMSE is slightly worse
than J3's baseline. It does not justify a full run; details and reproducible
commands are in `ADAPTIVE_RIDGE.md`. The historical R5 bank remains absent
under `/private/tmp/netflix-moonshot`. A fresh hash audit found five exact
inputs under other local names; they were copied to the expected paths and
reverified. **56 of 61 inputs remain unavailable**. The evidence is in
`experiments/reference/R5-input-audit-current.json` and
`experiments/reference/R5-input-audit-restored.json`. Reconstructing the
full bank requires dependency-aware model and derived-column runs. C9 remains
the best reproducible score.

## New checkpoint, fixed epoch-60 RBM

The strongest reproducible development CV is now **C9 0.8750626363**.
Replacing only C8's epoch-45 RBM column with the fixed epoch-60 continuation
improves all five user folds and lowers RMSE by 0.0002085827. Independent
float32 scoring gives 0.8750626363; the saved prediction SHA-256 is
`4404cfb596eb87c2fdb426c84cc292d243888494953e73785e63ef65a3bad435`.
All other C8 input hashes match. The training and blend evidence is in
`experiments/RBM-C7-visible-60e.json` and
`experiments/RBM-C9-60e-blend.json`. C9 is 0.0002532274 above historical R5
and 0.0200626363 above 0.855. It is still a probe development result.

A seven-day local GP scored 0.8923871307 alone, but cross-fitted one-weight
screens with C8 were slightly worse than C8. Its support columns were very
similar to the existing 30-day support columns. See
`experiments/GP-C8-7day-full-v2.json`, the two screen reports, and
`GP_COMPACT.md`. One preceding GP run failed writing to a missing output
directory; its failed manifest is retained.

Next: pivot to a more distinct model family or the complete R5 bank. Another long RBM continuation
has produced under 0.001 ensemble gain, so its compute priority is lower.

Git staging briefly accepted `PROGRESS.md`, but staging the full C5–C9 set
failed again at `.git/index.lock` with `Operation not permitted`. A valid
catch-up commit remains blocked by the sandbox; the reports and code are
saved in this workspace.

## Current checkpoint, 2026-09-30 Pacific

The best reproducible original-probe development CV is **C8 0.8752712189**
over five user folds, with independent float32 scoring **0.8752712190**.
Its out-of-fold prediction SHA-256 is
`cd839161fc771790813be45ddca86fbf6731bb9d3b58bc29a3f8a6d243af72e7`.
C8 replaces only the epoch-30 conditioned RBM prediction in the clean C7a
bank with a fixed epoch-45 continuation. The other input hashes match.
The gain is 0.0001745234, with four improved folds. Historical R5 is
0.8748094089; the target is 0.855. Probe feedback influenced model selection,
so these scores are development estimates, not independent test results.

Key evidence: `experiments/RBM-C8-45e-blend.json`,
`experiments/RBM-C7-visible-45e.json`,
`experiments/KNN-C7a-support-unique-blend.json`, and
`experiments/GP-C6-support-blend.json`. The local ignored `artifacts/` tree
holds prediction and model binaries. `RBM_CONTEXT.md`, `GP_COMPACT.md`,
`RESULTS.md`, and `STRATEGY_RESET.md` explain comparisons and failed screens.
The GP code emits three label-free support columns; its regenerated prediction
hash matches the previous column. `make moonshot-gp && make test-moonshot-gp`
passed after that change.

Commits `8496acc` and `e5643bc` preserve earlier RBM work. A prior sandbox
blocked `.git` writes and delayed committing the C5–C9 evidence. One single-file
stage succeeded, but the full stage failed again. Do not delete untracked
experiment reports or ignored model artifacts. The `RESEARCH_NEXT.md` and
DeepRecommender script/test were already untracked before this work.

Next: when quota permits, test a fixed epoch-60 conditioned RBM continuation
against a matched C8 blend, and pivot to a distinct model family or recover
the missing R5 bank if gains stay small. First preserve this working tree in
a writable Git environment and commit the verified C5–C8 gains in order.

## Earlier checkpoint, 2026-09-28

The best recorded blend is B7 at **0.8858460517** probe user-fold CV RMSE
(0.8857544990 fit RMSE). Its ten prediction columns include the frozen N2i
128-neighbor epoch-5 column. All five held-out user folds improve over B6's
0.8863207420 CV RMSE. These are development scores: base-model epoch selection
uses probe labels, so they are not an independent test result.

Completed work includes quadratic ridge blending (B5), a 256-factor joint
checkpoint and blend (J5/B6), the 128-neighbor graph and model (G2/N2), B7,
temporal neighborhood decay (N3), and paired frequency-factor and continuation
pilots (J6/J7). Experiment manifests, logs, timestamped events, source snapshots,
and the regenerated `experiments/score_timeline.csv` preserve the evidence.
`make test` passed at this checkpoint.

Next, compare N3's temporal-neighborhood prediction with B7 under the same
five user folds. Then test whether J6 adds more to the blend than J7, freezing
each prediction column before fitting weights. The saved N2i manifest identifies
the exact checkpoint used by B7; do not substitute a later N2 artifact.
