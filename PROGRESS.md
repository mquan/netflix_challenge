---
status: fresh-only-reproduction-active
branch: research/full-upstream-reproduction
timestamp: 2026-11-19T09:24:00+00:00
best_score: 0.8599745622436025
score_metric: adaptive five-fold known-ProbeX RMSE using own freshly trained predictors
qualifying_status: audited_miss
qualifying_target_per_recovered_group: 0.85725
---

# Netflix Prize progress checkpoint

## Current fresh-only reproduction (2026-11-19 UTC)

The first complete own-training qualifying audit **misses** the target:

| Recovered source group | Rows | Independently verified RMSE | Required |
| --- | ---: | ---: | ---: |
| source0 (`is_test=0`) | 1,408,789 | **0.8619187880311255** | <0.85725 |
| source1 (`is_test=1`) | 1,408,342 | **0.8628067765436453** | <0.85725 |

The worse group improves **9.4166113865%** over the Cinematch reference 0.9525,
short of 10%. The remaining gap is 0.0055567765 RMSE. These are adaptive
community-recovered outcomes, not official or untouched Netflix Test scores.
The published Quiz/Test sizes reverse the source labels, so source groups
are preserved without relabeling.

This result uses our freshly trained 85-predictor bank and full-known-ProbeX
blend fitting. Public source code is reused; no downloaded pretrained prediction
features or qualifying outcomes entered fitting or selection. Both saved full
families replay bit-for-bit, their fixed float32 mean verifies, and independent
compensated scoring covers all 2,817,131 qualifying rows. The separate completion
check binds 333 evidence files, 29 own source jobs and the actual development
models behind the per-seed budgets. It explicitly records `goal_achieved=false`.

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
| Matching86 production | All three full tree fits and saved-tree replay complete; neural refit PID14336 active under master88587. Its exact contextNN86 is required by the new combination |
| Fixed 91-predictor study | Complete: both families replayed; fixed mean 0.8601526901 is worse and not promoted |
| New tree91/contextNN86 production | Controller 10865 waits for the active matching86 run; it will verify and reuse the exact same full-fit NN86, fit tree91 with its own ridge and budgets, replay both, freeze the mean and audit |
| All91 context follow-up | Controller 10867 waits behind both production runs and two physical-memory admission samples; it now binds the stronger 0.8599745622 development reference |
| Original1,000-factor model | Native canonical 20-epoch job started with 16 logical chunks/four physical workers; no reduced capacity |
| Local temporal150 phase | Complete at08:48 UTC; all31 native NPY outputs match the original remote receipt exactly, including fulltrain residuals; all eight selected split vectors equal the recovered own pairs |
| CFNADE96 | Checkpoints33/31 preserved; both workers yielded after measured paging. Temporal PID46684 has exited and the original bank is complete; automatic resume still requires physical-footprint/compressor/swap admission |
| CFNADE48 | Remote live-process heartbeat 09:06 UTC (commit96b5f93) reports 64/62 epochs and zero swap use; deadline 14:33 UTC, return 15:03 UTC |
| AEX64/AEX128 | Completed; own AEX128 safely imported at an idle scheduler boundary |
| Primary DAG | Original temporal60 PID77902 has logged all20 fulltrain-phase epochs; final outputs are not yet receipted |

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
