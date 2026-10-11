---
status: in-progress
branch: main
timestamp: 2026-10-11
best_score: 0.8668706425
score_metric: five-fold held-out-user probe blend RMSE
---

# Netflix Prize progress checkpoint

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
