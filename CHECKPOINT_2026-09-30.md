---
status: in-progress
branch: research/dataset-enrichment
timestamp: 2026-10-01T05:12:32Z
files_modified: [PROGRESS.md, RESULTS.md, README.md, STRATEGY_RESET.md, RBM_CONTEXT.md, GP_COMPACT.md, C5_FOLLOWUPS.md, scripts/experiment.py, src/moonshot_gp.cpp, tests/test_moonshot_gp.py, experiments/]
---

# Netflix score progress

## 2026-10-01 continuation: C12

C11 adds two training-only same-day joint-residual features to C10 and scores
0.8748606495 development CV. A fixed equal average of C10 and C11 scores
**C12 0.8747780516**, 0.0000313573 below historical R5 and 0.0197780516
above the 0.855 target. Saved float32 SHA-256:
`93129d524dcd2749d7fc7ccdaf32ee03277f7b2a942035805a0c3dd07eed6940`.
The reproducible command is `scripts/fixed_average.py`; input/output hashes
and fold scores are in `experiments/SESSION-C12-fixed-half.json`.
`SESSION_META.md` records the matched C11 comparison and the leakage caveat
for the unused fitted-weight screen. This is the current best reproducible
development result. The goal remains unfinished.
A standard catch-up `git add`/`git commit` attempt again failed creating
`.git/index.lock` with `Operation not permitted`; the changes remain saved
locally but uncommitted.

## Latest measured gain: C10

Six training-only same-day context features added to C9 produce C10 at
**0.8749217647** original-probe five-user-fold development CV. All five
folds improve by a total 0.0001408716 RMSE. Independent float32 scoring
agrees. Saved SHA-256:
`189cbe747233fd31e149f000094de30645c000f9a6a9b5b2de3d75fb33a210f6`.
The reports are `experiments/SESSION-C9-training-day-meta.json` and
`experiments/SESSION-C10-blend.json`. `SESSION_META.md` has details.
C10 is 0.0199217647 above 0.855. The goal remains unfinished.

## Latest continuation: model pivot and bank recovery

A per-user adaptive ridge correction on frozen J3 factors was implemented and
screened on a fixed 1/16-user sample. Its best cross-fitted full-probe gain
over C9 was only 0.00000129; `ADAPTIVE_RIDGE.md` has the run and report paths.
The sample does not justify a full run. A fresh R5 hash audit found five exact
inputs under other local filenames. Those files were restored at the paths
expected by the historical R5 manifest and reverified. The restored audit
shows **5 exact, 56 unavailable** of 61 required inputs at
`experiments/reference/R5-input-audit-restored.json`. C9 remains the best
reproducible local development score. A standard `git add` retry still fails
on `.git/index.lock` with `Operation not permitted`.

## Follow-up, fixed epoch-60 continuation

C9 improves C8 to **0.8750626363** development CV. All five user folds
improve. Independent float32 scoring agrees; saved SHA-256:
`4404cfb596eb87c2fdb426c84cc292d243888494953e73785e63ef65a3bad435`.
Reports: `experiments/RBM-C7-visible-60e.json` and
`experiments/RBM-C9-60e-blend.json`. The seven-day GP experiment and two
screens under `experiments/GP-C8-7day-*` found no incremental linear signal.
The goal remains unfinished; C9 is 0.0200626363 above 0.855.
A single-file `git add PROGRESS.md` succeeded, but staging the full C5–C9 set
again failed at `.git/index.lock` with `Operation not permitted`. The catch-up
commit remains blocked by the sandbox.

## Summary

Latest measured original-probe five-user-fold development CV: **0.8752712189**
(C8). Independent float32 scoring: **0.8752712190**. Historical R5:
0.8748094089. Target: 0.855. The active goal is unfinished.

## Decisions and evidence

- C5 added a regenerated local GP predictor to C4: 0.8759027924.
- C6 added three GP support features: 0.8755711017.
- Clean C7a added one unique kNN graph-support feature: 0.8754457423.
- C8 replaced only C7a's epoch-30 conditioned RBM input with its fixed
  epoch-45 continuation: 0.8752712189. Four of five folds improved. The
  out-of-fold SHA-256 is
  `cd839161fc771790813be45ddca86fbf6731bb9d3b58bc29a3f8a6d243af72e7`.
- The first C7 fit duplicated the same kNN support input and scored
  0.8754260110. Keep C7a as the clean comparator.
- Exact reports: `experiments/GP-C5-blend.json`,
  `experiments/GP-C6-support-blend.json`,
  `experiments/KNN-C7a-support-unique-blend.json`, and
  `experiments/RBM-C8-45e-blend.json`. `PROGRESS.md` has fuller context.
- The GP prediction hash was unchanged after support generation.
  `make moonshot-gp && make test-moonshot-gp` passed. `git diff --check` passed.

## Remaining work

The fixed epoch-60 RBM continuation is complete and recorded above. Commit
the saved verified gains, then investigate a distinct model family or
reconstruct the missing R5 input bank.
Keep model artifacts and untracked experiment reports intact.

## Environment notes

An earlier sandbox allowed workspace file writes but only read access to `.git`.
`git add` failed on `.git/index.lock` with `Operation not permitted`. A later
single-file stage succeeded, but the full stage failed with the same error.
The context-save skill's normal `~/.gstack` checkpoint root is outside the
writable roots, so this checkpoint is stored in the repository workspace.
Commits `8496acc` and `e5643bc` already preserve earlier RBM work. The
`RESEARCH_NEXT.md` and DeepRecommender script/test predate this work as
untracked files. Probe-driven model selection means these are development
scores, not independent test results.
