# Qualifying path for a verified matching86-tree/context-NN improvement

The complete development gate passed November 18 Pacific at
**0.8600300601883686**, with all five folds, replay and independent scoring
verified. Implementation and testing of the separate production adapter remain
the next required work; it has not launched.

This is a conditional plan, not a launched production run. The matching-tree
study must finish all five folds, replay all saved trees bit-for-bit, verify the
unchanged reused neural vector and fixed average, and independently score the
complete result. Its aggregate RMSE must be strictly below the current verified
0.8600637147575518. Individual folds remain diagnostics; neither all-fold
improvement nor a fixed minimum gain is required. No recovered qualifying scores
or residuals choose this candidate.

The existing first85 qualifying attempt remains unchanged. Its original temporal
phase must complete all25 epochs, exports and paired-receipt verification before
the missing four bank columns are admissible. A previously verified target pass
on both recovered groups makes further production work unnecessary.

If the complete matching-tree candidate improves, use a separate production
adapter, report and artifact directory, preserving the existing frozen files.
The intended adapter is `scripts/upstream_cf12_matching_production.py`; it does
not exist or run yet. Both production families use the exact paired86-predictor
bank built by `upstream_cf12_bank.py`:162 ridge columns,222 nonlinear features
and the full-known-training ridge input as feature223.

| Component | Actual development-budget source | Production action |
| --- | --- | --- |
| Tree | Five matching-tree receipts under `experiments/upstream-cf12-tree/tree/` | Fit three new full-known-ProbeX trees on the86-predictor inputs |
| Context NN | Five original CF12 context receipts under `experiments/upstream-cf12-context/nn/` | Fit the unchanged223-input context model, using median budgets19/20/20 |

Use the existing per-seed median-budget function and original model parameters.
Bind each family's actual originating manifest, fold receipts, checkpoints,
complete replay and independent scoring evidence. Do not synthesize NN receipts
in the matching-tree directory: those neural fits were reused from the preceding
experiment. Likewise, the first85 production trees cannot substitute for a new
86-predictor tree fit. Preserve explicit input, code, library and source hashes.

Reuse the unchanged production core for ridge fitting on all known ProbeX rows,
family training, semantic replay, clipping, fixed equal float32-family average,
frozen final vector and independent recovered-group scoring. Verify the223-input
context checkpoint loader in the new adapter and exercise actual small tree/NN
fits and replay before launching it. Validate split shapes and both families'
budget-source bindings, including rejection of changed source artifacts.

Only after the complete development gate passes may the older CF12 production
queue be superseded. First verify that its actual process is idle, owns no
fitting child and has created no production plan or artifacts; archive its launch
and queue state before stopping that owned idle controller. No native training
worker is signalled. Keep the existing shared fitting lock, memory admission,
resource guards and first-audit target-pass gate. If the older production path
has already begun, preserve it and serialize the new path rather than replacing
an active fit.

Completion remains strict RMSE below 0.85725 on both community-recovered groups,
with a frozen prediction vector, model replay, own-training provenance and
independent full-row scoring. A development improvement or successful process
does not complete the goal, and recovered groups are not official or untouched
Netflix Test evidence.
