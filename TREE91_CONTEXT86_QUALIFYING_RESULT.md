# Own tree91/contextNN86 qualifying result

The frozen fixed mean scores **0.8614776640541453** on source0 (1,408,789 rows)
and **0.8624021195492101** on source1 (1,408,342 rows). Both improve slightly
on the preceding matching86 audit. Both still miss the strict0.85725 target;
the worse group improves9.45909506% over Cinematch0.9525. The remaining gap is
0.0051521195 RMSE. These are adaptive community-recovered groups, not official
Netflix or untouched Test scores.

The tree was freshly fitted on the paired91 bank, with its own167-column ridge
and228 inputs. Budgets873/746/855 rounds for seeds2026/7/13 come from its actual
five nested development receipts. The context NN is our exact preceding full-fit
86-predictor model: its162-column ridge,223 inputs,19/20/20 epochs, original bank,
source, versions, model set and replay evidence all verify. Its publication
receipt explicitly records no new neural fit; its original saved model inference
was replayed again in a clean process. No different-schema model was substituted.

The float32 half mean was frozen before recovered labels were read. Its SHA256 is
`4d86f33ebe17810d0b6f94aec158240e75f7c848a856b1d439308dbd45fb2a19`.
Both family semantic replay receipts verify. A separate checker rehashed408
frozen source inputs and28 code files, checked the model/replay bindings and
exact mean, and independently scored both groups against the canonical label
Git blob using compensated summation. This check verifies the existing inference
replays rather than rerunning inference itself.

The development mean0.8599745622436025 used the earlier cached own source-vector
realization. The paired production bank has nine changed predictor columns from
four independent own source retrains, as disclosed and quantified separately.
All actual production-column transforms verify, and known labels/user identities
match. The development score is recipe evidence, not a measured production-bank
score. No downloaded pretrained predictions or qualifying outcomes fit or select
this combination. The feature layouts, budgets and half mean were fixed using
known-ProbeX evidence before the qualifying audits.

Evidence: `experiments/upstream-tree91-context86-production/` contains the frozen
plan, fresh-tree receipt, explicit reused-NN receipt, both semantic replays,
recovered audit and `frozen-mean-score-verification.json`.
