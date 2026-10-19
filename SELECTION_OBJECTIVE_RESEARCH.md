# Choice-prediction objective: cumulative A2 fine-tuning

Prepared on `codex/local-context`, 2026-10-19. The promoted score remains
0.8668706425; no score gain is claimed by this preparation. LC1–LC6 are closed.

## Hypothesis and relationship to N1

N1 adds unrated choice history as input to the rating encoder. SO1 instead
tests whether an auxiliary task predicting observed movie identities changes
the shared representation in a way that helps rating prediction. It starts
from the existing A2h-select epoch-12 checkpoint; it does not restart the base
family from scratch or depend on N1's modified input tokens.

[Vernade and Cappé (2015)](https://arxiv.org/abs/1509.09130) motivate using the
relationship between movie selection and ratings. This experiment is our
masked-reconstruction adaptation, not a reproduction of their estimator or
evidence that their results transfer to this benchmark.

## Fixed first comparison

The frozen inputs, commands and gates are in `experiments/SO1-plan.json`.
Run the two arms sequentially only after the active N1 chain has finished or
stopped and current process inspection confirms GPU availability. N1's chain
can proceed from selection into a full refit, so completion of its selection
process alone does not free the GPU. No SO1 full-data job has been launched.

`scripts/selection_objective_queue.py` provides the waiting runner. It requires
N1's chain status to be terminal and two process snapshots 30 seconds apart
with no other Python executable active. This conservative check also waits
for unknown Python training jobs. It verifies the frozen plan, inputs and
runner before each arm, runs control then treatment and the fixed gate, and
stops on failure. It never promotes a result or starts the full-data refit.
The original SO1 plan remains immutable; a separate queue launch record binds
its digest and commands. Three targeted queue tests pass. A process check
cannot reserve the GPU against an unrelated future launch, so other research
must continue to avoid starting competing jobs once this queue takes over.

The queue was launched at 2026-10-19 08:29 UTC (PID 67792), and live process
inspection confirmed it is waiting for N1. The immutable launch evidence is
`experiments/SO1-queue.launch.json`; live status is `SO1-queue.status.json`.
Neither training arm has started. Do not edit the frozen runner or plan while
this process remains active, or launch another copy.

- Same A2h-select checkpoint, training-only P2 dataset, initialization seed,
  rating masks, auxiliary examples, optimizer restart, dropout stream, batch
  order and two complete fine-tuning epochs in both arms.
- Learning rate 0.0003, seed 20261019, existing A2h architecture and other
  optimizer settings. Main objective remains rating MSE.
- Control auxiliary weight 0; treatment weight 0.1. Both perform the auxiliary
  forward and backward pass, including the zero-weight control. Epoch 2 is the
  fixed scoring artifact; epoch 1 is diagnostic, not another selection option.
- Eight uniformly sampled unique observed movies per user (or all if fewer),
  plus 32 uniformly sampled unobserved movies with replacement. Negatives
  exclude all known train/probe/qualifying interactions. Positive and negative
  terms each receive half the user's auxiliary weight; users receive equal
  weight within a batch. Uniform complement sampling requires no popularity
  importance correction. The loss is balanced observation discrimination,
  **not** calibrated observation probability and not a rating for missing pairs.
- For the auxiliary pass, remove the positive movie IDs, their dates and their
  ratings from explicit histories, implicit histories, time features and
  summaries before building inputs. There are no candidate query tokens in
  this pass. A USER-slot code predicts candidates through a separate linear
  head. The normal rating pass retains its original inference semantics.

The two passes share the encoder but release the rating graph before building
the auxiliary graph. This limits peak activation memory. It does add work per
step; actual MPS throughput is not measured yet. Sampled gradient norms at the
code projection and first attention block show whether the weighted auxiliary
loss affects the shared model. A lower auxiliary loss alone is not success.

## Evaluation and continuation

`scripts/selection_objective_gate.py` verifies frozen source/data/checkpoint
hashes and both complete run reports before scoring. It uses all P2 rows and
the eleven frozen rehearsal columns: fit ridge penalties on user folds 2–4,
select from {1, 10, 100, 1000} on fold 1, refit on folds 1–4, evaluate on fold 0.

Primary comparison: bank11 + fine-tuned control + **duplicate control** versus
bank11 + fine-tuned control + **treatment**. Both have thirteen columns, so an
identical treatment has exactly zero gain. The replacement comparison,
bank11 + control versus bank11 + treatment, is diagnostic.

Require at least **0.0005 P2 RMSE gain** in the primary comparison before a
full-T fine-tuning pair. If passed, warm-start both full-T arms from the retained
A2h-refit checkpoint, preserving the two-epoch recipe. The trainer's
`--warm-stage refit` binds this run to that report's original training dataset.
Then run the unchanged full-bank blend comparison. Promotion still requires
all five development folds to improve; substantial continuation requires
at least 0.0003 RMSE gain. Do not tune the auxiliary weight or select a different
epoch after reading fold 0. P2 has already been reused for model selection;
this gate is development evidence, not an untouched confirmation set.

The next iteration must follow measured behavior: whether auxiliary gradients
reach the encoder, whether rating quality changes, and whether that change adds
information beyond the existing bank. A failed fixed two-epoch recipe does not
justify an unbounded weight/epoch search. Preserve its diagnostics and decide
which specific failure mechanism a follow-up would test.

## Validation before training

Eight targeted tests pass: held-positive ID/rating/date exclusion, observed-set
negative exclusion and weighting, empty context, auxiliary learning and shared
gradients, exact warm rating predictions, fixed-training query-label isolation,
matched RNG consumption, complete CLI checkpoints and no overwrite, zero gain
for identical columns, and evaluation-label isolation of ridge fitting and
selection. Several contracts are covered within one test method.

Additional compatibility checks reproduce the archived A2h trainer's predictions
byte for byte on small and windowed fixtures and on 27 actual P2 queries from
eight short-history users. This is bounded compatibility evidence, not a
full-P2 re-emission. Evidence: `experiments/SO0-contract-verification.json`.

A synthetic CPU run also completed both two-epoch arms and the frozen-input
gate end to end for 100 users / 200 queries. Its evidence is
`experiments/SO0-pipeline-smoke.json`; its scores are fixture values, not Netflix
results. Test logs are preserved in `experiments/SO0-training-tests.log` and
`experiments/SO0-gate-tests.log`.
