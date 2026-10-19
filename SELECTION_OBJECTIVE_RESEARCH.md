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

## Real-P2 initial gradient and optimizer audit

While N1 owns the GPU, `scripts/selection_gradient_audit.py` inspected the first
four consecutive batches from SO1's frozen epoch-1 sampler. Their mean training
history lengths were about 2,015, 992, 744 and 35 ratings per user. The original
A2h warm checkpoint, random choice-head initialization, weight 0.1, sampling,
masking and training-mode dropout were retained. This diagnostic ran on CPU
with `pad_rows=1` to remove unused padded users and bound memory. It applied no
optimizer step to the audited model and never scored P2 validation labels.

The first pass found that the weighted auxiliary gradient norm on the two
tensors logged by SO1 was only 1.0–2.2% of the rating gradient norm. That alone
does not show a negligible optimizer effect. A second pass repeated the same
gradients and compared actual first AdamW steps on temporary parameter copies.
The following comparisons all use the **same 37 parameters active in both
objectives**, rather than mixing parameter subsets:

| Batch | Users | Auxiliary/rating gradient norm | Change in first update / control update norm | Control/treatment update cosine |
|---|---:|---:|---:|---:|
| 1 | 10 | 1.25% | 26.30% | 0.9668 |
| 2 | 16 | 1.13% | 26.74% | 0.9657 |
| 3 | 21 | 1.98% | 25.56% | 0.9684 |
| 4 | 441 | 1.18% | 18.04% | 0.9840 |

Neither arm would trigger the global gradient clip in these batches. The
auxiliary gradient reaches the shared encoder and changes its initial optimizer
step measurably. Therefore **do not increase the frozen weight merely because
the raw gradient norm looks small**. These checks do not establish useful new
rating information or ensemble improvement. Every optimizer comparison starts
with zero moments independently; it is not a multi-step trajectory. CPU
dropout/padding also prevent a claim of exact MPS update or throughput identity.

Evidence is retained in `experiments/SO1-gradient-audit-result.json`,
`experiments/SO1-update-audit-result.json`, and
`experiments/SO1-gradient-verification.json`, with both experiment logs and source
archives. All 21 original frozen inputs were rechecked. Model parameter hashes
are identical before and after both audits; all four batches' losses and raw
gradient measurements repeat exactly across the two CPU runs. The queued SO1
script, plan, weight, epochs and promotion gates remain unchanged.

## Checkpoint continuation capability (SC0)

The frozen SO1 trainer writes its learned choice head, but its `warm_model`
loader accepts only a rating-only checkpoint. It also omits optimizer and RNG
state. Consequently, its output cannot be fed directly back through that loader
for a cumulative iteration, and exact optimizer resumption cannot be recovered
from an original SO1 checkpoint.

`scripts/selection_objective_continue.py` supplies a separate continuation path
without changing the queued trainer. It strictly loads the complete learned
state, including `choice_output` and `choice_bias`, and binds the checkpoint to
its parent's recorded hash, configuration, dataset, libraries and training source.
An explicit `--optimizer-mode restart` is required for legacy SO1 checkpoints.
The existing seed, learning rate, objective weight and sampling settings are
inherited, while fine-tuning epoch numbers continue from the recorded parent.
New output prefixes preserve the prior stage and its evidence.

New checkpoints additionally save optimizer, CPU RNG and, on MPS, device RNG
state. `--optimizer-mode resume` restores those states and requires the same
device and continuation source. An epoch-boundary checkpoint with a completed
record can be loaded even if its parent stage did not finish its entire planned
budget; the next stage's additional epoch count must be fixed explicitly.
Unknown query labels produce predictions but no validation RMSE.

Four CPU fixture tests pass. The strongest runs two continuation epochs
uninterrupted, then compares against one epoch followed by a resume in a **fresh
process**: model tensors, optimizer state, RNG state and prediction bytes match
exactly, with dropout enabled. Tests also verify preservation of the learned
choice head and rejection of missing resume state, changed inputs, device changes
during exact resume, and output overwrites. Evidence and archived sources are in
`experiments/SC0-continuation-verification.json` and
`experiments/SC0-continuation-tests.log`. All 24 queue launch bindings remain
unchanged. MPS resume equivalence has not been tested while N1 owns the GPU.

This capability does **not** extend SO1's frozen two-epoch comparison or authorize
post-hoc epoch selection. No Netflix continuation was launched. A subsequent
experiment needs its own fixed budget, matched control/treatment continuation
policy and gate; the existing SO1 gate continues to evaluate only the original
two-epoch outputs.

## Conditional full-training comparison (SF0)

`scripts/selection_objective_full.py` and `experiments/SO1-full-plan.json` prepare
the next score-producing stage without launching it. Preparation verified 166
frozen inputs and reproduced the retained score as 0.8668706424233039 from its
saved predictions. The six stages are: two fixed two-epoch fine-tuning arms from
the full-training A2h-refit checkpoint, then tree and neural blends for each of
the duplicate-control and treatment banks. Both banks retain the original 79
prediction columns and append two columns. Their additions are control/control
and control/treatment respectively; duplicate columns are intentionally preserved.
All other blend settings remain fixed, with three CPU fold workers.

Execution requires the original P2 gate to pass its 0.0005 minimum. The runner
recomputes that gate, checks arm hashes and selected penalties, and freezes the
resulting evidence. Before each stage it verifies inputs and waits for N1's chain
to be terminal and two idle Python snapshots. Reports, prediction lengths,
finiteness, library versions and saved blend scores are checked after execution.
Failures stop the chain and preserve artifacts for explicit recovery. Existing
outputs or status records cannot be silently overwritten or restarted.

The comparison records treatment versus duplicate-control, treatment versus the
retained reference, and control versus that reference. Substantial auxiliary
investment requires at least 0.0003 RMSE gain in both treatment comparisons and
improvement on every user fold. A useful control-only gain is also reported.
No result is automatically promoted. This is original training with the probe
held out, not final train+probe production or a hidden-test score.

Seven CPU fixture tests pass, covering gate rejection and recomputation, no
training launch on missing/failed evidence, matched stage construction, stopping
on validation failure, paired decision rules, and blend artifact validation.
Evidence is archived under `experiments/SF0-full-comparison-tests-v2.*` and
`experiments/SF0-full-plan.*`. A real full-training chain has not run.

The prepared plan binds the current R9 reference. **Refresh it before execution
if N1 promotes a different reference bank.** Once the P2 gate passes and the
reference is confirmed current, the direct launch command is:

```sh
/Users/quan/.cache/netflix-venv/bin/python scripts/selection_objective_full.py run --plan experiments/SO1-full-plan.json
```

Do not wrap that driver in `experiment.py`: its conservative idle check would
wait for the wrapper Python process. The driver wraps its individual stages
itself. No additional waiting process has been launched, because it would also
block the existing SO1 queue.

## Iteration-speed diagnosis: row padding

The synchronous encoder path builds a dense `[rows, width, d_model]` tensor,
rounding user rows up to multiples of 64 before every transformer block. The
padding mask protects semantic attention, but the dense tensor and attention
shapes still include those rows. SO1 pays this cost for both rating and
auxiliary passes; the prefetch thread overlaps only CPU input construction.

`scripts/selection_padding_audit.py` replays exact CPU preparation RNG streams
for the first 128 of 6,168 frozen SO1 epoch-1 shuffled batches (8,927 users).
There is no model, gradient, optimizer or GPU work. Actual batch sizes range
from 10 to 512 users and padded widths from 32 to 1,536. Relative to keeping
only actual user rows at each observed width, the current policy multiplies
dense token slots by 2.43 for rating / 2.68 for auxiliary, and the dense
attention-size estimate by 3.64 / 4.32 respectively.

| Row-padding multiple | Rating attention-size multiplier | Auxiliary multiplier | Rating / auxiliary distinct shapes |
|---|---:|---:|---:|
| 64, current | 3.644 | 4.316 | 34 / 32 |
| 32 | 2.011 | 2.274 | 37 / 36 |
| 16 | 1.337 | 1.397 | 41 / 41 |
| 8 | 1.249 | 1.321 | 47 / 48 |
| 1 | 1.000 | 1.000 | 78 / 82 |

**First benchmark candidate: padding to 16 rows.** Across these two passes it
uses 2.87 times fewer dense attention cells and 2.07 times fewer dense token
slots than 64. These are analytical workload proxies, not measured speedups,
kernel operations or allocated attention matrices. Confidence is high in the
shape counts and unestablished in MPS wall-time benefit. More distinct shapes
can increase compilation/cache costs. A broader dynamic batching redesign is
not justified before this small comparison is timed.

Compatibility is behavior-visible for training: a CPU fixture confirms that
the actual encoder receives the predicted shapes, evaluation outputs agree
within 1e-6, but the training RNG state differs after the forward pass. Thus
this must not be inserted silently into an existing learning comparison.
**No optimization is applied; the queued SO1 script and plan remain frozen.**
Future adoption requires an explicitly fixed matched recipe plus an idle-MPS
benchmark of representative real batches, including warm-up/compilation,
sustained time, peak memory and shape-cache behavior. CPU input agreement is
not MPS equivalence, and this sample is not the full epoch.

The experiment/source archive, shape report and independent arithmetic check
are `experiments/SO1-padding-audit.*` and
`experiments/SO1-padding-audit-verification.json`. One CPU contract test passes;
all 24 live queue bindings are unchanged. This finding addresses iteration
cost, not the remaining score gap. The `optimize-performance` skill was used
for this audit; no skill updates are suggested.
