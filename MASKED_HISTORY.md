Masked rating-history experiments

The user authorized implementing the next architecture proposed in
[RESEARCH_NEXT.md](RESEARCH_NEXT.md). This round tests a sparse network trained
to predict ratings removed from its input. The original probe is evaluated
only after training-only epoch selection and a fresh refit.

**Implementation and controls**

`scripts/masked_history.py` implements pooled explicit and implicit histories,
64-dimensional embeddings, a 256-unit hidden layer, and a decoder evaluated
only for requested movies. All input rating summaries are recomputed after
masking. The implicit channel retains unlabeled train/probe/qualifying
interactions, consistent with the existing transductive protocol.

| Run | Explicit history | Output and training loss | Device |
| --- | --- | --- | --- |
| MH1-categorical-ce | Separate embedding per movie/star | Five probabilities, cross entropy | MPS |
| MH2-scalar-mse | Centered scalar multiplying movie embedding | Scalar, MSE | CPU, two threads |
| MH3-categorical-mse | Same categorical architecture as MH1 | MSE on expected stars | CPU, two threads |

All use the same user order and input masks, 50% input removal, hidden dropout
0.1, AdamW learning rate 0.001 and weight decay 0.001, up to 512 users/131,072
ratings per batch, and seed 20260928. Every nonempty user supplies at least
one target; users with at least two ratings retain at least one context value.
The categorical model predicts the expectation of its star probabilities.

The scalar/categorical comparison changes representation and output head.
MH1/MH3 hold the architecture fixed, but device-specific arithmetic and dropout
randomness mean they are not bitwise matched. This initial round has no
reconstruction-objective control, so any gain cannot be assigned solely to
masking. Session and second-axis contexts are later extensions.

**Selection and evaluation**

Epoch selection uses `artifacts/validation/train-matched-20260927.bin` from
[HOLDOUT.md](HOLDOUT.md). The initial launch allowed 12 epochs and patience
three. After measuring 2.5–6.5 minutes per full epoch under the existing load,
the first pilot was bounded to the first four completed selection epochs.
The amendment was recorded before any original-probe prediction was scored.
Only the explicitly identified selection trainers are stopped, and their
termination is intentional. Selection scores beyond epoch four are discarded.
That dataset's validation ratings come exclusively from original training
rows. Each arm then starts over on the original training set for the selected
number of epochs. The original probe cannot alter the resulting checkpoint
or prediction column. Reaching the epoch budget while still improving is not
evidence of convergence.

The reference is R5-meta at 0.8748094089 development CV RMSE. All 61 original
prediction/support columns were copied into `artifacts/masked/reference/` and
their hashes verified against the R5 report. The frozen configuration and
hashes are recorded in `experiments/MH-reference.json`. No in-progress model
outputs are included. `experiments/MH-round-plan.json` records the comparisons
before their scores are available.

`scripts/masked_history_compare.py` adds completed raw columns to this frozen
bank and invokes the existing residual-tree blender with the reference's
five user folds, inner epoch selection, learning rate 0.02, 8,000-tree cap,
63 leaves, minimum leaf 200, metadata, and seed 2026. It emits paired user
uncertainty and history-size slices. The existing cross-fitted blend is used
only as a comparison, never as a new training feature. Control fold 1 reproduced
the original ridge score, selected 1,300 rounds, and final 0.8744987237587174
RMSE exactly. Together with all 61 matching input hashes, this verified the
copied bank and current implementation. Remaining redundant control fits were
stopped to free resources; full comparisons use the original R5 cross-fitted
prediction file. `experiments/MH-control-verification.json` explicitly records
that this was a partial control reproduction, not a new full-probe score.

The promotion threshold is a repeatable 0.0005 ensemble RMSE gain for an
affordable component; 0.001 would justify further architecture work. These
are compute decisions. The matched training holdout is easier than the original
probe, and repeated probe research still makes the final comparison a
development estimate.

**Verification and artifacts**

`tests/test_masked_history.py` passes three tests covering masked-target value
isolation, nonzero learning across all supported losses, checkpoint reload and
query order, and complete selection/refit invariance to changed original probe
labels. The learning checks require nonzero gradients and prediction changes
from the same initialization.

Each run saves source/data hashes, configuration, epoch metrics, a reloadable
selected checkpoint, a final refit checkpoint, and atomic prediction files.
`scripts/experiment.py` records timestamped logs under `experiments/MH*`.
Large artifacts remain under ignored `artifacts/masked/`. The throughput run
MH0 uses only 20 batches and is not an architecture-quality result.

**Results (single blend seed 2026, against frozen R5-meta 0.8748094)**

| Arm | Standalone probe | Blend | Δ vs R5 | Folds better |
| --- | --- | --- | --- | --- |
| MH1 categorical, CE | 0.9227930 | 0.8744102 | −0.0003992 | 5/5 |
| MH2 scalar, MSE | 0.9171091 | 0.8744530 | −0.0003564 | 5/5 |
| MH3 categorical, MSE | 0.9174102 | 0.8741354 | −0.0006740 | 5/5 |
| All three | — | 0.8740353 | −0.0007741 | 5/5 |

MH3 alone clears the 0.0005 retention threshold on every fold; the user-cluster
95% interval on ΔMSE excludes zero for every arm. Gains concentrate on users
with fewer than 50 ratings (under-20 slice −0.0020 for MH3), the slice where
the factor/kNN bank is weakest. On identical categorical architecture MSE beat
cross entropy by 0.0003 in the blend and 0.005 standalone.

All arms were still improving at the four-epoch cap (MH3 holdout 0.8988 →
0.8958 between epochs 3 and 4), so these are lower bounds on the family.
These comparisons are against R5-meta, not the later R6-RBM5 bank (0.8731935),
which added six conditional RBM columns; the RBMs may overlap with this gain
and the next step is to re-score against R6-RBM5.
