# Ordered rating-history experiments

Started on `research/dataset-enrichment` after the enrichment review. The three
alternatives ran sequentially so GPU workloads did not compete. The queue
completed successfully; its final state is in
`experiments/history-queue-status.json`.

| Order | Run | Question |
|---|---|---|
| 1 | HN1-history | Does a masked rating-history encoder add predictive information? |
| 2 | HN2-sparse | Does emphasizing users with little training history improve the same architecture? |
| 3 | HN3-session | Does a target-date-local history representation add information beyond the global history? |

These are bounded pilots, not full reproductions of published systems.
[CF-NADE](https://arxiv.org/abs/1605.09477) motivates random history contexts
and a five-rating distribution. The implementation uses separate item/rating
embeddings, a shared encoder and a factored softmax decoder. Cross-entropy plus
0.2 times the ranked probability score trains the output. This is not CF-NADE's
exact ordinal objective or full autoregressive likelihood.

Each training batch samples a disjoint context and target subset for every
user. At most 128 context ratings and 16 target ratings are used per user per
epoch. An epoch visits every active user, but does not score every training
rating. Sampling and caps make this a user-oriented pilot objective, not a
uniform pass over all 99 million labeled rows. Evaluation uses a deterministic
sample of at most 128 training ratings per user. No probe rating enters context,
item-prior initialization or gradients. Probe RMSE selects checkpoints and
early stopping, so results are development estimates.

HN2 changes only user loss weights to
`clip(sqrt(50 / training_history_size), 0.5, 3)`.
HN3 starts independently from the same seed and adds a learned transform of
context embeddings weighted by `exp(-abs(target_day-context_day)/30)`.
It uses normal user weights, so it can be compared directly to HN1. This tests
date-local rating context; it does not assume ordering within a day or reproduce
a strictly causal deployment setting. Our original competition protocol permits
the available training ratings on either side of a query date.

All arms use rank 64, hidden width 128, dropout 0.15, AdamW at 0.001 with weight
decay 0.0001, batch size 256, seed 2026, at most 20 epochs and patience 4.
Sparse/session additions are exploratory architectural bets; no gain is promised.

## Control and interpretation

The newer R5 artifact bank and previously described unfinished RBM/AutoRec
checkpoints are absent from this checkout and the old temporary locations.
The available ten-column B7 bank is therefore frozen and hashed. Before any
long training, the queue reproduces its quadratic contextual ensemble and
compares predictions with `artifacts/blend-b7.f32` to tolerance 2e-6.
The historical score is 0.8858460517. This is a weaker control than R5 at
0.8748094089; an improvement here cannot establish an improvement over R5.

After each training run, the original C++ ensemble is refitted in its five
user folds with that candidate added to the same ten columns. Assessments
report paired overall/fold RMSE, user histories below 20, 20–49 and at least
50, and descriptive user-cluster MSE intervals. Compare HN2/HN3 against HN1
as well as against B7. Intervals do not adjust for checkpoint/model selection.
No new candidate is automatically promoted to the project best.

## Completed pilot results

The frozen B7 control reproduced at **0.8858461** RMSE. Each row adds one
history prediction column to the same ten-column bank and refits the original
quadratic ensemble using five held-out user folds.

| Added column | Standalone probe RMSE | Ensemble CV RMSE | Change from B7 |
|---|---:|---:|---:|
| HN1 masked history | 0.9229317 | 0.8838617 | −0.0019844 |
| HN2 sparse weighting | 0.9273574 | 0.8839329 | −0.0019132 |
| HN3 date-local context | 0.9166124 | **0.8830615** | **−0.0027845** |

HN3 beats HN1 by 0.0008002 RMSE in the matched ensemble comparison. Its
descriptive paired user-cluster interval for the MSE difference (HN3 minus HN1)
is [−0.0015713, −0.0012563]. HN3 improves on HN1 in all five folds and in
each reported user-history slice, including users with fewer than 20 ratings.
HN2 is 0.0000712 RMSE worse than HN1 in the ensemble, so the sparse weighting
change did not help this pilot. The checkpoints for all three arms came from
epoch 20, the configured cap; this does not establish convergence.

These are repeatedly selected original-probe development results. The interval
does not correct for checkpoint or architecture selection, and the user folds
only separate ensemble fitting rows. The strongest recorded result in the
project record, R5 at **0.8748094**, remains substantially better, but its
artifact bank is missing here. Thus HN3 is the most promising of these three
history variants, without evidence that it improves R5. Recover and verify the
exact R5 inputs before spending more compute on this family or promoting a
new project best.

## Operations

`scripts/history_queue.py` stops on any failed command or changed frozen bank.
It refuses duplicate launches when a status file already exists. A failed queue
requires inspection rather than automatic retries or overwriting evidence.

```sh
cat experiments/history-queue-status.json
tail -20 experiments/history-queue.log
```

The status file contains the supervisor PID, final stage, completed results
and errors. Per-run logs, source archives, commands, data/binary hashes and
timing are captured by `scripts/experiment.py`. Models/predictions live under
`artifacts/history/`; assessment JSON files live in `experiments/`.
Checkpoints save parameters/configuration, not
optimizer/RNG state, and do not support exact training resume.

The project `.venv` contains NumPy 2.2.6 and PyTorch 2.14.0. The queue uses
MPS; this environment requires running outside the sandbox to access the GPU.
LightGBM was installed, but its import lacks the system OpenMP library; this
queue uses the C++ blender and does not depend on LightGBM.

## Verification before launch

- Three tests in `tests/test_history_model.py` pass: disjoint sampled targets,
  probe-label independence and row-order/batch-size consistency, and masked
  ordinal loss/backpropagation.
- HN0-timing completed 100 GPU training batches in 6.2 seconds and full-probe
  evaluation within 24.3 seconds total for the epoch. Its RMSE 1.0230062 is
  a timing pilot result, not a trained candidate result.
- Both new scripts compile and `git diff --check` passes.

To run the tests: `.venv/bin/python -m unittest discover -s tests -p test_history_model.py`.
