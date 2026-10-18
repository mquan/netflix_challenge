# Local rating-path pilot

2026-10-12. Implements the retrieval and cheap signal tests proposed in
[RESEARCH_RESET.md](RESEARCH_RESET.md). These are training-only P2 experiments,
not original-probe ensemble scores. The promoted ensemble remains
R9-LAT-A2h at 0.8668706425 development CV.

## Frozen pilot recipe

`scripts/local_paths_pilot.py` reads the existing matched P2 dataset, SHA-256
`94932220829f3ed80e1396688ba5a670f61dd6a44ff0f89a551e1e7fae9b44b8`.
All P2 ratings are physically absent from its training arrays. Original probe
ratings are not in this dataset's training or query arrays.

For each sampled query `(u, i, day)`:

1. Take the 32 training movies of `u` nearest the query date, breaking ties by
   source row. Reject a query if its target movie occurs in that user's training.
2. Uniformly sample at most 256 training raters of `i`, using a deterministic
   seed from the run seed and query identifiers. A global inverted index is
   built once; retrieval does not scan all raters of popular movies per query.
3. Look up their ratings of those 32 history movies. Retain at most 32 neighbors
   with nonempty intersections, ranked by overlap count, target-rating date
   proximity, and user ID. Retrieval never uses rating values.
4. Preserve the sparse paths `(rating(u,j), rating(v,j), rating(v,i))` and
   summarize their 125 rating-triplet counts, normalized by path count.

The simple matched user-kNN adjusts each neighbor's target rating by that
neighbor's mean difference from the user over shared movies. Weights are
`overlap / (overlap + 5) * exp(-mean_squared_rating_difference / 2)`.
No-path queries fall back to smoothed user plus item means minus global mean.

Two regularized linear rating predictors use exactly the same retrieval:

- **Pooled + kNN:** smoothed means, the matched kNN prediction, support counts,
  separately pooled rating distributions, and date-gap summaries (27 features).
- **Pooled + kNN + paths:** the control plus 125 triplet histogram cells and
  eight agreement/support summaries (160 features total).

Both choose ridge penalty from `{1, 10, 100, 1000}`. Existing user folds 2–4 fit,
fold 1 selects, and fold 0 evaluates after refitting on folds 1–4. Feature
normalization is fitted only on the corresponding fit users. The new learners
predict ratings directly; there are no in-sample residual targets or pretrained
embeddings. Reported uncertainty is a normal 95% interval on paired MSE
differences clustered by evaluation user.

## LP1: 20,000 queries

Seed 20261012. Index construction: 2.50 seconds. Extraction: 15.00 seconds,
1,333 queries/second. Index arrays: 0.782 GB; process peak RSS: 3.36 GB.
Linear extrapolation gives about 17.6 minutes for all 1,407,566 P2 queries,
excluding fitting, serialization, and any neural encoder.

All sampled queries have at least one path. Mean retained support is 573 paths
and 31.98 neighbors; retained intersection density is 62.87%. This is density
after selecting the strongest-overlap neighbors, not density of an arbitrary
32-by-32 subgraph.

| Predictor | P2 evaluation RMSE |
|---|---:|
| Matched simple user-kNN | 0.95672455 |
| Pooled + kNN ridge | 0.94513202 |
| Pooled + kNN + paths ridge | 0.94290946 |

The delta is −0.00222255 on 4,049 rows / 3,976 evaluation users. The paired
MSE interval is [−0.01338372, +0.00499117], which includes zero. Users with
at most 20 training ratings get worse by +0.00427710 RMSE (621 rows).
This supports a larger pilot, but does not establish a reliable gain.

## LP2: 200,000 queries and feature ablations

Seed 20261013; retrieval, features, split rules, and penalty grid unchanged.
Index construction took 2.57 seconds and extraction 161.29 seconds (1,240
queries/second). Peak RSS was 3.34 GB. Coverage was 99.9925%, with mean 573.59
paths and 62.93% retained density. Projected full P2 extraction is 18.9 minutes.

Evaluation uses 39,887 rows from 33,109 users. Both full predictors select
penalty 100 on the selection users.

| Predictor | P2 evaluation RMSE | Change from pooled + kNN |
|---|---:|---:|
| Matched simple user-kNN | 0.95425570 | +0.01103322 |
| Pooled + kNN ridge | 0.94322248 | — |
| Add agreement summaries only | 0.94037156 | −0.00285092 |
| Add triplet histogram only | 0.93772198 | −0.00550050 |
| Add histogram and agreement summaries | **0.93753599** | **−0.00568649** |

The full predictor's paired MSE interval is [−0.01310225, −0.00828757].
Users with at most 20 training ratings improve by −0.00411564 RMSE (5,736
evaluation rows); the remaining users improve by −0.00597608 (34,151 rows).
The initial sparse-user regression does not persist in this larger sample.

The attribution run, performed after LP2 on its same saved features, shows
that the triplet histogram adds −0.00283557 RMSE beyond the agreement-augmented
control. Its paired MSE interval is [−0.00716227, −0.00348760]. Thus the observed
gain is not explained entirely by the eight agreement/support summaries.
This is development attribution, not a separate validation sample, and a
nonlinear pooled control could still close the gap.

**Decision:** retrieval is feasible and the cheap summary test is positive.
Proceed next to the matched joint-versus-pooled structural test from the reset
plan. No original-probe columns, full-bank blends, or neural graph training were
run here, and there is no ensemble promotion.

## Reproduction and artifacts

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/quan/.cache/netflix-venv/bin/python scripts/local_paths_pilot.py \
  --data artifacts/validation/train-matched-20260927.bin \
  --output artifacts/local-paths/new-run --queries 20000 --seed 20261012

/Users/quan/.cache/netflix-venv/bin/python tests/test_local_paths_pilot.py

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/quan/.cache/netflix-venv/bin/python scripts/local_paths_ablation.py \
  --run artifacts/local-paths/LP2-p2-200k \
  --output experiments/new-path-ablation.json
```

The actual runs use `scripts/experiment.py`; exact commands and timestamped
logs are in `experiments/LP1-p2-paths-20k.*` and
`experiments/LP2-p2-paths-200k.*`. Reports include dataset, source, and output
hashes. Local `sample.npz` files retain query indices, features, support metrics,
and evaluation predictions in `artifacts/local-paths/LP1-p2-20k/` and
`artifacts/local-paths/LP2-p2-200k/`. The compact reports are also copied to
`experiments/LP1-p2-paths-20k.report.json` and
`experiments/LP2-p2-paths-200k.report.json`; attribution is recorded in
`experiments/LP2-path-ablation.json`.
The generic experiment wrapper's metric description is not applicable here:
the pilot's own report correctly identifies P2 as the evaluation dataset.

Tests verify exact sparse path identities and histogram cells, numerical kNN
weights, target-edge rejection, no-path behavior, deterministic bounded
retrieval independent of ratings, probe-label invariance, and evaluation-label
isolation from both regularization selection and final predictions.

## Limits and decision criteria

P2 is easier than the original probe; these absolute scores cannot be compared
to the promoted ensemble. The larger LP2 sample can overlap LP1 and is a
development follow-up, not an independent final test. A gain against the pooled
control may reflect signal already represented in the existing predictor bank.
The histogram also discards neighbor identity: it does not test a joint
subgraph encoder or establish a causal benefit from graph topology.

A structural follow-up needs a matched joint-versus-pooled encoder comparison
and a measured structure-destroying ablation. Any eventual promotion still
requires the frozen R9 raw-bank blending recipe, all five original-probe folds,
paired user uncertainty, and the existing all-fold improvement rule.
