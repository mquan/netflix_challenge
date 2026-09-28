# Netflix Prize reconstruction

This project targets the **original Netflix Prize dataset and split**, not the
later PRML 2019 classroom contest. It implements selected BellKor-style models,
not the complete winning ensemble.

## Validation

The primary local metric is RMSE on Netflix's labeled probe set. Probe ratings
are physically excluded from the training CSR during ingestion. The unlabeled
qualifying pairs are retained only as implicit-feedback events, matching the
published BellKor definition of `N(u)`.

```sh
make download   # resumable download from the Internet Archive
make verify     # checks Netflix archive MD5
make extract
make ingest
make baseline
build/netflix eval-baseline \
  --data data/processed/netflix.bin \
  --model artifacts/baseline.bin
make time-baseline
build/netflix eval-time-baseline \
  --data data/processed/netflix.bin \
  --model artifacts/time-baseline.bin
# Advanced checkpoints (longer-running)
make svdpp-f32
make svdpp
make time-svdpp
make svdpp-seed42
make blend
```

The published checkpoints to keep in mind are:

- Cinematch: 0.9514 quiz RMSE
- BellKor time-aware baseline: about 0.9278 on its reported competition split
- BellKor timeSVD++ variants: about 0.8762–0.8914 reported quiz RMSE
- winning private test submission: 0.856704 RMSE

Probe, quiz, and private test scores are similar but not interchangeable. We
will tune only on the probe data. Any later evaluation against recovered
quiz/test labels must be a final, isolated audit—not a training signal.

## Current implementation

- verified/resumable original-data download
- deterministic parser for all 100,480,507 dated ratings
- exact probe removal and local RMSE evaluation
- implicit-feedback CSR containing train, probe, and qualifying interactions
- regularized user/movie bias baseline
- frequency-aware temporal baseline with item time bins, user drift,
  per-user-day effects, and time-varying rating-scale effects
- SVD++ residual factorization with train/probe/qualifying interactions used as
  unlabeled implicit feedback

Local development checkpoints (probe labels excluded from model gradients,
but used for epoch selection):

- static regularized baseline: `0.98256528`
- 40-epoch temporal baseline: `0.93506871` (best observed epoch: `0.93491601`)
- 32-factor SVD++: `0.90267216` (best observed epoch: 7)
- 64-factor SVD++: `0.90079729` (best observed epoch: 7)
- 64-factor post-hoc temporal drift: `0.90055257` (best observed epoch: 5)
- 32/64-factor blend: `0.90019305`
- three-model SVD++/timeSVD++ blend: `0.89970985`

- jointly trained temporal SVD++, 128 factors: `0.89617220`
- seven-model contextual ridge ensemble: `0.89013424` development CV
- add learned item-neighborhood predictions: `0.88666465` development CV
- residual item kNN on every joint model, conditional RBM, denoising
  autoencoder, and a LightGBM blender with meta-features:
  **`0.88040209` development CV** (see `RESULTS.md`)

The older `time-svdpp` command freezes its static model before fitting drift;
it is not a jointly optimized timeSVD++ implementation. The new `joint` command
jointly trains temporal biases, user/item factors, implicit feedback and user
factor drift. Optional per-user-day factors model short-lived preferences.

`scripts/gbdt_blend.py` needs `numpy` and `lightgbm`; `scripts/autorec.py`
needs `numpy` and `torch` (MPS is used when available). Install them in a
virtualenv; the C++ tool has no extra dependencies.

## Joint training and reproducible experiments

```sh
make test
python3 scripts/experiment.py --name my-joint128 \
  --strategy 'Joint temporal and implicit model, 128 factors, regularization .02' -- \
  build/netflix joint --data data/processed/netflix.bin \
    --model artifacts/my-joint128.bin --factors 128 --iterations 40 \
    --regularization .02 --decay .95 --patience 6 \
    --predictions artifacts/my-joint128.f32
build/netflix eval-joint --data data/processed/netflix.bin \
  --model artifacts/my-joint128.bin --predictions artifacts/reloaded.f32
```

Experiment names must be new. Each run records its strategy, exact command,
source/binary hashes, input hashes, epoch log and output hashes in `experiments/`.
Artifacts and original data remain local and untracked. Logs measure wall time,
which can include computer sleep. Models retain the best probe epoch.
`--init-model` is a warm start: it resets the learning-rate schedule, random
generator and epoch counter, so it is not an exact interrupted-run resume.
`--day-factors 128` adds about 9 GB of parameters on the original data; leave it
at zero for lower memory use.

Raw prediction files are float32 columns in processed probe order. Only blend
columns from this exact dataset and row order; do not read a column while its
trainer is updating it. For example, with the recorded completed artifacts:

```sh
build/netflix ensemble --data data/processed/netflix.bin \
  --prediction artifacts/legacy.a.f32 --prediction artifacts/legacy.b.f32 \
  --prediction artifacts/legacy.c.f32 --prediction artifacts/joint64.f32 \
  --prediction artifacts/joint128-reg005.f32 \
  --prediction artifacts/joint128-reg02.f32 \
  --prediction artifacts/joint128-day128.f32 \
  --contexts 1 --predictions artifacts/recheck-b3.f32 \
  --weights artifacts/recheck-b3.weights
```

`blend_cv_rmse` fits ridge weights on four user folds and scores the fifth,
repeated for all five folds. `blend_fit_rmse` fits and scores on all probe rows.
The output column holds cross-fitted predictions; its `.fit` companion holds
full-fit predictions. Contexts let weights vary with training-history size,
movie popularity, date and same-day activity. These features contain no probe
rating values. **CV is still a development estimate:** base-model checkpoint
selection uses the whole probe. It is not an untouched final-test evaluation.

## Structurally different model: learned neighborhoods

```sh
build/netflix graph --data data/processed/netflix.bin \
  --graph artifacts/neighbors64.bin --neighbors 64 --history-cap 256
build/netflix neighborhood --data data/processed/netflix.bin \
  --graph artifacts/neighbors64.bin --time-model artifacts/time-baseline.bin \
  --model artifacts/neighborhood64.bin --predictions artifacts/neighborhood64.f32 \
  --iterations 12
build/netflix eval-neighborhood --data data/processed/netflix.bin \
  --graph artifacts/neighbors64.bin --time-model artifacts/time-baseline.bin \
  --model artifacts/neighborhood64.bin --predictions artifacts/neighborhood-reloaded.f32
```

This is an approximation to the learned neighborhood family, not a full
reproduction of BellKor's integrated model. Graph discovery uses training-only
bias-adjusted residual cosine with histories capped for compute. Explicit and
implicit neighbor weights are then learned on all training ratings against a
frozen temporal baseline. The graph, dataset and original temporal baseline
must accompany the neighborhood checkpoint when evaluating it.
Its success criterion is improvement when added to the existing ensemble,
not necessarily a better standalone score.

See [RESULTS.md](RESULTS.md) for the experiment-by-experiment score log,
strategy summaries, and next-step decisions.

For score over time, run `python3 scripts/score_timeline.py` and plot
[`experiments/score_timeline.csv`](experiments/score_timeline.csv).
Historical timestamps have explicit provenance/uncertainty; new experiment
runs timestamp every output line. The CSV also supports elapsed time minus
recorded idle/rate-limit/timeout intervals, without assuming unknown gaps are
idle. See [timing and activity accounting](experiments/TIMING.md).
