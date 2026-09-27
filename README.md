# Netflix Prize reconstruction

This project targets the **original Netflix Prize dataset and split**, not the
later PRML 2019 classroom contest. It is building a leakage-safe reconstruction
of the published BellKor approach.

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

Verified local checkpoints on the leakage-free probe split:

- static regularized baseline: `0.98256528`
- 40-epoch temporal baseline: `0.93506871` (best observed epoch: `0.93491601`)
- 32-factor SVD++: `0.90267216` (best observed epoch: 7)
- 64-factor SVD++: `0.90079729` (best observed epoch: 7)
- 64-factor timeSVD++ drift: `0.90055257` (best observed epoch: 5)
- 32/64-factor blend: `0.90019305`
- three-model SVD++/timeSVD++ blend: `0.89970985`

The next model stages are timeSVD++, the neighborhood residual model, and
leakage-safe stacking.

See [RESULTS.md](RESULTS.md) for the experiment-by-experiment score log,
strategy summaries, and next-step decisions.
