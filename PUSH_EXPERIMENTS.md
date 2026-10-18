# Controlled paper follow-up experiments

Worktree: `/private/tmp/netflix-paper-impl`, branch `paper-implementations`.
Base merge: `054d08b` combines the BAR exploration with `round4-combined`.
All scores below use the original 1,408,395-row Netflix Probe holdout.
They are development scores, not the competition's private test score.

## Frozen baseline

`experiments/PUSH-baseline.json` records the complete R4-B configuration and
SHA256 hashes. Its 51 prediction columns and five support columns are copied
under `artifacts/push/bank`. The previous CV RMSE is **0.8758047393**.
The replay uses identical column order, user folds, inner early stopping,
contextual ridge, and residual LightGBM parameters. The new file paths and
number of concurrent fold processes do not change the statistical protocol.

## Experiments

- **Reference factor model:** unmodified `TxModel` from
  [gtakacs/netflix_prize](https://github.com/gtakacs/netflix_prize), commit
  `855c7404c02f55072acbb0e9c8d45be55fa608cf`; authored `tsvdx4-60` configuration
  (60 factors, 20 epochs, seed 42). `scripts/reference_tx.rs` adapts our binary
  data directly. Train ratings exclude Probe; masked heldout identities include
  Probe and qualifying. Output preserves our Probe order. No qualifying labels
  are loaded. Full source/data hashes are in `PUSH-reference-provenance.json`.
- **RBM visible context:** zero-initialize user, user-day, and movie-frequency
  softmax biases on the frozen epoch-30 RBM200 checkpoint. Compare with five
  epochs of identical continuation without context. Both use two threads and
  the same sampled CD chains/seeds, CD schedule, and existing weight rates.
  User/day bias rates are .0025/.008; frequency uses minibatch reduction with
  effective rate .00004, context regularization .001. New context rates anneal
  from their introduction. This preserves our minibatch implementation; it is
  not an exact reproduction of the paper's fully online frequency updates.
- **Temporal neighborhood:** continue frozen I2R epoch 23 for five epochs,
  using the same 64-neighbor graph and optimizer in both arms. Add
  `exp(-beta_user * abs(target_day - neighbor_day))` to explicit and implicit
  neighbor contributions. Initialize beta to zero, constrain nonnegative, and
  learn at 1e-7 with regularization .01. Existing matching-neighbor normalization
  is retained. Graph size is fixed to isolate the temporal change.

Each experiment selects its best epoch on Probe. The blend cross-validation
therefore remains development CV even though blend weights and tree fitting
exclude each outer fold's users.

## Verification

- `make test` and `python3 tests/test_push.py`.
- `tests/check_temporal_neighbor.cpp`: finite-difference derivative and a
  nonzero learning update; also verifies that 1e-7 CLI defaults survive parsing.
- Zero context rates reproduce the legacy RBM predictions exactly.
- Zero decay rate reproduces static-neighborhood predictions exactly.
- Changing Probe labels leaves one-epoch model parameters and predictions
  unchanged for both new models and for the reference adapter.
- Both new checkpoint formats reload to byte-identical prediction files and
  still read their old checkpoint formats.

An initial temporal launch exposed a pre-existing parser bug: converting a
floating-point default with `std::to_string` rounded 1e-7 to zero. It was stopped
and preserved as `PUSH-neighbor-temporal`; use `PUSH-neighbor-temporal-v2` for
results. The parser now returns numeric defaults directly.

## Reproduction

Exact model commands, source hashes, binary hashes, and timestamped output are
in `experiments/PUSH-*.json`, `.log`, and `.events.jsonl`. The reference adapter
is copied to upstream `src/bin/our-probe.rs` and built with:

```sh
cargo build --release --bin our-probe --manifest-path /private/tmp/netflix-reference/Cargo.toml
```

Run all frozen blend ablations with a Python environment containing NumPy and
LightGBM (the measured environment uses NumPy 2.5.3 / LightGBM 4.7.0):

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/push_blends.py
```

The script waits for successful model completion before consuming a prediction
file; each blend's exact command is recorded separately. Completed outputs are
skipped. Large models and prediction files remain in ignored `artifacts/push`.
