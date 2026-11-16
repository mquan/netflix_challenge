# Direct upstream reproduction

This branch prepares a fresh training replay of [gtakacs/netflix_prize](https://github.com/gtakacs/netflix_prize/tree/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c), pinned to commit `49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. It uses the author's Rust models, hyperparameters, data ingestion, ProbeX split, and pipeline recipes. Python only orchestrates execution and independently checks the reference blend.

**The first eight-model replay is complete. Next assignment: [UPSTREAM_EXPANSION.md](UPSTREAM_EXPANSION.md).** See [UPSTREAM_PROGRESS.md](UPSTREAM_PROGRESS.md) and [COORDINATION.md](COORDINATION.md). Preserve that existing queue; the manual commands below describe the original setup and should not be used to start duplicate training.

## What has been verified

Both the unmodified upstream Rust `ridge --ensemble new` and an independent NumPy implementation reproduce **0.8561671559 development RMSE** on **1,128,164 ProbeX rows**, using **188 released columns**. All eight upstream reference rows agree within the stored tolerance of `1e-5`:

| Group | Columns | Reproduced RMSE |
| --- | ---: | ---: |
| Temporal/factor models | 69 | 0.863909370 |
| RBM | 35 | 0.869304244 |
| Neural models | 21 | 0.873265991 |
| Other | 12 | 0.875514009 |
| GBM blends | 18 | 0.858144535 |
| MLP blends | 23 | 0.857264580 |
| Feature-weighted blends | 10 | 0.856522156 |
| Final ridge ensemble | 188 | 0.856167156 |

These are **development-fitted reference scores using downloaded predictions**, not a result of our own model training, independent cross-validation, or an official Netflix test score. Intermediate learned blends can have their own cross-fitting, but the final ridge is fitted and measured on ProbeX. This protocol deliberately uses upstream's single-split `new`, rather than its `cross3` headline that uses qualifying-score feedback. Fresh training still inherits historically selected upstream hyperparameters.

Numerical details matter: upstream clips eligible input columns to `[0,6]`, leaves factor/bias columns unclipped, adds an unpenalized intercept, applies ridge lambda 1000 to feature weights, and clips predictions to `[1.05,4.95]`. The registry and dependency graph are resolved using upstream's own Rust parsers.

Committed receipts are in [experiments/upstream-repro](experiments/upstream-repro). Full binaries, downloaded fixtures, and run logs stay under ignored `artifacts/upstream-repro/`.

## Isolation and evaluation protocol

- `artifacts/upstream-repro/source/`: pinned source plus one untracked registry reader. No upstream model source changes.
- `artifacts/upstream-repro/reference/`: released **ProbeX-only** prediction fixtures and known development labels. The native reference explicitly reports qualifying evaluation skipped.
- `artifacts/upstream-repro/fresh/`: separate pinned checkout for ingestion and actual training. No fixture predictions are placed here. Before ingestion, its bundled recovered qualifying outcomes are replaced with 2,817,131 zero sentinels. Qualifying diagnostics against these zeros are meaningless; do not use them to select models.
- `fresh-receipts/`: a log and output SHA256 receipt per completed training job. Existing model outputs without execution receipts cause the wrapper to stop, preventing upstream's normal `DONE` shortcut from silently accepting imported predictions. Resume verifies completed output hashes.
- `comparisons/`: fresh-vs-released ProbeX prediction differences and scores. Thread scheduling, BLAS, and platform differences may prevent bitwise-identical training even with upstream seeds.

First compare each fresh model with its corresponding fixture, then test the gain from its residual chain. Only after these checks should we expand the model bank and retrain higher-level blends. A blend that still contains released columns must continue to be described as a hybrid. Legacy epoch-blend jobs embed coefficients fitted upstream; refit those coefficients on fresh development predictions before claiming all learned weights were fitted locally.

Do not merge this work into the running ensemble branch or modify its frozen scripts to launch this experiment.

## Start on the other computer

Use a separate checkout/worktree. On a clone that already has the ensemble branch running:

```sh
git fetch origin research/upstream-reproduction
git worktree add -b research/upstream-reproduction ../netflix_upstream_repro origin/research/upstream-reproduction
cd ../netflix_upstream_repro
python3 -m venv .venv
.venv/bin/python -m pip install numpy tomli
```

Install a current stable Rust toolchain with `rustup` if Cargo is unavailable. The source uses Rust edition 2024; the preparation machine verified Rust/Cargo 1.100.0. macOS needs Xcode Command Line Tools and Accelerate; the wrapper supplies explicit Accelerate linker flags. On Linux, upstream's `blas` feature uses OpenBLAS and requires its native build prerequisites. Higher-level LightGBM jobs additionally need their CMake/OpenMP toolchain; those jobs are outside the first eight-model replay.

Build the pinned source, then prepare a fresh training directory. If you already have the original Netflix archive, pass its **absolute path** to avoid downloading it again:

```sh
.venv/bin/python scripts/upstream_repro.py --bootstrap --build
.venv/bin/python scripts/upstream_repro.py --prepare-training \
  --archive /absolute/path/to/nf_prize_dataset.tar.gz
.venv/bin/python scripts/upstream_repro.py --setup-data
```

Omit `--archive` to let upstream's downloader obtain and checksum the original archive during setup. `--setup-data` runs upstream's own download/ingest/newsplit commands and records the known ProbeX label hash. Setup output is in `artifacts/upstream-repro/fresh-setup.log`. Setup and base training are explicit actions; preparing the branch does not launch them.

Download just the pinned development fixtures and verify both reference implementations:

```sh
.venv/bin/python scripts/upstream_repro.py --prepare-reference \
  --index experiments/upstream-repro/reference-index.toml \
  --ratings-npy artifacts/upstream-repro/fresh/data/probex/ratings.npy \
  --python-reference --native-reference
```

The reference index contains the 188 ensemble development arrays plus two DNN parent fixtures, with pinned size and MD5. The `rbmx2-40` base has no released fixture in the pinned store; its comparison reports unavailable. This is roughly 0.85 GB of float32 predictions; upstream documents about 3.3 GB for ingested datasets. Saved training predictions, epochs, latent features, compilation, and expanded blends require additional disk space.

## First fresh jobs

The initial eight jobs cover three model families and their residual chains:

```text
tsvdx4-60
tsvdx4-60__knns
rbmx2-40
rbmx2-40__knn3
dnn-24
dnn-24__knn3
dnn-24__dnn-16p50
dnn-24__dnn-16p50__knn3
```

Print a plan without training:

```sh
.venv/bin/python scripts/upstream_repro.py --plan-job dnn-24__dnn-16p50__knn3
```

After the current ensemble run finishes, launch the three chain endpoints **sequentially**, followed by the separate direct DNN residual job. The wrapper runs missing model parents first and resumes completed jobs using receipts:

```sh
export UPSTREAM_REPRO_THREADS=8
.venv/bin/python scripts/upstream_repro.py --run-job tsvdx4-60__knns
.venv/bin/python scripts/upstream_repro.py --run-job rbmx2-40__knn3
.venv/bin/python scripts/upstream_repro.py --run-job dnn-24__dnn-16p50__knn3
.venv/bin/python scripts/upstream_repro.py --run-job dnn-24__knn3
```

Run inside `tmux`; on macOS use `caffeinate` as needed. The default is two compute threads to coexist with other work; the example raises this to eight for the larger machine. BLAS defaults to one thread to limit nested parallelism; override `UPSTREAM_REPRO_BLAS_THREADS` separately if measured timings justify it. These upstream recipes use native Rust CPU code, including their neural models. They do not automatically use the GPU. More threads are not a guarantee of faster or identical training.

Compare a completed job:

```sh
.venv/bin/python scripts/upstream_repro.py --compare-model tsvdx4-60
.venv/bin/python scripts/upstream_repro.py --compare-model tsvdx4-60__knns
```

Compare all eight completed jobs similarly. Preserve receipts and logs even if a result disappoints. A failed job's partial outputs are intentionally preserved and block silent retries; inspect and move them aside explicitly before retrying. Do not run concurrent wrappers against the same fresh directory.

This first pass proves whether we can retrain representative upstream models and their chains closely enough to justify expanding. It does **not** regenerate all 188 inputs or guarantee 0.856167 from eight models. The resolved full pipeline contains 201 jobs, including data setup and intermediate outputs; a full replay is a much larger CPU experiment. We have no measured fresh-training runtime yet and should estimate it from the first completed model rather than reuse the unrelated ensemble run's 5–8 hour estimate.

## Preparation checks

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p test_upstream_repro.py -v
```

For the unattended worker and local verification assignments, see [COORDINATION.md](COORDINATION.md).

The tests check dependency ordering (including implicit file producers), cycle rejection, outcome-zero enforcement, refusal of unreceipted existing predictions, and detection of altered completed outputs. The native build and both real-data reference reproductions are separate integration checks.
