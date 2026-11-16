# Run the local ensemble experiment on a second computer

Branch: **`research/independent-ensemble`**. Use a separate clone on Linux or
macOS (or Linux under WSL). Python 3.10–3.12, Git, and internet access are
needed. Downloading/ingesting the raw dataset additionally needs `curl`,
`make`, and a C++17 compiler. Allow roughly 15 GB of free disk space for inputs,
raw-data extraction, dependencies and outputs; a CUDA PyTorch installation can
need additional space. The 100 GB RAM machine can run five tree folds together.

The default assignment is a fresh baseline and the combined snapshot/calibration
treatment, plus three-seed subset-network stacking. This concentrates the second
machine on the most complete proposed treatment. The Mac continues its existing
run; its output directory is separate. All inputs are predictions trained in
our research repository. No upstream public prediction bank or recovered
qualifying labels is used.

## 1. Get the branch and environment

```sh
git clone --single-branch --branch research/independent-ensemble \
  https://github.com/mquan/netflix_challenge.git netflix-independent
cd netflix-independent
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-local-ensemble.txt
```

Use a Python 3.10–3.12 executable in the venv command. For NVIDIA GPUs, install
the appropriate CUDA-enabled PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/) if
the default wheel does not detect your GPU. The prepared run records actual
package versions and refuses version changes on resume. PyTorch backend/RNG
differences are handled with a fresh same-machine, same-device control.

## 2. Prepare verified inputs

If you copy this Mac's `data/processed/netflix.bin` to the second computer:

```sh
python scripts/setup_local_ensemble_remote.py --dataset /absolute/path/to/netflix.bin
```

Otherwise download the original Netflix archive and build that file locally:

```sh
python scripts/setup_local_ensemble_remote.py --download-data
```

The setup restores our exact 79-column bank, 46 support/latent columns,
12 A2h snapshots and control predictions from pinned commit
`18390f4bf778e305c940c62756226d1ea792778a` on `codex/household-personas`.
It verifies the dataset and prediction hashes and reproduces the saved
0.8668706424 development score before training. Restoring predictions needs
hundreds of MB of Git data. The raw-data path takes additional download and
ingestion time. Do not rerun setup after preparation; resume the worker below.

## 3. Verify and launch

```sh
python -m unittest discover -s tests -p test_local_ensemble_isolation.py -v
python -m unittest discover -s tests -p test_local_ensemble_device.py -v
python scripts/run_local_ensemble_remote.py --dry-run
```

The dry run prints `cuda`, `mps`, or `cpu` and the exact commands. To keep
running after disconnecting, use `tmux` or run in the background:

```sh
nohup python -u scripts/run_local_ensemble_remote.py \
  --arms combined --device auto --fold-workers 5 --tree-threads 3 --nn-threads 4 \
  > remote-ensemble.log 2>&1 &
```

This allows one multi-fold CPU tree job and one neural job to overlap. Neural
jobs use CUDA first, then the Apple GPU, then CPU. Only one neural job occupies
the GPU at a time. Five tree workers × three threads uses up to 15 CPU threads;
reduce either setting on a smaller CPU. GPU memory requirements depend on the
backend; use `--device cpu` if the GPU cannot fit the feature table. The worker
fails explicitly on allocation errors rather than silently switching backends.

To require an NVIDIA GPU, replace `--device auto` with `--device cuda`; use
`--device mps` for an Apple GPU. To run every treatment on this machine, use
`--arms snapshots calibration combined` instead of `--arms combined`.

Inspect progress:

```sh
cat artifacts/independent-ensemble/remote/status.json
tail -n 20 remote-ensemble.log
ls artifacts/independent-ensemble/remote/logs/
```

On a laptop, prevent idle sleep while training. After an interruption, rerun
the launch command with the **same arguments**. Completed reports are verified
and skipped. A report without a successful stage receipt requires inspection;
the worker refuses to overwrite it. Changing device, worker settings, arms,
code or package versions requires a new `--output` directory and a separately
prepared input plan if code/packages changed.

## 4. Return the results

```sh
tar -czf independent-ensemble-results.tar.gz \
  artifacts/independent-ensemble/remote \
  artifacts/independent-ensemble/plan.json
```

Send that archive back, or provide `remote/comparison.json` and `status.json`
for an initial review. Keep the full outputs for independent rescoring and
combining predictions. Do not call the result a qualifying score: this is a
matched original-Probe development experiment on historically selected models.

## Timing evidence

On the M1 Pro, short synthetic tests measured roughly 6× NN2 and 3.8× ENN
throughput on its GPU versus two CPU threads. A 131k-row tree benchmark was
about 1.8× faster with three threads than one. These are microbenchmarks,
not promised end-to-end speedups or measurements of the second computer.
