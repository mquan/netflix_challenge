#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
shasum -a 256 --check experiments/short-first-20261108/inputs.sha256
while true; do
    if /Users/quan/.cache/netflix-venv/bin/python experiments/short-first-20261108/barrier.py; then
        sleep 30
        shasum -a 256 --check experiments/short-first-20261108/inputs.sha256
        if /Users/quan/.cache/netflix-venv/bin/python experiments/short-first-20261108/barrier.py --release; then exit 0; fi
    else
        code=$?
        if [ "$code" != 1 ]; then exit "$code"; fi
    fi
    sleep 30
done
