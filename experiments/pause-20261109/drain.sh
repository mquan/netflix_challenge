#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
while true; do
    if /Users/quan/.cache/netflix-venv/bin/python experiments/pause-20261109/control.py --drain; then
        exit 0
    else
        code=$?
        if [ "$code" != 1 ]; then exit "$code"; fi
    fi
    sleep 30
done
