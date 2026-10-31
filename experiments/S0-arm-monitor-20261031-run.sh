#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "S0 arm monitor exited with status $?"' EXIT
while true; do
    if nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/monitor_structural_arms.py --plan /Users/quan/workspace/netflix_challenge/experiments/S0-arm-monitor-20261031-plan.json; then
        sleep 60
    else
        code=$?
        if [ "$code" = 10 ]; then exit 0; fi
        exit "$code"
    fi
done
