#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "E3 fixed treatment exited with status $?"' EXIT
mkdir -p artifacts/ordinal
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/E3-select-20261031.sha256 > artifacts/ordinal/E3-select-20261031-input-check.log
echo START_E3_FIXED_TREATMENT
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/e3_distribution_loss.py --data /Users/quan/workspace/netflix_challenge/artifacts/validation/train-matched-20260927.bin --warm-checkpoint /Users/quan/workspace/netflix_challenge/artifacts/masked/A2h-select.select.pt --warm-report /Users/quan/workspace/netflix_challenge/artifacts/masked/A2h-select.json --output /Users/quan/workspace/netflix_challenge/artifacts/ordinal/E3-select-20261031 --weight 0.0 --warm-stage selection --epochs 2 --lr 0.0003 --seed 20261019 --positives 8 --negatives 32 --threads 1 --device mps --max-batches 0 --ce-weight 0.1
echo COMPLETE_E3_RECIPE_AUDIT_AND_BOTH_P2_GATES_REQUIRED
