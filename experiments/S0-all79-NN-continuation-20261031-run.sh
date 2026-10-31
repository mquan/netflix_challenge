#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "S0 NN continuation exited with status $?"' EXIT
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/S0-all79-NN-continuation-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/structural/S0-scoping-20261031/all79-neural.log.hash-check
echo START_all79_neural_unchanged
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/round8_nn_blend.py --data /Users/quan/workspace/netflix_challenge/data/processed/netflix.bin --like /Users/quan/workspace/netflix_challenge/artifacts/structural/S0-scoping-20261031/results/all79-s3.json --output /Users/quan/workspace/netflix_challenge/artifacts/structural/S0-scoping-20261031/results/all79-NN2 --hidden 128 --layers 2 --dropout 0.2 --lr 0.0003 --weight-decay 0.01 --batch 4096 --epochs 60 --patience 5 --seeds 2026 7 13 --threads 2 > /Users/quan/workspace/netflix_challenge/artifacts/structural/S0-scoping-20261031/all79-neural.log 2>&1
echo COMPLETE_ALL79_PARITY_VERIFICATION_REQUIRED
