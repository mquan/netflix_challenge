#!/usr/bin/env bash
# Three predeclared structural bets, sequentially and with bounded CPU use.
set -euo pipefail
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATA=/Users/quan/workspace/netflix_challenge/data/processed/netflix.bin
PYTHON=${MOONSHOT_PYTHON:?set MOONSHOT_PYTHON to a Python with NumPy}
controls=(joint64 joint128-reg005 joint128-day128 neighborhood64 knn64-j128r005 knn64-j128day)
for bet in transient persistent implicit; do
  case "$bet" in
    transient) options=(--time-scale 30); strategy='Transient GP: covariance decays over 30 days, testing session-dependent tastes.' ;;
    persistent) options=(--time-scale 30 --persistent .5); strategy='Two-timescale GP: equal persistent and transient covariance, testing long- and short-term tastes together.' ;;
    implicit) options=(--implicit-mix 1); strategy='Selection geometry GP: use implicit-feedback embeddings instead of explicit movie factors.' ;;
  esac
  python3 scripts/experiment.py --name "GP2-$bet-pilot" --strategy "$strategy" -- \
    build/moonshot_gp --data "$DATA" --model artifacts/snapshot/joint128-reg005.bin \
    --output "artifacts/gp2-$bet-pilot" --neighbors 32 --user-modulo 10 "${options[@]}"
  args=(--data "$DATA" --mask "artifacts/gp2-$bet-pilot.mask.f32" --output "experiments/GP2-$bet-pilot-eval.json")
  for col in "${controls[@]}"; do args+=(--control "artifacts/snapshot/$col.f32"); done
  for variant in 0 1 2; do args+=(--control "artifacts/gp-full.$variant.f32"); done
  for variant in 0 1 2 3; do args+=(--candidate "artifacts/gp2-$bet-pilot.$variant.f32"); done
  "$PYTHON" scripts/moonshot_evaluate.py "${args[@]}" > "experiments/GP2-$bet-pilot-eval.log"
done
