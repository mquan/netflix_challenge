#!/usr/bin/env bash
set -euo pipefail
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON=${MOONSHOT_PYTHON:?set Python with NumPy and PyTorch}
NAME=${1:-NN1-blend}
args=(--data /Users/quan/workspace/netflix_challenge/data/processed/netflix.bin --output "artifacts/$NAME" --hidden 16 --epochs 60)
for col in joint64 joint128-reg005 joint128-day128 neighborhood64 knn64-j128r005 knn64-j128day; do
  args+=(--prediction "artifacts/snapshot/$col.f32")
done
for variant in 0 1 2; do args+=(--prediction "artifacts/gp-full.$variant.f32"); done
args+=(--prediction artifacts/gp2-persistent-full.0.f32 --prediction artifacts/local-global.epoch6.f32 --prediction artifacts/local-local.epoch6.f32)
if [[ ${2:-} == rbm ]]; then args+=(--prediction artifacts/snapshot/rbm100-fast.f32); fi
"$PYTHON" -u scripts/moonshot_nn_blend.py "${args[@]}" > "experiments/$NAME.log" 2>&1
