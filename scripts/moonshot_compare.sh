#!/usr/bin/env bash
# Full-probe paired ablation using the existing five-context, five-user-fold blender.
set -euo pipefail
DATA=${1:?usage: moonshot_compare.sh DATA GP_PREFIX OUTPUT_NAME}
GP=${2:?missing GP prefix}
NAME=${3:?missing output name}
controls=(joint64 joint128-reg005 joint128-day128 neighborhood64 knn64-j128r005 knn64-j128day)
args=(ensemble --data "$DATA" --contexts 1)
for col in "${controls[@]}"; do args+=(--prediction "artifacts/snapshot/$col.f32"); done
build/netflix "${args[@]}" --predictions "artifacts/$NAME-control.f32" > "experiments/$NAME-control.log"
for variant in 0 1 2 3; do
  build/netflix "${args[@]}" --prediction "$GP.$variant.f32" \
    --predictions "artifacts/$NAME-$variant.f32" > "experiments/$NAME-$variant.log"
done
build/netflix "${args[@]}" --prediction "$GP.0.f32" --prediction "$GP.1.f32" --prediction "$GP.2.f32" \
  --predictions "artifacts/$NAME-all.f32" > "experiments/$NAME-all.log"
