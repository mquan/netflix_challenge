#!/usr/bin/env bash
# Original six components + first GP family are the fixed reference for round two.
set -euo pipefail
DATA=${1:?dataset required}
NAME=${2:?output name required}
shift 2
args=(ensemble --data "$DATA" --contexts 1)
for col in joint64 joint128-reg005 joint128-day128 neighborhood64 knn64-j128r005 knn64-j128day; do
  args+=(--prediction "artifacts/snapshot/$col.f32")
done
for variant in 0 1 2; do args+=(--prediction "artifacts/gp-full.$variant.f32"); done
for col in "$@"; do args+=(--prediction "$col"); done
build/netflix "${args[@]}" --predictions "artifacts/$NAME.f32" > "experiments/$NAME.log"
