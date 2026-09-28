#!/usr/bin/env bash
# Round 6: residual kNN columns for a converged joint checkpoint.
# usage: scripts/round6_columns.sh NAME   (expects artifacts/NAME.bin)
# Writes artifacts/r6/knn256td-NAME.f32 (+ .support.f32) and knn-NAME-p1.f32,
# matching the R4 recipes (neighbors256 + time decay .003; neighbors64 power 1).
set -euo pipefail
name=$1
data=data/processed/netflix.bin
mkdir -p artifacts/r6
build/netflix knn --data $data --graph artifacts/r4/neighbors256.bin \
  --joint-model artifacts/$name.bin --neighbors 256 --power 1 --shrink .5 --time-decay .003 \
  --predictions artifacts/r6/knn256td-$name.f32 --support artifacts/r6/knn256td-$name.support.f32
build/netflix knn --data $data --graph artifacts/neighbors64.bin \
  --joint-model artifacts/$name.bin --power 1 --shrink .5 \
  --predictions artifacts/r6/knn-$name-p1.f32
