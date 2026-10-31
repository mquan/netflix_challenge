#!/usr/bin/env bash
# E1 (NEXT_EXPLORATIONS.md): Stage C retrain of K1 + its two kNN columns with
# P2 rows up-weighted. Same commands as Phase 0's P0-B-K1 / P0-B-knn* runs
# (scripts/rehearsal_chain.sh), plus --row-weights. The epoch (41) is the
# P2-selected one from Phase 0; the original probe chooses nothing here.
#
# Deliberately bash, not scripts/experiment.py: the active SO1 queue waits for
# a moment with no other Python process, and a Python wrapper alive for hours
# would hold it back. Provenance is written by this script instead.
#
# usage: scripts/e1_chain.sh BINARY ARM [ARM ...]   (ARM = w1 | w2 | w3)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$1; shift
FULL=data/processed/netflix.bin
E=artifacts/rehearsal/e1
LOGS=experiments
K1_ARGS=(--factors 256 --learning-rate .008 --regularization .02 --decay .97 --iterations 41 --fixed-epoch --threads 5)
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

record() {  # name status seconds command...
  local name=$1 status=$2 seconds=$3; shift 3
  {
    printf '{\n  "name": "%s",\n  "returncode": %s,\n  "elapsed_seconds": %s,\n' "$name" "$status" "$seconds"
    printf '  "binary_sha256": "%s",\n  "joint_hpp_sha256": "%s",\n  "netflix_cpp_sha256": "%s",\n' \
      "$(sha "$BIN")" "$(sha src/joint.hpp)" "$(sha src/netflix.cpp)"
    printf '  "git_head": "%s",\n  "finished_utc": "%s",\n  "command": "%s"\n}\n' \
      "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" "$*"
  } > "$LOGS/$name.json"
}

step() {  # name command...
  local name=$1; shift
  if [ -e "$LOGS/$name.json" ] || [ -e "$LOGS/$name.log" ]; then echo "refusing to overwrite $name" >&2; exit 1; fi
  local start=$SECONDS status=0
  nice -n 10 "$@" > "$LOGS/$name.log" 2>&1 || status=$?
  record "$name" "$status" "$((SECONDS-start))" "$@"
  [ "$status" = 0 ] || { echo "$name failed ($status)" >&2; exit "$status"; }
}

for arm in "$@"; do
  W=$E/p2-${arm}.f32
  [ -f "$W" ] && [ -f "$W.json" ] || { echo "missing $W" >&2; exit 1; }
  step "E1-B-K1-$arm" "$BIN" joint --data $FULL --model $E/B-K1-$arm.bin --predictions $E/B-K1-$arm.f32 \
    "${K1_ARGS[@]}" --row-weights "$W"
  step "E1-B-knn256td-$arm" "$BIN" knn --data $FULL --graph artifacts/r4/neighbors256.bin \
    --joint-model $E/B-K1-$arm.bin --neighbors 256 --power 1 --shrink .5 --time-decay .003 \
    --predictions $E/B-knn256td-$arm.f32 --support $E/B-knn256td-$arm.support.f32
  step "E1-B-knn64-$arm" "$BIN" knn --data $FULL --graph artifacts/neighbors64.bin \
    --joint-model $E/B-K1-$arm.bin --power 1 --shrink .5 --predictions $E/B-knn64-$arm.f32
done
echo "E1 chain done: $*"
