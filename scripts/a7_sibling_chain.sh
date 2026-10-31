#!/usr/bin/env bash
# A7 (HEADROOM.md): sibling-split Stage C rehearsal, K1 family. Retrains on
# T + probe half A (artifacts/sibling/T-plus-A-v2.bin) and scores probe half B,
# with the Phase 0 recipes: graphs (256 and 64 neighbours, history cap 256),
# K1 (256 factors, lr .008, reg .02, decay .97, 41 epochs fixed; 2 threads to respect the session thread cap, Hogwild noise only),
# time-decayed 256-kNN (+support) and 64-kNN. Epoch 41 is the P2-selected
# Phase 0 epoch; probe half B selects nothing.
# Bash on purpose (GOAL.md rule 3); provenance JSON per step.
#
# usage: scripts/a7_sibling_chain.sh BINARY
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$1
D=artifacts/sibling/T-plus-A-v2.bin
O=artifacts/sibling/v2
mkdir -p $O
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
step() {
  local name=$1; shift
  if [ -e "experiments/$name.json" ] || [ -e "experiments/$name.log" ]; then echo "refusing to overwrite $name" >&2; exit 1; fi
  local start=$SECONDS status=0
  nice -n 10 "$@" > "experiments/$name.log" 2>&1 || status=$?
  printf '{\n  "name": "%s",\n  "returncode": %s,\n  "elapsed_seconds": %s,\n  "binary_sha256": "%s",\n  "data_sha256": "%s",\n  "git_head": "%s",\n  "finished_utc": "%s",\n  "command": "%s"\n}\n' \
    "$name" "$status" "$((SECONDS-start))" "$(sha "$BIN")" "$(sha $D)" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" "$*" \
    > "experiments/$name.json"
  [ "$status" = 0 ] || { echo "$name failed ($status)" >&2; exit "$status"; }
}
step A7v2-graph256 "$BIN" graph --data $D --graph $O/neighbors256.bin --neighbors 256 --history-cap 256
step A7v2-graph64 "$BIN" graph --data $D --graph $O/neighbors64.bin --neighbors 64 --history-cap 256
step A7v2-K1 "$BIN" joint --data $D --model $O/K1.bin --predictions $O/K1.f32 --factors 256 --learning-rate .008 \
  --regularization .02 --decay .97 --iterations 41 --fixed-epoch --threads 2
step A7v2-knn256td "$BIN" knn --data $D --graph $O/neighbors256.bin --joint-model $O/K1.bin --neighbors 256 \
  --power 1 --shrink .5 --time-decay .003 --predictions $O/knn256td.f32 --support $O/knn256td.support.f32
step A7v2-knn64 "$BIN" knn --data $D --graph $O/neighbors64.bin --joint-model $O/K1.bin --power 1 --shrink .5 \
  --predictions $O/knn64.f32
echo "A7 chain done"
