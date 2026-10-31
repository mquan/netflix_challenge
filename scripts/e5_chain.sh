#!/usr/bin/env bash
# E5 pilot (NEXT_EXPLORATIONS.md): ordinal (interval-censored) Bayes timeSVD++
# vs its same-binary Gaussian control, B2 recipe (rank 64, 128 sweeps, burn-in
# 16, seed 2026, 4 threads) on the P2 holdout (train minus P2; probe = P2).
# Bash on purpose (GOAL.md rule 3). One arm per invocation, so the two arms
# run as separate processes.
#
# usage: scripts/e5_chain.sh BINARY gaussian|ordinal
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$1; ARM=$2
case $ARM in gaussian|ordinal) ;; *) echo "arm must be gaussian or ordinal" >&2; exit 2;; esac
D=artifacts/validation/train-matched-20260927.bin
OUT=artifacts/e5
NAME=E5-p2-r64-$ARM
mkdir -p $OUT
if [ -e "experiments/$NAME.json" ] || [ -e "experiments/$NAME.log" ] || [ -e "$OUT/$NAME.f32" ]; then
  echo "refusing to overwrite $NAME" >&2; exit 1
fi
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
CMD=("$BIN" bayes --data $D --model $OUT/$NAME.bin --predictions $OUT/$NAME.f32 --factors 64 --samples 128
     --burn-in 16 --snapshot-every 16 --checkpoint-every 8 --threads 4 --seed 2026 --likelihood "$ARM")
start=$SECONDS status=0
nice -n 10 "${CMD[@]}" > "experiments/$NAME.log" 2>&1 || status=$?
printf '{\n  "name": "%s",\n  "returncode": %s,\n  "elapsed_seconds": %s,\n  "binary_sha256": "%s",\n  "bayes_hpp_sha256": "%s",\n  "data_sha256": "%s",\n  "git_head": "%s",\n  "finished_utc": "%s",\n  "command": "%s"\n}\n' \
  "$NAME" "$status" "$((SECONDS-start))" "$(sha "$BIN")" "$(sha src/bayes.hpp)" "$(sha $D)" \
  "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" "${CMD[*]}" > "experiments/$NAME.json"
exit "$status"
