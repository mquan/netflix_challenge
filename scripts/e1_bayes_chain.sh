#!/usr/bin/env bash
# E1-Bayes (NEXT_EXPLORATIONS.md): repeat the Stage C B4 retrain (B4-full-r128:
# rank 128, 160 sweeps, burn-in 16, seed 13, item-bin 2, drift 2, 4 threads)
# with P2 rows given noise precision alpha*w via --row-weights. Bash on purpose
# (see GOAL.md rule 3); provenance JSON written per arm.
#
# usage: scripts/e1_bayes_chain.sh BINARY ARM   (ARM = w1 | w2)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$1; ARM=$2
FULL=data/processed/netflix.bin
E=artifacts/rehearsal/e1
W=$E/p2-$ARM.f32
NAME=E1-B-B4-$ARM
[ -f "$W" ] && [ -f "$W.json" ] || { echo "missing $W" >&2; exit 1; }
if [ -e "experiments/$NAME.json" ] || [ -e "experiments/$NAME.log" ]; then echo "refusing to overwrite $NAME" >&2; exit 1; fi
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
CMD=("$BIN" bayes --data $FULL --model $E/B-B4-$ARM.bin --predictions $E/B-B4-$ARM.f32 --factors 128
     --samples 160 --burn-in 16 --snapshot-every 16 --checkpoint-every 8 --threads 4 --seed 13
     --item-bin 2 --drift 2 --row-weights "$W")
start=$SECONDS status=0
nice -n 10 "${CMD[@]}" > "experiments/$NAME.log" 2>&1 || status=$?
printf '{\n  "name": "%s",\n  "returncode": %s,\n  "elapsed_seconds": %s,\n  "binary_sha256": "%s",\n  "bayes_hpp_sha256": "%s",\n  "weights_sha256": "%s",\n  "git_head": "%s",\n  "finished_utc": "%s",\n  "command": "%s"\n}\n' \
  "$NAME" "$status" "$((SECONDS-start))" "$(sha "$BIN")" "$(sha src/bayes.hpp)" "$(sha "$W")" \
  "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" "${CMD[*]}" > "experiments/$NAME.json"
exit "$status"
