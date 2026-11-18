#!/bin/sh
# Run pinned native jobs one at a time under the remote-12h supervisor.
# No job starts after LAUNCH_CUTOFF; each is hard-stopped at DEADLINE.
# Usage: scripts/remote12h_queue.sh LAUNCH_CUTOFF_UTC DEADLINE_UTC JOB...
set -u
CUTOFF=$1; DEADLINE=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
Q=experiments/upstream-remote-12h/queue.log
cutoff_s=$(date -j -u -f '%Y-%m-%dT%H:%M:%SZ' "$CUTOFF" +%s)
for JOB in "$@"; do
  if [ "$(date -u +%s)" -ge "$cutoff_s" ]; then
    echo "$(date -u +%FT%TZ) skip $JOB: launch cutoff $CUTOFF passed" >> $Q; continue
  fi
  D=experiments/upstream-remote-12h/$JOB
  mkdir -p "$D"
  echo "$(date -u +%FT%TZ) start $JOB" >> $Q
  .venv/bin/python -u scripts/remote12h_supervisor.py --name "$JOB" --out-dir "$D" --deadline "$DEADLINE" \
    --sample-every 60 --grace 60 --publish-every 1800 \
    --publish-path "$D/status.json" --publish-path "$D/resources.jsonl" --publish-path "$D/supervisor.log" \
    --publish-path $Q \
    -- env UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_repro.py \
    --run-job "$JOB" >> "$D/wrapper.out" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) end $JOB rc=$rc state=$(grep -o '"state": "[^"]*"' $D/status.json)" >> $Q
  case $(grep -o '"state": "[^"]*"' $D/status.json) in
    *interrupted*) echo "$(date -u +%FT%TZ) stop queue: $JOB interrupted" >> $Q; break;;
  esac
done
echo "$(date -u +%FT%TZ) queue finished" >> $Q
