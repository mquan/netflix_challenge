#!/bin/sh
# Launch one pinned native job under the remote-12h supervisor in tmux.
# Usage: scripts/remote12h_launch.sh JOB DEADLINE_UTC
set -eu
JOB=$1
DEADLINE=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
D=experiments/upstream-remote-12h/$JOB
cd "$ROOT"
mkdir -p "$D"
LOG=artifacts/upstream-repro/fresh-receipts/$JOB.log
tmux new-session -d -s "r12-$JOB" -c "$ROOT" \
  "caffeinate -dimsu .venv/bin/python -u scripts/remote12h_supervisor.py --name $JOB --out-dir $D \
   --deadline $DEADLINE --sample-every 60 --grace 60 --publish-every 1800 \
   --publish-path $D/status.json --publish-path $D/resources.jsonl --publish-path $D/supervisor.log \
   --publish-path $D/native-log-timestamps.txt \
   -- env UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/upstream_repro.py \
   --run-job $JOB >> $D/wrapper.out 2>&1"
tmux new-window -t "r12-$JOB" -n stamps \
  "sleep 5; tail -n +1 -F $LOG | while IFS= read -r l; do printf '%s %s\n' \"\$(date -u +%FT%TZ)\" \"\$l\"; done >> $D/native-log-timestamps.txt"
echo "launched $JOB (tmux r12-$JOB) deadline $DEADLINE"
