#!/bin/zsh
# Serial weekend driver: run-job -> export -> archive -> draft-release upload ->
# commit/push metadata, per producer in handoff order. Never restarts a producer
# and never reruns prepare. A failed step is recorded and the next producer is
# still offered to the helper's own admission gate (which waits on capacity/time).
set -uo pipefail
W=~/workspace/netflix_remote_weekend
PY=~/workspace/netflix_upstream_repro/.venv/bin/python
R=$W/artifacts/upstream-remote-weekend-r2
TREE=$R/source
EXE=$W/experiments/upstream-remote-weekend/execution
DEADLINE=2026-11-23T14:00:00Z
cd $W
push() {
  for i in 1 2 3 4 5 6; do
    [[ -e $(git rev-parse --git-common-dir)/worktrees/netflix_remote_weekend/index.lock ]] && { sleep 20; continue; }
    git add experiments/upstream-remote-weekend/execution
    git commit -qm "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
    git push -q origin HEAD:research/remote-weekend-execution && return 0
    sleep 30
  done
  echo "push failed" >&2
}
collect() {  # small metadata only; never arrays or archive parts
  local J=$1; mkdir -p $EXE/$J
  for f in $R/$J*.json $R/$J.log $R/$J-watchdog.log; do [[ -e $f ]] && cp $f $EXE/$J/; done
  [[ -e $R/exports/$J/verification.json ]] && cp $R/exports/$J/verification.json $EXE/$J/export-verification.json
  [[ -e $R/archives/$J/index.json ]] && cp $R/archives/$J/index.json $EXE/$J/archive-index.json
  return 0
}
for JOB in s3-rbm400pf s3-tx300r s3-rbm800p; do
  if [[ -e $R/$JOB-receipt.json || -e $R/$JOB.log ]]; then
    echo "[$(date -u +%FT%TZ)] $JOB already has native evidence; not relaunching"
  else
    echo "[$(date -u +%FT%TZ)] run-job $JOB"
    if ! $PY scripts/upstream_remote_weekend.py run-job --tree $TREE --job $JOB --deadline-utc $DEADLINE; then
      echo "[$(date -u +%FT%TZ)] run-job $JOB FAILED"; collect $JOB; push "weekend: $JOB run-job failed (no retry)"; continue
    fi
  fi
  [[ -e $R/$JOB-receipt.json ]] || { echo "no receipt for $JOB"; collect $JOB; push "weekend: $JOB no receipt"; continue; }
  if [[ ! -e $R/exports/$JOB/verification.json ]]; then
    echo "[$(date -u +%FT%TZ)] export $JOB"
    $PY scripts/upstream_remote_weekend.py export --tree $TREE --job $JOB --output $R/exports/$JOB || {
      echo "export $JOB FAILED"; collect $JOB; push "weekend: $JOB export failed"; continue; }
  fi
  if [[ ! -e $R/archives/$JOB/index.json ]]; then
    echo "[$(date -u +%FT%TZ)] archive $JOB"
    $PY scripts/upstream_remote_weekend.py archive --tree $TREE --job $JOB --output $R/archives/$JOB || {
      echo "archive $JOB FAILED"; collect $JOB; push "weekend: $JOB archive failed"; continue; }
  fi
  collect $JOB
  echo "[$(date -u +%FT%TZ)] upload $JOB"
  if $PY -I $EXE/upload_release.py $JOB $R/archives/$JOB $EXE/$JOB/transfer.json; then
    push "weekend: $JOB complete, exported, archived, uploaded to draft release"
  else
    echo "upload $JOB FAILED; local artifacts retained"; push "weekend: $JOB complete+archived; upload FAILED (local retained)"
  fi
done
echo "[$(date -u +%FT%TZ)] driver done"
