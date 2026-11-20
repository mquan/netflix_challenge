#!/bin/zsh
# Serial remote15h driver: prepare once, then run-job -> export -> archive ->
# release upload -> commit/push for each producer in the handoff order.
# Stops at the first failure; never restarts a producer or reruns prepare.
set -euo pipefail
W=~/workspace/netflix_remote15h
PY=~/workspace/netflix_upstream_repro/.venv/bin/python
TREE=$W/artifacts/upstream-remote15h/source
EXP=$W/artifacts/upstream-remote15h/exports
ARC=$W/artifacts/upstream-remote15h/archives
EXE=$W/experiments/upstream-remote15h/execution
DEADLINE=2026-11-20T17:44:19Z
TAG=remote15h-own-20261120
REPO=mquan/netflix_challenge
cd $W
mkdir -p $ARC

push() {
  git add experiments/upstream-remote15h/execution
  git commit -qm "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
  git push -q origin HEAD:research/remote15h-handoff || echo "push failed" >&2
}

if [[ ! -e $TREE ]]; then
  echo "[$(date -u +%FT%TZ)] prepare"
  $PY scripts/upstream_remote15h.py prepare \
    --source-tree ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh \
    --data-root ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data \
    --tree $TREE
  mkdir -p $EXE/bootstrap
  cp $TREE/../preparation.json $TREE/../build.log $TREE/../split3.log $EXE/bootstrap/
  push "remote15h: bootstrap prepared and verified"
fi

for JOB in "$@"; do
  echo "[$(date -u +%FT%TZ)] run-job $JOB"
  $PY scripts/upstream_remote15h.py run-job --tree $TREE --job $JOB --deadline-utc $DEADLINE
  echo "[$(date -u +%FT%TZ)] export $JOB"
  $PY scripts/upstream_remote15h.py export --tree $TREE --job $JOB --output $EXP/$JOB
  mkdir -p $EXE/$JOB
  for f in $EXP/$JOB/*; do [[ $f == *.npz ]] || cp $f $EXE/$JOB/; done
  A=$ARC/$JOB-export.tar.gz
  [[ -e $A ]] && { echo "archive exists, refusing to overwrite: $A" >&2; exit 1; }
  tar -czf $A -C $EXP $JOB
  ASHA=$(shasum -a 256 $A | cut -d' ' -f1); ABYTES=$(stat -f %z $A)
  if ! gh release view $TAG --repo $REPO >/dev/null 2>&1; then
    gh release create $TAG --repo $REPO --draft --title 'Own remote15h training transfer' \
      --notes 'Transfer of our own original-source training outputs; no raw ratings.'
  fi
  UPLOADED=false
  if gh release upload $TAG $A --repo $REPO; then UPLOADED=true; fi
  $PY - $EXE/$JOB/transfer.json $TAG ${A:t} $ASHA $ABYTES $UPLOADED $EXP/$JOB/verification.json <<'EOF'
import json, sys, datetime
out, tag, name, h, n, up, ver = sys.argv[1:]
v = json.load(open(ver))
json.dump(dict(observed=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  release_tag=tag, asset_name=name, archive_sha256=h, archive_bytes=int(n),
  uploaded=(up == 'true'), draft_release=True,
  records=v['records'], prediction_arrays_in_git=False), open(out, 'w'), indent=2)
EOF
  push "remote15h: $JOB complete, verified, exported (uploaded=$UPLOADED)"
done
echo "[$(date -u +%FT%TZ)] driver done: $*"
