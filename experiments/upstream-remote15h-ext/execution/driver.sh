#!/bin/zsh
# Serial remote15h driver: prepare once, then run-job -> export -> archive ->
# release upload -> commit/push for each producer in the handoff order.
# Extension: waits for the assigned driver; skips a job that is not admitted or fails;
# stops at the deadline. Never restarts a producer or reruns prepare.
set -uo pipefail
W=~/workspace/netflix_remote15h
PY=~/workspace/netflix_upstream_repro/.venv/bin/python
TREE=$W/artifacts/upstream-remote15h-ext/source
EXP=$W/artifacts/upstream-remote15h-ext/exports
ARC=$W/artifacts/upstream-remote15h-ext/archives
EXE=$W/experiments/upstream-remote15h-ext/execution
DEADLINE=2026-11-20T17:44:19Z
TAG=remote15h-own-20261120
REPO=mquan/netflix_challenge
cd $W
while pgrep -f "^zsh experiments/upstream-remote15h/execution/driver.sh" >/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] assigned driver finished; starting extension"
mkdir -p $ARC

push() {
  git add experiments/upstream-remote15h-ext
  git commit -qm "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
  git push -q origin HEAD:research/remote15h-handoff || echo "push failed" >&2
}

if [[ ! -e $TREE ]]; then
  echo "[$(date -u +%FT%TZ)] prepare"
  $PY scripts/upstream_remote15h_ext.py prepare \
    --source-tree ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh \
    --data-root ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data \
    --tree $TREE
  mkdir -p $EXE/bootstrap
  cp $TREE/../.cargo/config.toml $EXE/bootstrap/cargo-config.toml; cp $TREE/../preparation.json $TREE/../build.log $TREE/../split3.log $EXE/bootstrap/
  push "remote15h-ext: bootstrap prepared and verified"
fi

for JOB in "$@"; do
  echo "[$(date -u +%FT%TZ)] run-job $JOB"
  if ! $PY scripts/upstream_remote15h_ext.py run-job --tree $TREE --job $JOB --deadline-utc $DEADLINE; then
    [[ -e $TREE/../$JOB-incomplete.json ]] && { echo "deadline reached"; push "remote15h-ext: $JOB deadline-interrupted (ineligible)"; break; }
    echo "[$(date -u +%FT%TZ)] $JOB not admitted or failed; continuing"; continue
  fi
  echo "[$(date -u +%FT%TZ)] export $JOB"
  $PY scripts/upstream_remote15h_ext.py export --tree $TREE --job $JOB --output $EXP/$JOB || { echo "export $JOB failed"; push "remote15h-ext: $JOB export failed"; continue; }
  mkdir -p $EXE/$JOB
  for f in $EXP/$JOB/*; do [[ $f == *.npz ]] || cp $f $EXE/$JOB/; done
  A=$ARC/$JOB-export.tar.gz
  [[ -e $A ]] && { echo "archive exists, refusing to overwrite: $A" >&2; continue; }
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
  push "remote15h-ext: $JOB complete, verified, exported (uploaded=$UPLOADED)"
done
echo "[$(date -u +%FT%TZ)] driver done: $*"
