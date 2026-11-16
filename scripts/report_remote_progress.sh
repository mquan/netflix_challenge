#!/bin/zsh
# Publish the second computer's remote-ensemble progress to the branch so the
# other machine can follow it with `git pull`. Commits REMOTE_PROGRESS.md when
# a stage completes or the run state changes, plus an hourly heartbeat.
# Usage (inside tmux): scripts/report_remote_progress.sh [interval_seconds]
set -u
cd "${0:A:h}/.."
INTERVAL=${1:-120}
HEARTBEAT=3600
BRANCH=research/independent-ensemble
OUT=REMOTE_PROGRESS.md
RUN=artifacts/independent-ensemble/remote
last_key=""
last_commit=0

render() {
  python3 - "$RUN" <<'PY'
import json, pathlib, sys, datetime as dt
run = pathlib.Path(sys.argv[1])
status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
plan = json.loads((run/'plan.json').read_text()) if (run/'plan.json').exists() else {}
now = dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
lines = ['# Second-computer progress (M3 Max, 96 GB)', '',
         f'Auto-updated by `scripts/report_remote_progress.sh`. Last update: {now}.', '',
         f'- State: **{status.get("state", "unknown")}**, device `{status.get("device", plan.get("device", "?"))}`, arms {plan.get("arms")}',
         f'- Completed: {len(status.get("completed", []))}/{status.get("total", len(plan.get("stages", [])))}',
         f'- Active: {", ".join(status.get("active", {})) or "none"}',
         f'- Runner heartbeat (UTC): {status.get("updated_utc", "n/a")}']
if status.get('error'):
    lines.append(f'- Error: `{status["error"]}`')
lines += ['', '| Stage | Family | State | Elapsed |', '|---|---|---|---|']
done = set(status.get('completed', []))
active = set(status.get('active', {}))
for s in plan.get('stages', []):
    r = run/'logs'/f'{s["name"]}.receipt.json'
    el = ''
    if r.exists():
        el = f'{json.loads(r.read_text())["elapsed_seconds"]/3600:.2f} h'
    st = 'done' if s['name'] in done or r.exists() else 'running' if s['name'] in active else 'queued'
    lines.append(f'| {s["name"]} | {s["family"]} | {st} | {el} |')
cmp = run/'comparison.json'
if cmp.exists():
    lines += ['', '## comparison.json', '', '```json', cmp.read_text().strip()[:6000], '```']
print('\n'.join(lines))
PY
}

while true; do
  key=$(python3 -c "import json;s=json.load(open('$RUN/status.json'));print(s.get('state'),sorted(s.get('completed',[])),sorted(s.get('active',{})))" 2>/dev/null)
  now=$(date +%s)
  if [[ "$key" != "$last_key" || $((now - last_commit)) -ge $HEARTBEAT ]]; then
    render > "$OUT"
    git add "$OUT"
    if ! git diff --cached --quiet -- "$OUT"; then
      git -c user.email=mquannie@gmail.com commit -q -m "progress(remote M3 Max): ${key:-no status}" -- "$OUT"
      GIT_TERMINAL_PROMPT=0 git pull -q --rebase origin "$BRANCH" && GIT_TERMINAL_PROMPT=0 git push -q origin "HEAD:$BRANCH" \
        && echo "$(date +%T) pushed: $key" || echo "$(date +%T) push failed; will retry"
    fi
    last_key=$key; last_commit=$now
  fi
  [[ "$key" == complete* || "$key" == failed* ]] && { echo "run finished: $key"; exit 0; }
  sleep "$INTERVAL"
done
