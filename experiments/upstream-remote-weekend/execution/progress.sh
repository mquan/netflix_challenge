#!/bin/zsh
# Hourly progress reporter (read-only w.r.t. native jobs): writes progress.json and pushes.
W=~/workspace/netflix_remote_weekend
PY=~/workspace/netflix_upstream_repro/.venv/bin/python
cd $W
while true; do
  $PY -I - <<'PYEOF'
import json, re, shutil, subprocess, datetime as dt
from pathlib import Path
W = Path.home() / 'workspace/netflix_remote_weekend'
TREES = {'s3-rbm400pf': 'r2', 's3-tx300r': 'r4', 's3-rbm800p': 'r3'}
now = dt.datetime.now(dt.timezone.utc)
deadline = dt.datetime(2026, 11, 23, 14, 0, tzinfo=dt.timezone.utc)
jobs = {}
for name in ['s3-rbm400pf', 's3-tx300r', 's3-rbm800p']:
    R = W / f'artifacts/upstream-remote-weekend-{TREES[name]}'
    info = {'tree': R.name}
    if (R / (name + '-launch.json')).exists():
        info['launch'] = json.loads((R / (name + '-launch.json')).read_text())
    if (R / (name + '-status.json')).exists():
        info['status'] = json.loads((R / (name + '-status.json')).read_text())
    log = R / (name + '.log')
    if log.exists():
        text = log.read_text(errors='replace')
        eps = [int(n) for n in re.findall(r'^Epoch\s+(\d+)\b', text, re.M)]
        info['epoch_lines'] = len(eps); info['last_epoch'] = eps[-1] if eps else None
        info['phase'] = 'train_3->probe_3' if len(set(eps)) == len(eps) else 'fulltrain->qual'
        info['log_tail'] = text.splitlines()[-2:]
    for k in ['receipt', 'incomplete', 'failure', 'guard-sample', 'watchdog-recovery']:
        info[k] = (R / f'{name}-{k}.json').exists()
    info['exported'] = (R / 'exports' / name / 'verification.json').exists()
    info['archived'] = (R / 'archives' / name / 'index.json').exists()
    info['uploaded'] = (W / 'experiments/upstream-remote-weekend/execution' / name / 'transfer.json').exists()
    jobs[name] = info
ps = subprocess.run(['ps', '-o', 'pid,lstart,etime,time,rss,state,command', '-p',
     ','.join(subprocess.run(['pgrep', '-f', 'target/release/fit'], capture_output=True, text=True).stdout.split()) or '1'],
     capture_output=True, text=True).stdout.strip()
out = dict(observed=now.isoformat(timespec='seconds'), training_hours_left=round((deadline - now).total_seconds() / 3600, 2),
           native_process=ps,
           memory=subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True).stdout.strip(),
           swap=subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.strip(),
           disk_free_gib=round(shutil.disk_usage(W).free / 2**30, 1), jobs=jobs)
(W / 'experiments/upstream-remote-weekend/execution/progress.json').write_text(json.dumps(out, indent=1) + '\n')
PYEOF
  for i in 1 2 3 4 5 6; do
    [[ -e $(git rev-parse --git-common-dir)/worktrees/netflix_remote_weekend/index.lock ]] && { sleep 20; continue; }
    git add experiments/upstream-remote-weekend/execution/progress.json && \
      git commit -qm "weekend: hourly progress $(date -u +%FT%TZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- experiments/upstream-remote-weekend/execution/progress.json && \
      git push -q origin HEAD:research/remote-weekend-execution
    break
  done
  [[ -n $(pgrep -f 'upstream-remote-weekend/execution/driver') ]] || { echo "driver gone; final progress written"; exit 0; }
  sleep 3600
done
