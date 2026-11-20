#!/bin/zsh
# Hourly progress reporter: writes progress.json (phase/epoch from native logs,
# memory, disk, exports) and pushes it. Read-only with respect to native jobs.
W=~/workspace/netflix_remote15h
PY=~/workspace/netflix_upstream_repro/.venv/bin/python
cd $W
while true; do
  $PY - <<'EOF'
import json, re, shutil, subprocess, datetime as dt
from pathlib import Path
W = Path.home() / 'workspace/netflix_remote15h'
R = W / 'artifacts/upstream-remote15h-r2'
now = dt.datetime.now(dt.timezone.utc)
deadline = dt.datetime(2026, 11, 20, 17, 44, 19, tzinfo=dt.timezone.utc)
arrival = dt.datetime(2026, 11, 20, 3, 44, 19, tzinfo=dt.timezone.utc)
jobs = {}
ext = ['s3-bknbrx-100w','s3-mfrbmx-60h','s3-tx60o','s3-tx50n','s3-bk4-g100','s3-rbm100p','s3-bknbrx-200t','s3-mfrbmx-50w']
for name in ['s3-dnn24', 's3-attn32', 's3-rbm200e'] + ext:
    R = W / ('artifacts/upstream-remote15h-ext' if name in ext else 'artifacts/upstream-remote15h-r2')
    log = R / (name + '.log'); info = {}
    if (R / (name + '-launch.json')).exists():
        info['launch'] = json.loads((R / (name + '-launch.json')).read_text())
    if log.exists():
        text = log.read_text(errors='replace')
        eps = [int(n) for n in re.findall(r'^Epoch\s+(\d+)\b', text, re.M)]
        info['completed_epoch_lines'] = len(eps)
        info['last_epoch'] = eps[-1] if eps else None
        info['phase'] = 'train_3->probe_3' if len(set(eps)) == len(eps) else 'fulltrain->qual'
        info['log_tail'] = text.splitlines()[-2:]
    info['receipt'] = (R / (name + '-receipt.json')).exists()
    info['incomplete'] = (R / (name + '-incomplete.json')).exists()
    info['exported'] = (R / 'exports' / name / 'verification.json').exists()
    jobs[name] = info
ps = subprocess.run(['pgrep', '-fl', 'target/release/fit'], capture_output=True, text=True).stdout.strip()
vm = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True).stdout.strip()
swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.strip()
out = dict(observed=now.isoformat(), elapsed_hours=round((now - arrival).total_seconds() / 3600, 2),
           training_hours_left=round((deadline - now).total_seconds() / 3600, 2),
           native_process=ps, memory=vm, swap=swap,
           disk_free_gib=round(shutil.disk_usage(W).free / 2**30, 1), jobs=jobs,
           next_action='assigned driver then extension driver, serially: run-job -> export -> upload -> push')
(W / 'experiments/upstream-remote15h/execution/progress.json').write_text(json.dumps(out, indent=2) + '\n')
EOF
  for i in 1 2 3 4 5 6; do
    [[ -e .git/index.lock || -e $(git rev-parse --git-dir)/index.lock ]] && { sleep 20; continue; }
    git add experiments/upstream-remote15h/execution/progress.json && \
      git commit -qm "remote15h: hourly progress $(date -u +%FT%TZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- experiments/upstream-remote15h/execution/progress.json && \
      git push -q origin HEAD:research/remote15h-handoff
    break
  done
  [[ -n $(pgrep -f 'execution/driver.sh') ]] || { echo "driver gone; final progress written"; exit 0; }
  sleep 3600
done
