#!/usr/bin/env python3
"""Deadline- and memory-bounded supervisor for one REMOTE_12H_PLAN native job.

Starts a command in its own session/process group, samples the whole group's
RSS, host memory level, swap and disk, and stops only that group (SIGTERM,
grace, SIGKILL) at the hard deadline or on sustained memory pressure.
Optionally commits/pushes named heartbeat paths on a fixed interval.
Never kills by process name.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc)


def parse_utc(text):
    value = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


def group_members(pgid):
    """Return [(pid, rss_kb)] for every live process in the process group."""
    out = subprocess.run(['ps', '-A', '-o', 'pid=,pgid=,rss=,stat='], capture_output=True, text=True).stdout
    members = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and int(parts[1]) == pgid and not parts[3].startswith('Z'):
            members.append((int(parts[0]), int(parts[2])))
    return members


def host_sample(disk_path):
    sample = {}
    try:
        sample['memory_level_pct'] = int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_level'],
                                                        capture_output=True, text=True).stdout.strip())
    except ValueError:
        sample['memory_level_pct'] = None
    swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    m = re.search(r'used = ([\d.]+)M', swap)
    sample['swap_used_mb'] = float(m.group(1)) if m else None
    st = os.statvfs(disk_path)
    sample['disk_free_gb'] = round(st.f_bavail * st.f_frsize / 2**30, 2)
    return sample


def stop_group(pgid, grace, log):
    log(f'SIGTERM to process group {pgid}')
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + grace
    while time.time() < end and group_members(pgid):
        time.sleep(0.5)
    if group_members(pgid):
        log(f'SIGKILL to remaining members of process group {pgid}')
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def publish(repo, paths, message, email, log):
    paths = [p for p in paths if (Path(repo)/p).exists()]
    if not paths:
        return
    for attempt in range(3):
        add = subprocess.run(['git', '-C', repo, 'add', '--', *paths], capture_output=True, text=True)
        if add.returncode == 0:
            break
        time.sleep(5)
    staged = subprocess.run(['git', '-C', repo, 'diff', '--cached', '--quiet', '--', *paths])
    if staged.returncode == 0:
        return
    commit = subprocess.run(['git', '-C', repo, '-c', f'user.email={email}', 'commit', '-q', '-m', message,
                             '--', *paths], capture_output=True, text=True)
    if commit.returncode != 0:
        log('heartbeat commit failed: ' + commit.stderr.strip()[-300:])
        return
    push = subprocess.run(['git', '-C', repo, 'push', '-q', 'origin', 'HEAD'], capture_output=True, text=True)
    log('heartbeat pushed' if push.returncode == 0 else 'heartbeat push failed: ' + push.stderr.strip()[-300:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--deadline', required=True, help='Hard UTC stop time (ISO 8601)')
    ap.add_argument('--cwd', default='.')
    ap.add_argument('--sample-every', type=float, default=60)
    ap.add_argument('--grace', type=float, default=60)
    ap.add_argument('--min-memory-level', type=int, default=20, help='kern.memorystatus_level percent')
    ap.add_argument('--pressure-samples', type=int, default=5)
    ap.add_argument('--max-swap-growth-mb', type=float, default=4096)
    ap.add_argument('--min-disk-gb', type=float, default=10)
    ap.add_argument('--publish-every', type=float, default=0)
    ap.add_argument('--publish-repo', default='.')
    ap.add_argument('--publish-path', action='append', default=[])
    ap.add_argument('--publish-email', default='mquannie@gmail.com')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        ap.error('missing command after --')
    deadline = parse_utc(a.deadline)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    events = (a.out_dir/'supervisor.log').open('a')
    resources = (a.out_dir/'resources.jsonl').open('a')
    status_path = a.out_dir/'status.json'

    def log(message):
        events.write(f'{now().isoformat()} {message}\n')
        events.flush()

    def status(**fields):
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(fields, indent=2) + '\n')
        temp.replace(status_path)

    if now() >= deadline:
        log('deadline already passed; not launching')
        status(name=a.name, state='not-launched-deadline-passed', deadline=deadline.isoformat())
        return 3
    base = host_sample(a.out_dir)
    if base['disk_free_gb'] < a.min_disk_gb:
        log(f"insufficient disk {base['disk_free_gb']} GB; not launching")
        status(name=a.name, state='not-launched-disk', **base)
        return 4
    started = now()
    proc = subprocess.Popen(command, cwd=a.cwd, start_new_session=True)
    pgid = proc.pid
    log(f'launched pid/pgid {pgid}: {" ".join(command)} deadline {deadline.isoformat()}')
    meta = dict(name=a.name, command=command, cwd=str(Path(a.cwd).resolve()), pgid=pgid,
                started=started.isoformat(), deadline=deadline.isoformat(), baseline=base)
    peak_rss = 0
    pressure = 0
    reason = None
    last_publish = time.time()
    next_sample = 0.0
    while True:
        code = proc.poll()
        if code is not None:
            break
        if time.time() >= next_sample:
            next_sample = time.time() + a.sample_every
            members = group_members(pgid)
            rss_mb = sum(r for _, r in members) / 1024
            peak_rss = max(peak_rss, rss_mb)
            host = host_sample(a.out_dir)
            swap_growth = (host['swap_used_mb'] or 0) - (base['swap_used_mb'] or 0)
            row = dict(time=now().isoformat(), group_rss_mb=round(rss_mb, 1), peak_group_rss_mb=round(peak_rss, 1),
                       n_procs=len(members), swap_growth_mb=round(swap_growth, 1), **host)
            resources.write(json.dumps(row) + '\n')
            resources.flush()
            low = host['memory_level_pct'] is not None and host['memory_level_pct'] < a.min_memory_level
            pressure = pressure + 1 if (low and swap_growth > a.max_swap_growth_mb) else 0
            status(**meta, state='running', last_sample=row,
                   elapsed_h=round((now() - started).total_seconds() / 3600, 3))
            if pressure >= a.pressure_samples:
                reason = 'memory-pressure'
            elif host['disk_free_gb'] < a.min_disk_gb:
                reason = 'disk'
        if reason is None and now() >= deadline:
            reason = 'deadline'
        if reason:
            log(f'stopping owned group: {reason}')
            stop_group(pgid, a.grace, log)
            code = proc.wait()
            break
        if a.publish_every and time.time() - last_publish >= a.publish_every:
            last_publish = time.time()
            publish(a.publish_repo, a.publish_path, f'remote-12h heartbeat: {a.name} running', a.publish_email, log)
        time.sleep(min(1.0, a.sample_every))
    leftover = group_members(pgid)
    if leftover:
        stop_group(pgid, a.grace, log)
        leftover = group_members(pgid)
    state = 'interrupted-' + reason if reason else ('completed' if code == 0 else 'failed')
    final = dict(**meta, state=state, exit_code=code, finished=now().isoformat(),
                 elapsed_h=round((now() - started).total_seconds() / 3600, 3),
                 peak_group_rss_mb=round(peak_rss, 1), leftover_group_members=leftover, final_host=host_sample(a.out_dir))
    status(**final)
    log(f'finished state={state} exit={code} leftover={leftover}')
    return 0 if state == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
