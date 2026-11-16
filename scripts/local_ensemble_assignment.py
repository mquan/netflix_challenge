#!/usr/bin/env python3
"""End this Mac's queue after its assigned, verified stages complete."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from local_ensemble_run import ROOT, RUN, read, write, digest

REQUIRED = ['baseline-tree', 'baseline-nn', 'snapshots-tree', 'snapshots-nn',
            'calibration-tree', 'calibration-nn']


def assignment_ready(run, stages):
    for stage in stages:
        receipt = run/'logs'/f'{stage["name"]}.receipt.json'
        if not receipt.exists():
            return False
        done = read(receipt)
        if done.get('returncode') != 0 or done['report_sha256'] != digest(stage['report']):
            raise ValueError(f'Invalid completed assigned stage: {stage["name"]}')
    return True


def main():
    os.chdir(ROOT)
    plan = read(RUN/'plan.json')
    status = read(RUN/'status.json')
    pid = status['pid']
    expected_plan = digest(RUN/'plan.json')
    required = [s for s in plan['stages'] if s['name'] in REQUIRED]
    if len(required) != len(REQUIRED):
        raise ValueError('Assigned stages do not match the frozen plan')
    policy = dict(primary_required=REQUIRED,
                  delegated=[s['name'] for s in plan['stages'] if s['name'] not in REQUIRED],
                  remote_assignment='User confirmed the 10-stage remote job: baseline, combined treatment and ENN stacking',
                  primary_pid=pid, monitor_pid=os.getpid(),
                  plan_sha256=expected_plan,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  rule='Finish all six assigned stages and verify their receipts before ending the remaining local queue; no quality-based selection')
    write(RUN/'assignment.json', policy)
    while not assignment_ready(RUN, required):
        if digest(RUN/'plan.json') != expected_plan:
            raise ValueError('Frozen plan changed during assignment monitoring')
        os.kill(pid, 0)
        time.sleep(5)
    command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
    if ('scripts/local_ensemble_run.py --run' not in command or str(ROOT) not in command
            or os.getpgid(pid) != pid):
        raise ValueError('Refuse to signal a process that is not the owned trial runner')
    # Every assigned model has exited and its report is hash-verified. Any
    # remaining child belongs to the now-delegated portion of this queue.
    os.killpg(pid, signal.SIGTERM)
    for _ in range(50):
        time.sleep(.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    lock = RUN/'runner.lock'
    if lock.exists() and lock.read_text().strip() == str(pid):
        lock.unlink()
    write(RUN/'status.json', dict(state='assigned-work-complete',
          completed=REQUIRED, delegated=policy['delegated'],
          updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
          explanation='Remote computer covers combined treatment and ENN; all six local assigned stages verified'))
    write(RUN/'assignment-complete.json', dict(policy=policy, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print('Assigned local work complete; duplicate remaining queue ended.', flush=True)


if __name__ == '__main__':
    main()
