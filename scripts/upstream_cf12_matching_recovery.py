#!/usr/bin/env python3
"""Restart the unchanged matching queue only after its original bank is complete."""
import datetime as dt
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import upstream_phase_bank as bank

u=bank.u
REPORT=u.ROOT/'experiments/upstream-cf12-matching-recovery'
ORIGINAL=u.ROOT/'experiments/upstream-cf12-matching-production'
SCRIPT=u.ROOT/'scripts/upstream_cf12_matching_production.py'


def running(pid,expected):
    row=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],text=True,capture_output=True).stdout.strip()
    return bool(row and not row.split()[0].startswith('Z') and expected in row)


def preflight():
    old=u.read(ORIGINAL/'launch.json')
    if running(old['pid'],'upstream_cf12_matching_production.py'):
        raise ValueError('Matching controller still live; duplicate restart refused')
    if u.sha(SCRIPT)!=old['script_sha256']:raise ValueError('Original matching controller code changed')
    if (ORIGINAL/'plan.json').exists() or (u.ROOT/'artifacts/upstream-cf12-matching-production').exists():
        raise ValueError('Existing fit preparation must be reviewed rather than restarted')
    if 'ValueError: Production bank still incomplete' not in (ORIGINAL/'launch.log').read_text():
        raise ValueError('Known terminal missing-bank failure not established')
    return old


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);old=preflight()
        u.write(REPORT/'plan.json',dict(previous_launch=old,original_script_sha256=u.sha(SCRIPT),
            failure='First recovered-bank audit completed before the independent local phase-bank dependency. Original queue exited on its explicit incomplete-bank guard.',
            policy='Wait for original phase-bank manifest, verify every bound file, then restart the unchanged queue once. No fitting artifacts overwritten; no native process controlled.'))
        while not (bank.REPORT/'manifest.json').exists():
            if not running(46595,'scripts/upstream_phase_bank.py --watch'):
                raise RuntimeError('Original bank watcher is not live; inspect preserved dependency state')
            state=dict(state='waiting_for_original_phase_bank',bank_watcher_pid=46595,bank_watcher_observed_live=True,
                observed=dt.datetime.now(dt.timezone.utc).isoformat(),previous_controller_terminal=True)
            u.write(REPORT/'status.json',state);u.write(ORIGINAL/'queue-status.json',state);time.sleep(30)
        bank.build();preflight()
        with (ORIGINAL/'queue.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (REPORT/'child.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SCRIPT)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        u.write(REPORT/'launch-child.json',dict(pid=child.pid,command=child.args,
            script_sha256=u.sha(SCRIPT),bank_manifest_sha256=u.sha(bank.REPORT/'manifest.json'),
            observed=dt.datetime.now(dt.timezone.utc).isoformat(),all_original_bank_sources_rehashed=True))
        # The original queue owns its own lock, memory admission and extension
        # priority policy. Preserve its exit status; never automatically retry.
        code=child.wait()
        u.write(REPORT/'status.json',dict(state='original_queue_exited',child_pid=child.pid,returncode=code,
            observed=dt.datetime.now(dt.timezone.utc).isoformat(),goal_requires_independent_review=True))
        if code:raise RuntimeError('Original matching queue failed again; inspect child.log without automatic retries')


if __name__=='__main__':
    try:run()
    except Exception as error:
        u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise
