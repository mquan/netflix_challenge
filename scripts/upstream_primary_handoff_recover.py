#!/usr/bin/env python3
"""Finish an interrupted idle handoff only after the predecessor is absent."""
import datetime as dt
import fcntl
from pathlib import Path
import subprocess
import sys
import upstream_primary_handoff_aex as handoff

u=handoff.u
transfer=handoff.inherited.transfer
REPORT=handoff.REPORT
PRIMARY=transfer.PRIMARY


def require_absent(pid,state,rows):
    if any(row['pid']==pid for row in rows):
        raise ValueError('Predecessor PID still present; recovery sends no signals')
    if transfer.busy_primary(pid,state):
        raise ValueError('Predecessor child or wrapper group remains live')
    names={'upstream_full_reproduce.py','upstream_primary_guarded.py','upstream_primary_guarded_aex.py'}
    for row in rows:
        if 'Z' not in row['state'] and any(Path(arg).name in names for arg in row['command'].split()):
            raise ValueError('Another primary scheduler is live')


def run():
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'result.json').exists():raise ValueError('Handoff already published')
        plan=u.read(REPORT/'plan.json');original=plan['original_launch'];state=u.read(PRIMARY/'status.json')
        if u.read(PRIMARY/'launch.json')!=original:raise ValueError('Original primary launch changed')
        failed=u.read(REPORT/'status.json')
        if failed.get('state')!='incomplete_preserved' or failed.get('error')!="ValueError('Predecessor PID changed identity')":
            raise ValueError('Expected preserved post-termination identity-check failure required')
        if state.get('state')!='interrupted' or state.get('error')!="KeyboardInterrupt('Signal 15')":
            raise ValueError('Predecessor did not acknowledge idle termination')
        require_absent(original['pid'],state,transfer.process_rows())
        for name,digest in plan['code_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen handoff binding changed: '+name)
        imported=transfer.REPORT/'result.json'
        if (u.read(transfer.REPORT/'status.json').get('state')!='complete' or
                u.read(imported)['plan_sha256']!=u.sha(transfer.REPORT/'plan.json')):
            raise ValueError('Completed auxiliary import required')
        boundary=state['current'];jobs=u.read(u.BASE/'registry.json')['jobs']
        u.validate_fresh()
        if not handoff.successor.core.receipt_valid(boundary,jobs):raise ValueError('Completed boundary receipt required')
        with (PRIMARY/'runner.lock').open('a') as primary_lock:
            fcntl.flock(primary_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require_absent(original['pid'],state,transfer.process_rows())
            recovery=dict(state='verified_absent_predecessor',original_launch=original,
                prior_failure=failed,completed_boundary_job=boundary,
                native_receipt_sha256=u.sha(u.BASE/'fresh-receipts'/(boundary+'.json')),
                handoff_plan_sha256=u.sha(REPORT/'plan.json'),script_sha256=u.sha(Path(__file__)),
                no_process_signals_sent=True,primary_runner_lock_available=True,
                scope='Recover after identity validation rejected the exiting predecessor. Absence, acknowledged TERM, no live wrapper group, unchanged source bindings and completed native receipt independently rechecked. Original frozen handoff and successor code unchanged.')
            u.write(REPORT/'recovery.json',recovery)
        # The verified successor acquires the same runner lock itself.
        with (REPORT/'successor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(handoff.successor.__file__))],
                cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        awake=subprocess.Popen(['caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        launch=dict(pid=child.pid,caffeinate_pid=awake.pid,utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            predecessor=original,script=str(Path(handoff.successor.__file__).relative_to(u.ROOT)),
            handoff_plan_sha256=u.sha(REPORT/'plan.json'),recovery_sha256=u.sha(REPORT/'recovery.json'))
        u.write(PRIMARY/'launch.json',launch)
        u.write(REPORT/'result.json',dict(state='successor_launched',launch=launch,
            auxiliary_import_sha256=u.sha(imported),completed_boundary_job=boundary,
            recovery_sha256=u.sha(REPORT/'recovery.json'),qualifying_labels_read=False))
        u.write(REPORT/'status.json',dict(state='successor_launched',pid=child.pid,goal_complete=False))
        return launch


if __name__=='__main__':print(run())
