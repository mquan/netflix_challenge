#!/usr/bin/env python3
"""Replace only an idle primary scheduler after completed auxiliary import."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_import_backfill as transfer
import upstream_primary_guarded as successor

u = transfer.u
REPORT = successor.REPORT
PRIMARY = transfer.PRIMARY


def scheduler_present(pid):
    row = next((r for r in transfer.process_rows() if r['pid']==pid), None)
    if row is None or 'Z' in row['state']:
        return False
    if transfer.PRIMARY_FRAGMENT not in row['command']:
        raise ValueError('Predecessor PID changed identity')
    return True


def terminate_idle(pid, state):
    """Never send termination if an immediate child or wrapper group is live."""
    with transfer.paused_primary(pid):
        if transfer.busy_primary(pid, state):
            return False
        os.kill(pid, signal.SIGTERM)
    # paused_primary sends CONT even on error; the old Python signal handler
    # handles TERM only after resuming. Its child is confirmed absent here.
    for _ in range(100):
        if not scheduler_present(pid):
            return True
        time.sleep(.1)
    raise RuntimeError('Idle predecessor has not exited; no successor launched')


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(successor.__file__), Path(transfer.__file__),
             Path(successor.core.__file__)]
    plan = dict(code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in files},
        original_launch=u.read(PRIMARY/'launch.json'), source_commit=u.PIN,
        storage_holds='Every tsvdx4p-* native heap job; existing large serial models remain held.',
        delegated_source_jobs=sorted(successor.DELEGATED),
        model_policy='Original ranks, epochs and source recipes unchanged. Separate native workers and mapped backends remain responsible for held models.',
        safety='Wait for completed own auxiliary import and a hash-verified completed primary job. Stop only the idle scheduler with no live child/wrapper group. Resume it in finally; wait for its exit and exclusive scheduler lock before launching successor.',
        qualifying_labels_read=False)
    path=REPORT/'plan.json'
    if path.exists():
        if u.read(path)!=plan:
            raise ValueError('Existing handoff plan differs; preserve it')
    else:
        u.write(path,plan)
    return plan


def run():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'result.json').exists():
            raise ValueError('Handoff already published; inspect the successor instead')
        plan=prepare()
        pid=plan['original_launch']['pid']
        while True:
            if not scheduler_present(pid):
                raise RuntimeError('Predecessor ended before handoff; inspect preserved state')
            imported=transfer.REPORT/'result.json'
            if not imported.exists() or u.read(transfer.REPORT/'status.json').get('state')!='complete':
                u.write(REPORT/'status.json',dict(state='waiting_for_completed_auxiliary_import',primary_pid=pid))
                time.sleep(10)
                continue
            result=u.read(imported)
            if result['plan_sha256']!=u.sha(transfer.REPORT/'plan.json'):
                raise ValueError('Completed auxiliary import binding changed')
            state=u.read(PRIMARY/'status.json'); name=state.get('current')
            jobs=u.read(u.BASE/'registry.json')['jobs']
            if not name or not successor.core.receipt_valid(name,jobs):
                u.write(REPORT/'status.json',dict(state='waiting_for_idle_receipt_boundary',primary_pid=pid,current=name))
                time.sleep(2)
                continue
            for relative,digest in plan['code_sha256'].items():
                if u.sha(u.ROOT/relative)!=digest:raise ValueError('Frozen handoff code changed before suspension')
            # Re-read status under suspension in terminate_idle's own check of
            # immediate children. A next-wrapper race causes a harmless retry.
            if terminate_idle(pid,state):
                break
            time.sleep(2)
        with (PRIMARY/'runner.lock').open('a') as primary_lock:
            fcntl.flock(primary_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Verify frozen scripts before launching the new scheduler. No native
        # prediction tree is modified by this handoff.
        for name,digest in plan['code_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen handoff code changed')
        with (REPORT/'successor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(successor.__file__))],
                cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        awake=subprocess.Popen(['caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        launch=dict(pid=child.pid,caffeinate_pid=awake.pid,utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            predecessor=plan['original_launch'],script=str(Path(successor.__file__).relative_to(u.ROOT)),
            handoff_plan_sha256=u.sha(REPORT/'plan.json'))
        u.write(PRIMARY/'launch.json',launch)
        u.write(REPORT/'result.json',dict(state='successor_launched',launch=launch,
            auxiliary_import_sha256=u.sha(imported),completed_boundary_job=name,
            qualifying_labels_read=False))
        u.write(REPORT/'status.json',dict(state='successor_launched',pid=child.pid,goal_complete=False))


if __name__=='__main__':
    try:
        run()
    except BaseException as error:
        if not isinstance(error,BlockingIOError):
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
        raise
