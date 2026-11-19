#!/usr/bin/env python3
"""Reserve a primary-DAG idle boundary for the unchanged checkpointed CF96 run."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_cf96_memory_resume as memory
import upstream_full_reproduce as primary

u=memory.u
REPORT=u.ROOT/'experiments/upstream-cf96-primary-lane'
PRIMARY_SCRIPT=u.ROOT/'scripts/upstream_primary_guarded_aex.py'


def process(pid):
    output=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,pgid=,stat=,lstart=,command='],capture_output=True,text=True).stdout.strip()
    if not output:return None
    row=output.split(None,9)
    if len(row)!=10 or row[3].startswith('Z'):return None
    return dict(pid=int(row[0]),ppid=int(row[1]),pgid=int(row[2]),state=row[3],started=' '.join(row[4:9]),command=row[9])


def same_owner(left,right):
    return bool(right and all(left[key]==right[key] for key in ['pid','started','command']))


def children(pid):
    rows=subprocess.check_output(['ps','-axo','pid=,ppid=,stat=,command='],text=True).splitlines()
    return [row for row in rows if int(row.split(None,3)[1])==pid and not row.split(None,3)[2].startswith('Z')]


def stop_only_if_idle(pid,token,condition=None):
    before=process(pid)
    if not before or token not in before['command'] or 'T' in before['state']:
        raise ValueError('Expected running primary owner required')
    stopped=False
    try:
        os.kill(pid,signal.SIGSTOP);stopped=True
        after=process(pid)
        if not same_owner(before,after) or 'T' not in after['state']:
            raise RuntimeError('Primary stop identity differs')
        if children(pid) or (condition is not None and not condition()):return None
        # The owner is frozen and has no live child: SIGTERM cannot stop an
        # uncheckpointed job. Continue it only to execute its cleanup handler.
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False
        for _ in range(100):
            if not same_owner(before,process(pid)):return before
            time.sleep(.1)
        raise RuntimeError('Idle primary did not exit')
    finally:
        if stopped and same_owner(before,process(pid)):os.kill(pid,signal.SIGCONT)


def other_cf96():
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    tokens=[str(memory.SCRIPT),str(u.ROOT/'artifacts/upstream-cfnade-checkpoint/target/release/cfnade-stage')]
    return [row for row in rows if any(token in row for token in tokens)]


def restart_primary():
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    active=[row for row in rows if str(PRIMARY_SCRIPT) in row]
    if active:return dict(state='existing_primary_retained',actual_processes=active)
    with (REPORT/'primary-successor.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(PRIMARY_SCRIPT)],cwd=u.ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(state='primary_successor_launched',pid=child.pid,command=child.args,
                script_sha256=u.sha(PRIMARY_SCRIPT),observed=dt.datetime.now(dt.timezone.utc).isoformat())
    u.write(REPORT/'primary-successor.json',record);return record


def run(pid):
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'handoff.json').exists():raise ValueError('Existing lane handoff preserved')
        owner=process(pid)
        if not owner or str(PRIMARY_SCRIPT) not in owner['command']:raise ValueError('Actual primary owner required')
        if other_cf96():raise ValueError('Another CF96 owner already exists')
        states=memory.preflight();jobs=u.read(u.BASE/'registry.json')['jobs']
        u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
            script_sha256=u.sha(Path(__file__)),primary_owner=owner,
            primary_script_sha256=u.sha(PRIMARY_SCRIPT),cf96_runner_sha256=u.sha(memory.SCRIPT),
            initial_checkpoints=states,rule='Wait for an actual completed primary native job receipt and no live direct child; stop only that idle scheduler. Keep it absent through CF96 admission/training, then restart its unchanged source. Never stop an uncheckpointed native child.'))
        handed_off=False;cf96=None;primary_lock=None
        try:
            while True:
                if not same_owner(owner,process(pid)):raise RuntimeError('Primary owner changed before handoff')
                status=u.read(primary.REPORT/'status.json');job=status.get('current')
                busy=children(pid)
                u.write(REPORT/'status.json',dict(state='waiting_for_primary_idle_boundary',
                    observed=dt.datetime.now(dt.timezone.utc).isoformat(),primary_pid=pid,current_job=job,
                    actual_live_children=busy))
                if not busy and job in jobs and primary.receipt_valid(job,jobs):
                    # The original scheduler writes the next job name before
                    # Popen. Recheck it while frozen, closing the race between
                    # receipt hashing and the next launch (even before fork).
                    stopped=stop_only_if_idle(pid,str(PRIMARY_SCRIPT),
                        lambda:u.read(primary.REPORT/'status.json').get('current')==job)
                    if stopped:
                        handed_off=True
                        u.write(REPORT/'handoff.json',dict(state='verified_idle_primary_handoff',
                            owner=stopped,completed_job=job,native_receipt_sha256=u.sha(u.BASE/'fresh-receipts'/(job+'.json')),
                            no_live_child_at_stop=True,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                        primary_lock=(primary.REPORT/'runner.lock').open('a')
                        fcntl.flock(primary_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        break
                time.sleep(1)
            previous=memory.memory();good=0
            while good<2:
                current=memory.memory();owned=memory.footprints();total=sum(r['bytes'] for r in owned.values())
                disk=os.statvfs(u.ROOT);free=disk.f_bavail*disk.f_frsize
                good=good+1 if memory.admitted(previous,current,total) and free>28*2**30 else 0
                u.write(REPORT/'status.json',dict(state='waiting_for_stable_memory_with_primary_idle',
                    observed=dt.datetime.now(dt.timezone.utc).isoformat(),memory=current,owned_workers=owned,
                    owned_footprint_bytes=total,consecutive_good_samples=good))
                if good<2:previous=current;time.sleep(30)
            if other_cf96():raise RuntimeError('Another CF96 owner appeared')
            states=memory.preflight()
            with (u.ROOT/'experiments/upstream-cfnade-parallel/run.lock').open('a') as shared:
                fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'cf96.log').open('a') as log:
                cf96=subprocess.Popen([sys.executable,'-u',str(memory.SCRIPT)],cwd=u.ROOT,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            u.write(REPORT/'cf96-launch.json',dict(pid=cf96.pid,command=cf96.args,
                runner_sha256=u.sha(memory.SCRIPT),checkpoints=states,
                observed=dt.datetime.now(dt.timezone.utc).isoformat()))
            u.write(REPORT/'status.json',dict(state='original_cf96_running_with_primary_idle',pid=cf96.pid))
            if cf96.wait():raise RuntimeError('Original CF96 controller failed; checkpoint states preserved')
            u.write(REPORT/'status.json',dict(state='original_cf96_controller_complete',goal_requires_independent_review=True))
        finally:
            if cf96 is not None and cf96.poll() is None:
                primary.stop(cf96)
            if primary_lock is not None:
                fcntl.flock(primary_lock,fcntl.LOCK_UN);primary_lock.close()
            if handed_off:
                successor=restart_primary();u.write(REPORT/'restored-primary.json',successor)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-pid',type=int,required=True);a=p.parse_args()
    try:run(a.primary_pid)
    except BaseException as error:
        if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
        raise
