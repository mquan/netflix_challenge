#!/usr/bin/env python3
"""Import completed own backfill artifacts at an idle primary-wrapper boundary."""
from contextlib import contextmanager
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import upstream_repro as u
import upstream_qualifying_backfill as auxiliary
import upstream_dnn_backfill as followon

REPORT=u.ROOT/'experiments/upstream-import-backfill'
PRIMARY=u.ROOT/'experiments/full-upstream-reproduction'
PRIMARY_FRAGMENT='scripts/upstream_full_reproduce.py'


def process_rows():
    rows=[]
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,stat=,command='],text=True).splitlines():
        fields=line.split(None,4)
        if len(fields)==5:rows.append(dict(pid=int(fields[0]),ppid=int(fields[1]),pgid=int(fields[2]),state=fields[3],command=fields[4]))
    return rows


@contextmanager
def paused_primary(pid):
    rows=process_rows();row=next((r for r in rows if r['pid']==pid),None)
    if row is None or PRIMARY_FRAGMENT not in row['command'] or 'T' in row['state']:
        raise ValueError('Primary PID is missing, changed, or already suspended')
    paused=False
    try:
        os.kill(pid,signal.SIGSTOP);paused=True
        for _ in range(100):
            current=next((r for r in process_rows() if r['pid']==pid),None)
            if current and 'T' in current['state']:break
            time.sleep(.01)
        else:raise RuntimeError('Primary suspension acknowledgement missing')
        yield
    finally:
        if paused:
            current=next((r for r in process_rows() if r['pid']==pid),None)
            if current and PRIMARY_FRAGMENT in current['command']:os.kill(pid,signal.SIGCONT)


def busy_primary(pid,state):
    group=state.get('child_pid')
    return any('Z' not in row['state'] and (row['ppid']==pid or (group and row['pgid']==group)) for row in process_rows())


def validate_record(name,record,fresh,jobs):
    if name not in jobs or record['job']!=name or record['source_commit']!=u.PIN or record['command']!=jobs[name]['command']:
        raise ValueError('Native import recipe mismatch: '+name)
    if not set(jobs[name]['outputs']).issubset(record['outputs']):raise ValueError('Incomplete native import receipt: '+name)
    for relative,digest in record['outputs'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or not relative.startswith('preds_new/'):
            raise ValueError('Import output outside the prediction tree')
        path=fresh/p
        if path.is_symlink() or not path.is_file() or u.sha(path)!=digest:
            raise ValueError('Native import output changed: '+relative)


def transfer_plan(source,source_receipts,target,target_receipts,jobs):
    entries=[]
    for path in sorted(source_receipts.glob('*.json')):
        name=path.stem;record=u.read(path);validate_record(name,record,source,jobs)
        destination=target_receipts/path.name
        if destination.exists():validate_record(name,u.read(destination),target,jobs)
        for relative,digest in record['outputs'].items():
            existing=target/relative
            if existing.exists() and (existing.is_symlink() or not existing.is_file() or u.sha(existing)!=digest):
                raise ValueError('Existing primary output conflicts: '+relative)
        entries.append(dict(job=name,source_receipt=str(path),source_receipt_sha256=u.sha(path),
            new_receipt=not destination.exists(),outputs=record['outputs']))
    if not entries:raise ValueError('No completed auxiliary native receipts')
    return entries


def apply_transfers(entries,source,target,target_receipts,jobs):
    completed=[];target_receipts.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        path=Path(entry['source_receipt']);name=entry['job'];destination=target_receipts/path.name
        if u.sha(path)!=entry['source_receipt_sha256']:raise ValueError('Auxiliary receipt changed before transfer')
        record=u.read(path);validate_record(name,record,source,jobs)
        if destination.exists():
            validate_record(name,u.read(destination),target,jobs);continue
        created=[]
        try:
            for relative,digest in record['outputs'].items():
                existing=target/relative
                if existing.exists():
                    if existing.is_symlink() or u.sha(existing)!=digest:raise ValueError('Primary output changed during transfer')
                else:
                    existing.parent.mkdir(parents=True,exist_ok=True)
                    os.link(source/relative,existing);created.append((existing,digest))
            validate_record(name,record,target,jobs)
            imported=dict(record,imported_from=dict(execution_tree=str(source),
                native_receipt=str(path),native_receipt_sha256=entry['source_receipt_sha256'],
                transfer_script_sha256=u.sha(Path(__file__)),
                scope='Own completed pinned-source execution in the auxiliary tree; identical input-data aliases. Artifact reuse, not a new execution in this directory.'))
            u.write(destination,imported);completed.append(name)
        except BaseException:
            # A committed receipt publishes a complete job. Preserve that job;
            # roll back only newly linked files from an unpublished job.
            if not destination.exists():
                for existing,digest in reversed(created):
                    if existing.is_file() and u.sha(existing)==digest:existing.unlink()
            raise
    return completed


def verify_source():
    u.validate_fresh()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=auxiliary.TREE,text=True).strip()!=u.PIN:
        raise ValueError('Auxiliary source pin changed')
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=auxiliary.TREE,text=True).splitlines()
    sentinel='data/qual_ratings/qual_ratings.csv.gz'
    if set(changed)-{sentinel} or u.sha(auxiliary.TREE/sentinel)!=u.sha(u.FRESH/sentinel):
        raise ValueError('Auxiliary code or zero-qualifying sentinel changed')
    for split in ['raw','train','probe','trainx','probex','fulltrain','qual']:
        if (auxiliary.TREE/'data'/split).resolve()!=(u.FRESH/'data'/split).resolve():
            raise ValueError('Auxiliary inputs are not the identical primary data aliases')
    for name in ['registry.json','fresh-training-plan.json','fresh-data-receipt.json']:
        if u.sha(auxiliary.BASE/name)!=u.sha(u.BASE/name):raise ValueError('Auxiliary setup identity changed')


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'result.json').exists():
            result=u.read(REPORT/'result.json');plan=u.read(REPORT/'plan.json')
            if result['plan_sha256']!=u.sha(REPORT/'plan.json') or plan['script_sha256']!=u.sha(Path(__file__)):
                raise ValueError('Completed import plan changed')
            verify_source();jobs=u.read(u.BASE/'registry.json')['jobs']
            for entry in plan['entries']:validate_record(entry['job'],u.read(u.BASE/'fresh-receipts'/(entry['job']+'.json')),u.FRESH,jobs)
            return
        while u.read(followon.REPORT/'status.json').get('state')!='complete':
            u.write(REPORT/'status.json',dict(state='waiting_for_complete_auxiliary_run'));time.sleep(30)
        with (auxiliary.REPORT/'run.lock').open('a') as source_lock:
            fcntl.flock(source_lock,fcntl.LOCK_EX)
            verify_source();jobs=u.read(u.BASE/'registry.json')['jobs']
            entries=transfer_plan(auxiliary.TREE,auxiliary.BASE/'fresh-receipts',u.FRESH,u.BASE/'fresh-receipts',jobs)
            u.write(REPORT/'plan.json',dict(script_sha256=u.sha(Path(__file__)),source_tree=str(auxiliary.TREE),
                target_tree=str(u.FRESH),entries=entries,qualifying_labels_read=False))
            pid=u.read(PRIMARY/'launch.json')['pid']
            while True:
                state=u.read(PRIMARY/'status.json');name=state.get('current')
                if not name or not (u.BASE/'fresh-receipts'/(name+'.json')).exists():
                    u.write(REPORT/'status.json',dict(state='waiting_for_primary_receipt_boundary',primary_pid=pid,current=name));time.sleep(2);continue
                # Suspend only the idle scheduler. A racing next wrapper causes
                # an immediate retry; no active training wrapper is signalled.
                with paused_primary(pid):
                    latest=u.read(PRIMARY/'status.json')
                    if busy_primary(pid,latest):continue
                    u.write(REPORT/'status.json',dict(state='importing_at_idle_boundary',primary_pid=pid,current=name))
                    verify_source()
                    completed=apply_transfers(entries,auxiliary.TREE,u.FRESH,u.BASE/'fresh-receipts',jobs)
                    for entry in entries:validate_record(entry['job'],u.read(u.BASE/'fresh-receipts'/(entry['job']+'.json')),u.FRESH,jobs)
                    u.write(REPORT/'result.json',dict(state='complete',imported_jobs=completed,
                        reused_existing_receipts=len(entries)-len(completed),primary_pid=pid,
                        plan_sha256=u.sha(REPORT/'plan.json'),qualifying_labels_read=False,
                        scope='Completed own native artifacts imported with source lineage; no active wrapper scan was modified. Primary scheduler resumes in the finally block.'))
                u.write(REPORT/'status.json',dict(state='complete',imported_jobs=completed,primary_resumed=True));return


if __name__=='__main__':
    try:run()
    except BaseException as error:
        if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error),primary_resume_attempted_by_finally=True))
        raise
