#!/usr/bin/env python3
"""Retire only idle original return coordinators; preserve every scientific state."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import upstream_remote_weekend_override_runtime as r


def alive(record):
    actual=r.identity.process(record['pid'])
    return bool(r.identity.same_owner(record,actual) and 'Z' not in actual['state'])


def old_report(job):return r.ROOT/'experiments/upstream-remote-weekend-return-queue'/job


def destination(job):return r.ROOT/'experiments/upstream-remote-weekend-override-idle-handoff'/job


def watchdogs(job,owner):
    target=str(r.ROOT/'scripts/upstream_remote_weekend_return_queue.py');found=[]
    text=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    for line in text.splitlines():
        pieces=line.strip().split(None,1)
        if len(pieces)!=2:continue
        try:args=shlex.split(pieces[1])
        except ValueError:continue
        if target in args and '--watch' in args and '--job' in args:
            if args[args.index('--watch')+1]==str(owner['pid']) and args[args.index('--job')+1]==job:
                actual=r.identity.process(int(pieces[0]))
                if actual and 'Z' not in actual['state']:found.append(actual)
    return found


def only_watchdog_child(owner,watches):
    children=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True).splitlines()
    child_ids={int(x.split()[0]) for x in children if len(x.split())==2 and int(x.split()[1])==owner['pid']}
    if len(watches)!=1 or child_ids!={watches[0]['pid']}:
        raise ValueError('Unexpected original child; no retirement')


def assert_idle(job,owner):
    old=old_report(job);args=shlex.split(owner['command'])
    target=str(r.ROOT/'scripts/upstream_remote_weekend_return_queue.py')
    if (target not in args or '--job' not in args or args[args.index('--job')+1]!=job
        or '--watch' in args or '--bootstrap' in args):
        raise ValueError('Exact original idle return owner command required')
    _,science,art=r.paths(job)
    forbidden=['first-scientific-baseline.json','child-ownership.json','pinned-remote.json','completion.json',
               *[phase+suffix for phase in r.PHASES for suffix in ('-ownership.json','-gate.json','-dispatch.json','.log')]]
    if science.exists() or art.exists() or any((old/name).exists() for name in forbidden):
        raise ValueError('Scientific namespace/baseline/worker exists; idle-only retirement refused')
    plan=r.read(old/'plan.json');r.check_bindings(plan['source_sha256'])
    if plan['job']!=job:raise ValueError('Exact original producer declaration required')
    status=r.read(old/'status.json')
    if alive(owner) and status['state']!='waiting_full_producer_and_independent_proof':
        raise ValueError('Only the original own15-waiting idle state may be retired')
    if not alive(owner) and (old/'failure.json').exists() and r.read(old/'failure.json').get('child') is not None:
        raise ValueError('Terminal original owner retained a scientific child')


def retire(job):
    if job not in r.JOBS:raise ValueError('Original producer required')
    r.check_protocol();old=old_report(job);owner=r.read(old/'monitor.json');assert_idle(job,owner)
    target=destination(job)
    if target.exists():raise ValueError('Existing handoff attempt retained')
    watches=watchdogs(job,owner)
    if alive(owner):
        if 'T' in r.identity.process(owner['pid'])['state']:
            raise ValueError('Previously paused original owner retained')
        only_watchdog_child(owner,watches)
    r.durable_directory(target)
    r.write(target/'preparation.json',dict(state='declared_exact_idle_only_control_retirement',job=job,
        owner=owner,watchdogs=watches,no_scientific_namespace_or_baseline=True,
        original_monitor_sha256=r.ops.sha(old/'monitor.json'),original_plan_sha256=r.ops.sha(old/'plan.json'),
        protocol_sha256=r.ops.sha(r.PROTOCOL),observed=r.stamp()),exclusive=True)
    if alive(owner):
        frozen=False;terminating=False
        try:
            os.kill(owner['pid'],signal.SIGSTOP);frozen=True
            limit=time.monotonic()+2
            while True:
                current=r.identity.process(owner['pid'])
                if not r.identity.same_owner(owner,current):raise ValueError('Idle owner identity changed')
                if 'T' in current['state']:break
                if time.monotonic()>limit:raise ValueError('Same idle owner did not freeze')
                time.sleep(.02)
            assert_idle(job,owner);only_watchdog_child(owner,watches)
            os.kill(owner['pid'],signal.SIGTERM);terminating=True
            os.kill(owner['pid'],signal.SIGCONT)  # Deliver termination to this verified idle coordinator only.
            frozen=False
        finally:
            # Restore only the running idle control process paused by this attempt.
            # A failed pre-termination check must not strand its scheduler.
            if frozen and not terminating and r.identity.same_owner(owner,r.identity.process(owner['pid'])):
                os.kill(owner['pid'],signal.SIGCONT)
    limit=time.monotonic()+15
    while alive(owner) or any(alive(w) for w in watches):
        if time.monotonic()>limit:raise RuntimeError('Idle retirement not terminal; no kill/retry')
        time.sleep(.2)
    assert_idle(job,owner)
    bindings={str(p.relative_to(r.ROOT)):r.ops.sha(p) for p in
        [r.PROTOCOL,Path(__file__),old/'plan.json',old/'monitor.json',old/'status.json',target/'preparation.json']}
    for name in ('failure.json','watchdog-recovery.json'):
        p=old/name
        if p.exists():bindings[str(p.relative_to(r.ROOT))]=r.ops.sha(p)
    r.write(target/'verification.json',dict(state='complete_exact_idle_original_return_coordinator_handoff',job=job,
        owner=owner,watchdogs=watches,all_original_control_processes_terminal=True,
        no_scientific_namespace_baseline_or_child_created=True,source_sha256=bindings,
        scientific_processes_signalled=False,models_or_numerical_files_changed=False,observed=r.stamp()),exclusive=True)
    print('Original idle return control retired:',job,flush=True)


def verify_handoff(job):
    p=destination(job)/'verification.json';v=r.read(p)
    if (v['state']!='complete_exact_idle_original_return_coordinator_handoff' or v['job']!=job
        or v['all_original_control_processes_terminal'] is not True
        or v['no_scientific_namespace_baseline_or_child_created'] is not True
        or v['scientific_processes_signalled'] is not False or v['models_or_numerical_files_changed'] is not False
        or alive(v['owner']) or any(alive(w) for w in v['watchdogs'])):
        raise ValueError('Full verified idle-only handoff and terminal original handles required')
    r.check_bindings(v['source_sha256']);return p


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',choices=r.JOBS,required=True);p.add_argument('--retire-idle',action='store_true')
    a=p.parse_args()
    if a.retire_idle:retire(a.job)
    else:print(verify_handoff(a.job))
