#!/usr/bin/env python3
"""Keep the pending full-size neural fit from overlapping a live CPU tree fit."""
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import time
import upstream_itemae_virtual_queue as pending
import upstream_latent90_local as memory
u=pending.u
REPORT=u.ROOT/'experiments/upstream-itemae-neural-admission'
QUEUE_PID=63754
EXCLUDED_GROUPS={63754,53156}  # Pending NN and serialized raw GPU epoch lane.

def processes():
    return [r.split(None,4) for r in subprocess.check_output(['ps','-axo','pid=,pgid=,rss=,stat=,command='],text=True).splitlines()]

def identity(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None

def sample():
    rows=processes();tree=[];groups=set()
    for r in rows:
        if len(r)!=5 or r[3].startswith('Z'):continue
        command=r[4];names=[Path(a).name for a in command.split()]
        if ('upstream_item154_tree.py' in names and ('--worker' in names or '--replay' in names)) or 'upstream_item154_tree_supervisor.py' in names:tree.append(int(r[0]))
        if '/target/release/' in command.split()[0] or str(u.ROOT/'.venv/bin/python') in command:groups.add(int(r[1]))
    groups-=EXCLUDED_GROUPS|{os.getpgrp()}
    return tree,sum(int(r[2])*1024 for r in rows if len(r)==5 and int(r[1]) in groups),memory.memory.memory()

def admitted(tree,other_rss,state,pair_ready):
    return bool(pair_ready and not tree and other_rss<8*2**30 and state['compressor_bytes']<5*2**30)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing neural admission preserved')
        original=identity(QUEUE_PID)
        if not original or 'upstream_itemae_virtual_queue.py' not in original:raise ValueError('Exact pending fixed NN queue required')
        if u.read(pending.REPORT/'status.json')['state']!='waiting_for_actual_predeclared_itemae_epoch5_10_pairs':raise ValueError('Neural queue already past its idle boundary')
        if (u.ROOT/'experiments/upstream-mps-item-autorec/pair-00010.json').exists():raise ValueError('Must establish gate before pair10 arrives')
        if any(int(r[1])==QUEUE_PID for r in [x.split(None,2) for x in subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines()] if len(r)==3):raise ValueError('Pending queue already has a child')
        u.write(REPORT/'plan.json',dict(created=pending.model.u.read(pending.REPORT/'plan.json')['created'],observed=pending.model.base.preceding.stamp() if hasattr(pending.model.base.preceding,'stamp') else __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),queue_pid=QUEUE_PID,queue_identity=original,
            original_queue_plan_sha256=u.sha(pending.REPORT/'plan.json'),script_sha256=u.sha(Path(__file__)),
            rule='Extra admission only: hold the still-idle fixed5/10 neural coordinator until all actual CPU tree fitting/replay/supervision processes finish, two stable samples show other jobs below8GiB RSS and compressor below5GiB, and paired10 is ready. Reserve16GiB for the observed full-size neural peak under the existing24GiB total guard. Raw GPU/native fitting continues. No numerical model/input/fold/seed/stopping/checkpoint/weight change.',qualifying_labels_read=False))
        os.kill(QUEUE_PID,signal.SIGSTOP)
        if identity(QUEUE_PID)!=original:raise ValueError('Pending queue identity changed')
        if any(int(r[1])==QUEUE_PID for r in [x.split(None,2) for x in subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines()] if len(r)==3):
            os.kill(QUEUE_PID,signal.SIGCONT);raise ValueError('A child appeared at admission boundary; coordinator resumed for investigation')
        good=0
        while good<2:
            if identity(QUEUE_PID)!=original:raise ValueError('Held neural coordinator missing or changed')
            tree,rss,state=sample();ready=(u.ROOT/'experiments/upstream-mps-item-autorec/pair-00010.json').exists()
            good=good+1 if admitted(tree,rss,state,ready) else 0
            u.write(REPORT/'status.json',dict(state='pending_neural_fit_held_for_complete_cpu_tree_and_peak_headroom',queue_pid=QUEUE_PID,actual_tree_pids=tree,
                other_jobs_rss_bytes=rss,memory=state,paired10_ready=ready,good_samples=good,observed=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
            if good<2:time.sleep(20)
        os.kill(QUEUE_PID,signal.SIGCONT)
        u.write(REPORT/'released.json',dict(state='same_original_neural_coordinator_released_after_safe_admission',queue_pid=QUEUE_PID,identity=identity(QUEUE_PID),original_queue_plan_sha256=u.sha(pending.REPORT/'plan.json'),numerical_recipe_unchanged=True,qualifying_labels_read=False))

if __name__=='__main__':run()
