#!/usr/bin/env python3
"""Extra12GiB peak reservation for the still-idle full tree production queue."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import upstream_item154_tree_production_recovered as model
u=model.u
REPORT=u.ROOT/'experiments/upstream-item154-tree-production-admission-recovered'
QUEUE=3709

def identity(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else None

def admitted(ready,original_tree_live,other_rss,memory,free):
    return ready and not original_tree_live and other_rss<12*2**30 and memory['compressor_bytes']<5*2**30 and free>10*2**30

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing extra admission preserved')
        owner=identity(QUEUE)
        if not owner or 'upstream_item154_tree_production_recovered.py' not in owner:raise ValueError('Original full-tree queue required')
        if u.read(model.REPORT/'status.json')['state']!='waiting_for_complete_independent_development':raise ValueError('Full-tree queue already past idle declaration')
        if any(int(r.split()[1])==QUEUE for r in subprocess.check_output(['ps','-axo','pid=,ppid='],text=True).splitlines()):raise ValueError('Queue already has a fitting child')
        evidence=u.read(u.ROOT/'experiments/upstream-item154-tree-production-admission/plan.json')['observed_development_peak_evidence']
        u.write(REPORT/'plan.json',dict(created=model.binding.stamp(),queue_pid=QUEUE,queue_identity=owner,
            original_declaration_sha256=u.sha(model.REPORT/'declaration.json'),script_sha256=u.sha(Path(__file__)),observed_development_peak_evidence=evidence,
            rule='Extra admission only, before any full fit. Hold the same idle production coordinator. After complete original development/replay, release a nonselected result to validate and end without fitting; otherwise require original tree worker/supervisor gone and two stable samples of all other own jobs below12GiB RSS,compressor below5GiB and disk above10GiB. Reserve12GiB for full-known tree peak under unchanged24GiB total guard. Count raw GPU jobs too. No numerical code,models,inputs,budgets,seeds or weights changed.',qualifying_labels_read=False))
        os.kill(QUEUE,signal.SIGSTOP)
        if identity(QUEUE)!=owner:raise ValueError('Original queue identity changed')
        u.write(REPORT/'held.json',dict(state='same_idle_full_tree_coordinator_held_before_fit',queue_pid=QUEUE,identity=owner,observed=model.binding.stamp()))
        good=0
        while good<2:
            if identity(QUEUE)!=owner:raise ValueError('Held full-tree coordinator disappeared or changed')
            ready=model.SUPERVISOR.exists();not_selected=False
            if ready:
                score=next(r['rmse'] for r in u.read(model.development.REPORT/'evaluation.json')['records'] if r['arm']=='fixed-half-tree-nn')
                not_selected=score>=u.read(model.REPORT/'declaration.json')['reference_rmse']
            old=any(identity(pid) is not None for pid in [60801,62158])
            other=model.development.template.parent.resources.other_rss({os.getpgrp(),QUEUE})
            memory=model.development.resources.memory.memory();free=__import__('shutil').disk_usage(u.ROOT).free
            good=good+1 if not_selected or admitted(ready,old,other,memory,free) else 0
            u.write(REPORT/'status.json',dict(state='reserving_full_known_tree_peak_before_original_queue_release',queue_pid=QUEUE,
                development_complete=ready,original_tree_live=old,nonselected_validation_only=not_selected,other_rss_bytes=other,memory=memory,disk_free_bytes=free,
                good_samples=good,observed=model.binding.stamp()))
            if good<2:time.sleep(20)
        if u.sha(model.REPORT/'declaration.json')!=u.read(REPORT/'plan.json')['original_declaration_sha256']:raise ValueError('Held original declaration changed')
        os.kill(QUEUE,signal.SIGCONT)
        u.write(REPORT/'released.json',dict(state='same_original_queue_released_after_stronger_peak_gate',queue_pid=QUEUE,identity=identity(QUEUE),
            numerical_recipe_unchanged=True,extra_admission_plan_sha256=u.sha(REPORT/'plan.json'),observed=model.binding.stamp()))

def test():
    memory=dict(compressor_bytes=4*2**30)
    assert admitted(True,False,11*2**30,memory,11*2**30)
    assert not admitted(False,False,11*2**30,memory,11*2**30)
    assert not admitted(True,True,11*2**30,memory,11*2**30)
    assert not admitted(True,False,12*2**30,memory,11*2**30)
    assert not admitted(True,False,11*2**30,dict(compressor_bytes=5*2**30),11*2**30)
    assert not admitted(True,False,11*2**30,memory,10*2**30)
    REPORT.mkdir(parents=True,exist_ok=True)
    u.write(REPORT/'test-verification.json',dict(state='complete_extra_peak_boundary_checks',cases=6,script_sha256=u.sha(Path(__file__)),observed=model.binding.stamp()))
    print('PASS: ready/live-worker/RSS/compressor/disk boundaries retain12GiB peak reserve')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');args=ap.parse_args()
    test() if args.test else run()
