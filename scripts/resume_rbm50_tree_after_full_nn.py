#!/usr/bin/env python3
"""Operational checkpoint handoff; original fitting code/config stay frozen."""
from pathlib import Path
import subprocess
import sys
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-rbm50-tree'
RUN=u.ROOT/'artifacts/upstream-rbm50-tree'

def identity(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat=,lstart=,command='],capture_output=True,text=True).stdout.strip()
    # Scheduling state changes between R and S are not an owner exit.
    return None if not result or result.startswith('Z') else result.split(None,1)[1]

def run():
    plan=u.read(REPORT/'priority-handoff-plan.json')
    while any(identity(int(pid))==expected.split(None,1)[1] for pid,expected in plan['owners'].items()):time.sleep(5)
    if (REPORT/'evaluation.json').exists() and (REPORT/'tree-replay.json').exists():
        u.write(REPORT/'priority-resume.json',dict(state='original_study_already_complete'));return
    for name,digest in plan['completed_fold0_outputs'].items():
        if u.sha(RUN/'tree/0'/name)!=digest:raise ValueError('Preserved completed tree fold changed')
    if (RUN/'tree/1').exists():raise ValueError('Incomplete fold1 was not archived; preserve for inspection')
    script=u.ROOT/'scripts/upstream_rbm50_tree.py'
    if u.sha(script)!=plan['original_script_sha256']:raise ValueError('Frozen original fitting implementation changed')
    with (REPORT/'launch-after-full-nn-priority.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    u.write(REPORT/'priority-resume.json',dict(state='original_tree_controller_resumed',pid=child.pid,
        completed_folds_preserved=[0],original_script_sha256=u.sha(script),
        scope='Same original model code, cache, manifest, folds and seeds. Reuses the actual completed fold0 receipt; no qualifying outcome is read to configure or select this remaining fit.'))
    print(child.pid,flush=True)

if __name__=='__main__':run()
