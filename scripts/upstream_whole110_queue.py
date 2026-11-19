#!/usr/bin/env python3
"""Await all three whole native proofs, check fixed batch, then queue larger NN."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_whole110_screen as screen
import upstream_latent90_local as resources

u=screen.u
REPORT=u.ROOT/'experiments/upstream-whole110-queue'

def identity(pid):
    row=subprocess.run(['ps','-p',str(pid),'-o','stat=,lstart=,command='],capture_output=True,text=True).stdout.strip()
    return None if not row or row.startswith('Z') else row.split(None,1)[1]

def retire_only_idle_prior():
    pid=8182;before=identity(pid)
    if not before or not before.endswith('scripts/upstream_latent90_wide_context.py'):return dict(state='prior_owner_absent')
    os.kill(pid,signal.SIGSTOP);retired=False
    try:
        children=[r for r in subprocess.check_output(['ps','-axo','pid=,ppid='],text=True).splitlines() if r.split()[1]==str(pid)]
        status=u.read(u.ROOT/'experiments/upstream-latent90-wide-context/status.json')
        model_cache=u.ROOT/'artifacts/upstream-latent90-wide-context/nn'
        if identity(pid)!=before or children or model_cache.exists() or not status['state'].startswith('waiting_'):
            return dict(state='prior_actual_work_preserved',identity=before,children=children,status=status)
        os.kill(pid,signal.SIGTERM);retired=True
        return dict(state='verified_idle_prior_retired_for_batched110',identity=before,no_child=True,no_nn_model_cache=True,status=status)
    finally:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
        if retired:
            for _ in range(20):
                if identity(pid)!=before:break
                time.sleep(.25)

def run():
    while not all(p.exists() for p in screen.PROOFS.values()):
        u.write(REPORT/'status.json',dict(state='waiting_for_three_complete_native_proofs',observed=resources.stamp(),missing=[str(p.relative_to(u.ROOT)) for p in screen.PROOFS.values() if not p.exists()]));time.sleep(30)
    previous=resources.memory.memory();good=0
    while good<2:
        current=resources.memory.memory();workers=resources.footprints()
        good=good+1 if (sum(r['bytes'] for r in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and
            current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and shutil.disk_usage(u.ROOT).free>24*2**30) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_whole110_preparation_memory',observed=resources.stamp(),workers=workers,memory=current,good_samples=good))
        if good<2:previous=current;time.sleep(30)
    for script in ['upstream_whole110_screen.py','verify_upstream_whole110_screen.py','upstream_whole110.py']:
        with (REPORT/(script+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    proof=u.read(u.ROOT/'experiments/upstream-whole110/cache-column-verification.json')
    if proof['state']!='complete_cache_columns' or not proof['all_retained_and_new_columns_checked']:raise ValueError('Complete verified whole110 cache required')
    handoff=retire_only_idle_prior();u.write(REPORT/'prior-idle-handoff.json',handoff)
    with (REPORT/'nn-launch.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(u.ROOT/'scripts/upstream_whole110_wide_context.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    u.write(REPORT/'status.json',dict(state='verified_whole110_cache_and_larger_nn_controller_started',pid=child.pid,observed=resources.stamp(),handoff=handoff))

if __name__=='__main__':run()
