#!/usr/bin/env python3
"""Prepare verified own166 context after warm59 replay, then queue its MPS study."""
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_latent90_local as resources

u=resources.u
REPORT=u.ROOT/'experiments/upstream-warm166-queue'

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=u.ROOT/'experiments/upstream-warm59-screen/ridge-verification.json'
        while not proof.exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_independently_replayed_warm59_ridge',observed=resources.stamp()));time.sleep(30)
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            previous=resources.memory.memory();good=0
            while good<2:
                current=resources.memory.memory();workers=resources.footprints();free=shutil.disk_usage(u.ROOT).free
                good=good+1 if (sum(v['bytes'] for v in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and
                    current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>28*2**30) else 0
                u.write(REPORT/'status.json',dict(state='waiting_for_known166_cache_memory_and_disk',observed=resources.stamp(),workers=workers,
                    memory=current,disk_free_bytes=free,good_samples=good))
                if good<2:previous=current;time.sleep(30)
            with (REPORT/'cache.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/upstream_warm166.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            checked=u.read(u.ROOT/'experiments/upstream-warm166/cache-column-verification.json')
            if checked['state']!='complete_cache_columns' or not checked['all_retained_and_new_columns_checked']:
                raise ValueError('Complete full-column cache check required')
        with (REPORT/'model.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(u.ROOT/'scripts/upstream_warm166_mps_context.py')],cwd=u.ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        u.write(REPORT/'status.json',dict(state='verified_cache_complete_mps_controller_started',pid=child.pid,observed=resources.stamp()))

if __name__=='__main__':run()
