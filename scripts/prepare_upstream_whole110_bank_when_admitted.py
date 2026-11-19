#!/usr/bin/env python3
"""Queue paired110 preparation behind verified paired90 and development inputs."""
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_latent90_local as resources

u=resources.u
REPORT=u.ROOT/'experiments/upstream-whole110-bank-queue'
REQUIRED=[u.ROOT/'experiments'/name/'column-verification.json' for name in ['upstream-latent90-bank']]
REQUIRED += [u.ROOT/'experiments/upstream-whole110/cache-column-verification.json']

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not all(p.exists() for p in REQUIRED):
            u.write(REPORT/'status.json',dict(state='waiting_for_verified_paired90_and_whole110',observed=resources.stamp(),
                missing=[str(p.relative_to(u.ROOT)) for p in REQUIRED if not p.exists()]))
            time.sleep(30)
        previous=resources.memory.memory();good=0
        while good<2:
            current=resources.memory.memory();workers=resources.footprints();free=shutil.disk_usage(u.ROOT).free
            good=good+1 if (sum(v['bytes'] for v in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and
                current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>24*2**30) else 0
            u.write(REPORT/'status.json',dict(state='waiting_for_paired110_memory',observed=resources.stamp(),workers=workers,
                memory=current,good_samples=good,disk_free_bytes=free))
            if good<2:previous=current;time.sleep(30)
        for script in ['upstream_whole110_bank.py','verify_upstream_whole110_bank.py']:
            with (REPORT/(script+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        u.write(REPORT/'status.json',dict(state='paired110_complete_and_independently_column_verified',observed=resources.stamp()))

if __name__=='__main__':run()
