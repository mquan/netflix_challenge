#!/usr/bin/env python3
"""Build the paired90 cache only after measured native/meta memory headroom."""
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_latent90_local as resources

u=resources.u
REPORT=u.ROOT/'experiments/upstream-latent90-bank-queue'

def run():
    previous=resources.memory.memory();good=0
    while good<2:
        current=resources.memory.memory();workers=resources.footprints();free=shutil.disk_usage(u.ROOT).free
        good=good+1 if (sum(v['bytes'] for v in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and
            current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>24*2**30) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_paired90_memory',observed=resources.stamp(),workers=workers,memory=current,good_samples=good,disk_free_bytes=free))
        if good<2:previous=current;time.sleep(30)
    for script in ['upstream_latent90_bank.py','verify_upstream_latent90_bank.py']:
        subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,check=True)
    u.write(REPORT/'status.json',dict(state='paired90_complete_and_independently_column_verified',observed=resources.stamp()))

if __name__=='__main__':run()
