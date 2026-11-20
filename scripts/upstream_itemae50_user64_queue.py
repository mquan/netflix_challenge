#!/usr/bin/env python3
"""Wait for actual full source completion, then supervise fixed context extraction."""
from pathlib import Path
import os
import subprocess
import sys
import shutil
import time
import upstream_itemae50_user64_support as source
import upstream_knn3pair2_virtual_queue as resource
u=source.u
REPORT=u.ROOT/'experiments/upstream-itemae50-user64-queue'

def run_worker(script,logname):
    with (REPORT/logname).open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(script)],cwd=u.ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=resource.model.base.preceding.group_rss(child.pid)
                other=resource.model.base.preceding.resources.other_rss({child.pid})
                free=shutil.disk_usage(u.ROOT).free
                high=high+1 if own>5*2**30 or own+other>24*2**30 else 0
                u.write(REPORT/'status.json',dict(state='full_cpu_context_extraction_or_verification',
                    observed=source.stamp(),script=str(script.relative_to(u.ROOT)),pid=child.pid,
                    own_rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free))
                if high>=3 or free<8*2**30:raise RuntimeError('Preserve full source and partial extraction: original resource guard')
                time.sleep(5)
        except BaseException:
            resource.model.base.preceding.stop(child);raise
        if child.returncode:raise RuntimeError('Full fixed context worker failed; no smaller retry')

def main():
    if REPORT.exists():raise ValueError('Existing fixed context queue preserved')
    REPORT.mkdir()
    files=[Path(__file__),Path(source.__file__),u.ROOT/'scripts/verify_upstream_itemae50_user64_support.py',
        u.ROOT/'ITEMAE50_USER64_PLAN.md']
    hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    u.write(REPORT/'plan.json',dict(created=source.stamp(),source_sha256=hashes,
        original_raw_epochs_each=50,users=480189,original_width=128,rank_each=32,
        rule='Wait for complete50/50 raw source and independent final snapshot proof. Full CPU-only encoder/decoder context projection and independent every-user/query replay. No model change, outcomes, automatic qualifying production or hidden row/capacity reduction. FiveGiB own reserve under24GiB total guard, disk8GiB continuous minimum.',qualifying_labels_read=False))
    required=[source.raw.REPORT/'completion.json',source.raw.REPORT/'pair-00050.json',
        source.raw.REPORT/'snapshot-00050-verification.json']
    while not all(p.exists() for p in required):
        u.write(REPORT/'status.json',dict(state='waiting_for_actual_complete50_50_source_and_independent_proof',
            observed=source.stamp(),ready=[str(p.relative_to(u.ROOT)) for p in required if p.exists()]))
        time.sleep(30)
    previous=resource.model.base.memory.memory()
    while True:
        current=resource.model.base.memory.memory()
        other=resource.model.base.preceding.resources.other_rss({os.getpgrp()})
        free=shutil.disk_usage(u.ROOT).free
        if (other<19*2**30 and current['compressor_bytes']<5*2**30 and
            current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>12*2**30):break
        u.write(REPORT/'status.json',dict(state='waiting_for_full_cpu_context_headroom',observed=source.stamp(),
            other_rss_bytes=other,memory=current,disk_free_bytes=free))
        previous=current;time.sleep(30)
    for name,digest in hashes.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared context extraction changed')
    run_worker(Path(source.__file__),'extraction.log')
    run_worker(u.ROOT/'scripts/verify_upstream_itemae50_user64_support.py','independent.log')
    proof=u.read(source.REPORT/'verification.json')
    if proof['state']!='complete_independent_itemae50_user_projection_and_all_query_rows':raise ValueError('Complete context replay required')
    u.write(REPORT/'completion.json',dict(observed=source.stamp(),
        verification_sha256=u.sha(source.REPORT/'verification.json'),qualifying_labels_read=False,
        scope='Verified support only; no ensemble score or qualifying promotion claimed.'))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        REPORT.mkdir(parents=True,exist_ok=True)
        u.write(REPORT/'failure.json',dict(error=repr(error),artifacts_preserved=True));raise
