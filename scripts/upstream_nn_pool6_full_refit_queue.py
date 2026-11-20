#!/usr/bin/env python3
"""Serial, guarded full refits; incomplete artifacts are never silently retried."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-nn-pool6-full-refit-queue'
DECLARATION=u.ROOT/'experiments/upstream-native-nn-pool6-production/declaration.json'
WORKER=u.ROOT/'scripts/upstream_nn_pool6_full_refit.py'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    if REPORT.exists():raise ValueError('Existing execution preserved')
    REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(created=stamp(),declaration_sha256=u.sha(DECLARATION),
        worker_sha256=u.sha(WORKER),queue_sha256=u.sha(Path(__file__)),
        rule='Serial unchanged full models, 24GiB research RSS / 11GiB child sustained guard,16GiB disk floor. Wait for completed isolated tree input audit. No qualifying outcomes or automatic score promotion.'))
    proof=u.ROOT/'experiments/upstream-item154-tree-realization-sensitivity/verification.json'
    while not proof.exists():
        # The independently launched tree audit holds the same compute lock.
        u.write(REPORT/'status.json',dict(state='waiting_for_isolated_tree_audit',observed=stamp()))
        time.sleep(10)
    if not u.read(proof)['all_five_original_folds_bitwise']:raise ValueError('Original tree replay required')
    lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    entries=[r for r in u.read(DECLARATION)['families'] if not r['reuse_existing_full_base']]
    for entry in entries:
        name=entry['development'];high=0
        with (REPORT/(name+'.log')).open('x') as log:
            child=subprocess.Popen(['nice','-n','10',sys.executable,str(WORKER),name],
                cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    out=subprocess.check_output(['ps','-axo','pid=,rss=,command='],text=True)
                    own=total=0
                    for line in out.splitlines():
                        parts=line.strip().split(None,2)
                        if len(parts)!=3:continue
                        pid,rss,cmd=parts
                        if int(pid)==child.pid:own=int(rss)*1024
                        if str(u.ROOT) in cmd or 'target/release/fit ' in cmd:total+=int(rss)*1024
                    free=shutil.disk_usage(u.ROOT).free
                    high=high+1 if own>11*2**30 or total>24*2**30 else 0
                    u.write(REPORT/'status.json',dict(state='full_refitting_and_replaying',development=name,pid=child.pid,
                        own_rss_bytes=own,research_rss_bytes=total,disk_free_bytes=free,observed=stamp()))
                    if high>=3 or free<16*2**30:raise RuntimeError('Resource guard exceeded; original outputs preserved')
                    time.sleep(5)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM);child.wait();raise
            if child.returncode:raise RuntimeError('Full refit failed; partial evidence preserved')
        result=u.read(u.ROOT/'experiments/upstream-nn-pool6-full-refit'/name/'verification.json')
        if not result['complete_known_and_qualifying_bitwise_replay']:raise ValueError('Actual full query replay required')
    u.write(REPORT/'completion.json',dict(state='complete_five_full_refits_and_both_query_replays',observed=stamp(),
        qualifying_labels_read=False,assembly_and_independent_verification_remaining=True))

if __name__=='__main__':main()
