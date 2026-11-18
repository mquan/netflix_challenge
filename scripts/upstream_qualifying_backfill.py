#!/usr/bin/env python3
"""Fill missing qualifying predictors in an isolated, unmodified native tree."""
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
from upstream_full_reproduce import group_rss,stop

ORIGINAL_BASE,ORIGINAL_TREE=u.BASE,u.FRESH
BASE=u.ROOT/'artifacts/upstream-qualifying-backfill'
TREE=BASE/'fresh'
REPORT=u.ROOT/'experiments/full-upstream-reproduction/qualifying-backfill'
ENDPOINTS=['rbmx2-40__knn3','tsvdx4-60__knns','dnn-24__knn3',
           'dnn-24__dnn-16p50__knn3']


def prepare():
    u.validate_fresh();BASE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if not TREE.exists():
        subprocess.run(['git','clone','--shared',str(u.SOURCE),str(TREE)],check=True)
        subprocess.run(['git','checkout','--detach',u.PIN],cwd=TREE,check=True)
    for name in ['registry.json','fresh-training-plan.json','fresh-data-receipt.json']:
        source=ORIGINAL_BASE/name;target=BASE/name
        if target.exists() and u.sha(target)!=u.sha(source):raise ValueError('Backfill setup record differs')
        shutil.copyfile(source,target)
    toolchain=BASE/'toolchain'
    if not toolchain.exists():toolchain.symlink_to(ORIGINAL_BASE/'toolchain',target_is_directory=True)
    if toolchain.resolve()!=(ORIGINAL_BASE/'toolchain').resolve():raise ValueError('Wrong toolchain')
    sentinel='data/qual_ratings/qual_ratings.csv.gz'
    shutil.copyfile(ORIGINAL_TREE/sentinel,TREE/sentinel)
    for name in ['raw','train','probe','trainx','probex','fulltrain','qual']:
        path=TREE/'data'/name;source=ORIGINAL_TREE/'data'/name
        if not path.is_symlink():
            if path.exists():raise ValueError('Unexpected backfill data path')
            path.symlink_to(source,target_is_directory=True)
        if path.resolve()!=source.resolve():raise ValueError('Wrong backfill data path')
    target=TREE/'target'
    if not target.is_symlink():
        if target.exists():raise ValueError('Unexpected backfill target path')
        target.symlink_to(BASE/'target',target_is_directory=True)
    if target.resolve()!=(BASE/'target').resolve():raise ValueError('Wrong backfill target')
    # Seed only completed, verified native parents. Hardlinks remain immutable.
    receipts=BASE/'fresh-receipts';receipts.mkdir(exist_ok=True)
    for source in sorted((ORIGINAL_BASE/'fresh-receipts').glob('*.json')):
        if (receipts/source.name).exists():continue
        receipt=u.read(source)
        if receipt['source_commit']!=u.PIN:raise ValueError('Wrong parent pin')
        for name,digest in receipt['outputs'].items():
            original=ORIGINAL_TREE/name;copy=TREE/name
            if u.sha(original)!=digest:raise ValueError('Changed native parent '+name)
            if copy.exists():
                if u.sha(copy)!=digest:raise ValueError('Conflicting backfill parent '+name)
            else:
                copy.parent.mkdir(parents=True,exist_ok=True);u.link_or_copy(original,copy)
        shutil.copyfile(source,receipts/source.name)
    # Reuse only the unmodified pipeline runner; model binaries build into this tree's own target.
    run=BASE/'target/release/run';run.parent.mkdir(parents=True,exist_ok=True)
    if not run.exists():u.link_or_copy(ORIGINAL_BASE/'target/release/run',run)
    u.BASE,u.FRESH=BASE,TREE
    u.validate_fresh()
    u.write(REPORT/'plan.json',dict(source_pin=u.PIN,endpoints=ENDPOINTS,source_model_code_modified=False,
        separation='Independent prediction/receipt/build directories; verified completed parents hardlinked. Never import outputs into an active original wrapper scan.',
        source_qualifying_ratings_zero=True,runner_sha256=u.sha(Path(__file__))))


def other_training_rss():
    out=subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,rss=,command='],text=True)
    rows=[line.split(None,4) for line in out.splitlines()]
    groups={int(r[2]) for r in rows if len(r)==5 and ('scripts/upstream_repro.py --run-job' in r[4] or
        '/target/release/tsvdx5-new' in r[4] or 'upstream_voting_blend.py --worker' in r[4])}
    groups.discard(os.getpgrp())
    return sum(int(r[3])*1024 for r in rows if len(r)==5 and int(r[2]) in groups)


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare();child=None
        state=dict(state='running',endpoints=ENDPOINTS,completed=[])
        try:
            for name in ENDPOINTS:
                while other_training_rss()>21*2**30:
                    state.update(state='waiting_for_memory',current=name);u.write(REPORT/'status.json',state);time.sleep(30)
                state.update(state='training',current=name);u.write(REPORT/'status.json',state)
                # A separate process binds the same frozen upstream module to this isolated base.
                invocation="import upstream_repro as u;from pathlib import Path;u.BASE=Path(%r);u.FRESH=u.BASE/'fresh';u.model_job(%r,execute=True)"%(str(BASE),name)
                env=dict(os.environ,PYTHONPATH=str(u.ROOT/'scripts'),UPSTREAM_REPRO_THREADS='2',UPSTREAM_REPRO_BLAS_THREADS='1')
                with (REPORT/(name+'.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','-c',invocation],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    high=0
                    while child.poll() is None:
                        own=group_rss(child.pid);other=other_training_rss();free=os.statvfs(BASE);free=free.f_bavail*free.f_frsize
                        high=high+1 if own+other>28*2**30 or own>10*2**30 else 0
                        state.update(pid=child.pid,rss_bytes=own,other_training_rss_bytes=other,disk_free_bytes=free,
                            heartbeat=dt.datetime.now(dt.timezone.utc).isoformat())
                        u.write(REPORT/'status.json',state)
                        if high>=3 or free<16*2**30:raise RuntimeError('Backfill resource guard; preserve partial work')
                        time.sleep(30)
                if child.returncode:raise RuntimeError('Backfill native worker failed')
                state['completed'].append(name)
                receipt=BASE/'fresh-receipts'/(name+'.json')
                shutil.copyfile(receipt,REPORT/receipt.name)
            state.update(state='complete',current=None);u.write(REPORT/'status.json',state)
        except BaseException as error:
            if child is not None:stop(child)
            state.update(state='incomplete_preserved',error=repr(error));u.write(REPORT/'status.json',state);raise


if __name__=='__main__':run()
