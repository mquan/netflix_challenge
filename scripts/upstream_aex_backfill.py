#!/usr/bin/env python3
"""Bring forward the original AEX128 source job in an isolated native tree."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_repro as u
import upstream_qualifying_backfill as preparation
import upstream_cfnade48 as resources
from upstream_full_reproduce import receipt_valid,group_rss,stop

BASE=u.ROOT/'artifacts/upstream-aex-backfill'
TREE=BASE/'fresh'
REPORT=u.ROOT/'experiments/upstream-aex-backfill'
JOB='aex-128'


def admission(other,free):
    return other+4*2**30<23*2**30 and free>28*2**30


def prepare():
    preparation.BASE,preparation.TREE,preparation.REPORT=BASE,TREE,REPORT
    preparation.ENDPOINTS=[JOB]
    preparation.prepare()
    u.validate_fresh()
    jobs=u.read(BASE/'registry.json')['jobs']
    if jobs[JOB]['inputs_from']!=['ingest','newsplit']:
        raise ValueError('AEX128 requires unexpected native model parents')
    public=(TREE/'src/bin/aex-new.rs').read_text()
    if 'n_epochs: 20' not in public or '"aex-128" => (128, 0.002, 0.1)' not in public:
        raise ValueError('Original public AEX128 preset changed')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()!=u.PIN:
        raise ValueError('AEX128 source pin changed')
    changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=TREE,text=True).splitlines())
    if changed-{'data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('AEX128 native source modified')
    plan=dict(source_commit=u.PIN,job=JOB,original_command=jobs[JOB]['command'],
        native_source_modified=False,isolated_tree=str(TREE),
        driver_sha256=u.sha(TREE/'src/bin/aex-new.rs'),
        code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(preparation.__file__),Path(resources.__file__),u.ROOT/'scripts/upstream_repro.py']},
        preparation_plan_sha256=u.sha(REPORT/'plan.json'),qualifying_labels_read=False,
        scope='Additional own complete original128-factor,20-epoch AEX job, both native phases. No smaller recipe or public predictions.')
    path=REPORT/'execution-plan.json'
    if path.exists() and u.read(path)!=plan:raise ValueError('Existing AEX128 execution plan changed')
    u.write(path,plan);return jobs


def verify_complete(jobs):
    if not receipt_valid(JOB,jobs):raise ValueError('Complete original paired AEX128 receipt required')
    source=BASE/'fresh-receipts'/(JOB+'.json');record=u.read(source);pairs={}
    for split,count in [('probex',1128164),('qual',2817131)]:
        name=f'preds_new/{JOB}.{split}.npy';p=TREE/name;a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.shape!=(count,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Invalid paired AEX128 prediction')
        pairs[split]=dict(rows=count,sha256=u.sha(p))
    if np.count_nonzero(np.load(TREE/'data/qual/ratings.npy',mmap_mode='r')):
        raise ValueError('AEX128 qualifying training targets are not zero')
    for name,digest in u.read(REPORT/'execution-plan.json')['code_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('AEX128 frozen execution code changed')
    shutil.copyfile(source,REPORT/(JOB+'.json'))
    u.write(REPORT/'verification.json',dict(state='verified_complete',native_receipt_sha256=u.sha(source),
        execution_plan_sha256=u.sha(REPORT/'execution-plan.json'),pairs=pairs,
        all_native_output_hashes_rechecked=True,qualifying_targets_all_zero=True,
        qualifying_labels_read=False,goal_complete=False))


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=prepare()
        if receipt_valid(JOB,jobs):
            verify_complete(jobs);u.write(REPORT/'status.json',dict(state='complete',job=JOB));return
        while True:
            other=resources.other_rss(set());fs=os.statvfs(u.ROOT);free=fs.f_bavail*fs.f_frsize
            if admission(other,free):break
            u.write(REPORT/'status.json',dict(state='waiting_for_memory',other_rss_bytes=other,disk_free_bytes=free));time.sleep(10)
        invocation="import upstream_repro as u;from pathlib import Path;u.BASE=Path(%r);u.FRESH=u.BASE/'fresh';u.model_job(%r,execute=True)"%(str(BASE),JOB)
        env=dict(os.environ,PYTHONPATH=str(u.ROOT/'scripts'),UPSTREAM_REPRO_THREADS='2',UPSTREAM_REPRO_BLAS_THREADS='1',CARGO_BUILD_JOBS='2')
        with (REPORT/'native-wrapper.log').open('a') as log:
            child=subprocess.Popen(['nice','-n','10',sys.executable,'-u','-c',invocation],cwd=u.ROOT,
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            high=0
            try:
                while child.poll() is None:
                    own,other=group_rss(child.pid),resources.other_rss({child.pid})
                    fs=os.statvfs(u.ROOT);free=fs.f_bavail*fs.f_frsize
                    high=high+1 if own+other>24*2**30 or own>5*2**30 else 0
                    u.write(REPORT/'status.json',dict(state='training',job=JOB,pid=child.pid,rss_bytes=own,other_rss_bytes=other,
                        disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                    if high>=3 or free<22*2**30:raise RuntimeError('AEX128 resource guard; partial outputs preserved')
                    time.sleep(5)
            except BaseException:stop(child);raise
            if child.returncode:raise RuntimeError('AEX128 native job failed; partial outputs preserved')
        verify_complete(jobs);u.write(REPORT/'status.json',dict(state='complete',job=JOB))


if __name__=='__main__':
    try:run()
    except BaseException as error:
        if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
        raise
