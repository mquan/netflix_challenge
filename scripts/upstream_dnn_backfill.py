#!/usr/bin/env python3
"""Prioritize the six missing production DNN chains after the active backfill."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_qualifying_backfill as prior
from upstream_full_reproduce import group_rss,stop

REPORT=u.ROOT/'experiments/full-upstream-reproduction/dnn-backfill'
OPERATIONAL=prior.REPORT/'status.json'
ENDPOINTS=['dnn-24__asym-16p75__knn3','dnn-24__dnn-16__knn3',
 'dnn-24__dnn-16p50__asym-16p50','dnn-24__dnn-16p50__asym-16p75__knn3b',
 'dnn-24__dnn-16p50__asym-16p75__knn3c','dnn-24__dnn-24p50']


def extra_rss():
    groups=set()
    path=u.ROOT/'experiments/upstream-cfnade-parallel/status.json'
    if path.exists():groups.update(u.read(path).get('active',{}).values())
    path=u.ROOT/'experiments/upstream-production/status.json'
    if path.exists():
        state=u.read(path)
        if state.get('state') in ['full_probe_refitting','replaying'] and state.get('pid'):groups.add(state['pid'])
    return sum(group_rss(pid) for pid in groups)


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    lock_path=prior.REPORT/'run.lock'
    with lock_path.open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                u.write(REPORT/'status.json',dict(state='waiting_for_active_backfill',endpoints=ENDPOINTS));time.sleep(30)
        old=u.read(OPERATIONAL)
        # An orphaned old wrapper must finish before another directory scan starts.
        while old.get('pid') and group_rss(old['pid'])>0:
            u.write(REPORT/'status.json',dict(state='waiting_for_old_owned_process_group'));time.sleep(30)
        prior.REPORT=REPORT;prior.ENDPOINTS=ENDPOINTS;prior.prepare()
        u.write(REPORT/'continuation.json',dict(endpoints=ENDPOINTS,script_sha256=u.sha(Path(__file__)),
            previous_operational_state=old,scope='Same isolated native tree, exclusive original backfill lock. Six fixed published endpoints needed by the 85-column bank. Completed parents alone are reused.'))
        state=dict(state='running',completed=[],scheduler='dnn-followon',endpoints=ENDPOINTS);child=None
        try:
            for name in ENDPOINTS:
                while prior.other_training_rss()+extra_rss()>21*2**30:
                    state.update(state='waiting_for_memory',current=name);u.write(REPORT/'status.json',state);time.sleep(30)
                command="import upstream_repro as u;from pathlib import Path;u.BASE=Path(%r);u.FRESH=u.BASE/'fresh';u.model_job(%r,execute=True)"%(str(prior.BASE),name)
                env=dict(os.environ,PYTHONPATH=str(u.ROOT/'scripts'),UPSTREAM_REPRO_THREADS='2',UPSTREAM_REPRO_BLAS_THREADS='1')
                with (REPORT/(name+'.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','-c',command],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    high=0
                    while child.poll() is None:
                        own=group_rss(child.pid);other=prior.other_training_rss()+extra_rss()
                        disk=os.statvfs(prior.BASE);free=disk.f_bavail*disk.f_frsize
                        # The checkpointed canonical scheduler yields above 27 GiB;
                        # give it priority to pause before this noncheckpointed DNN.
                        high=high+1 if own>12*2**30 or own+other>30*2**30 else 0
                        state.update(state='training',current=name,pid=child.pid,rss_bytes=own,other_rss_bytes=other,
                            disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat())
                        u.write(REPORT/'status.json',state);u.write(OPERATIONAL,state)
                        if high>=3 or free<16*2**30:raise RuntimeError('DNN backfill resource guard; partial outputs preserved')
                        time.sleep(30)
                if child.returncode:raise RuntimeError('DNN native continuation failed')
                state['completed'].append(name)
                receipt=prior.BASE/'fresh-receipts'/(name+'.json')
                import shutil
                shutil.copyfile(receipt,REPORT/receipt.name)
            state.update(state='complete',current=None);u.write(REPORT/'status.json',state);u.write(OPERATIONAL,state)
        except BaseException as error:
            if child is not None:stop(child)
            state.update(state='incomplete_preserved',error=repr(error));u.write(REPORT/'status.json',state);u.write(OPERATIONAL,state);raise


if __name__=='__main__':run()
