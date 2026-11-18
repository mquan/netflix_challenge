#!/usr/bin/env python3
"""Resume the published native DAG locally, with receipts and resource guards."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_repro as u
from upstream_missing_screen import producer_for

REPORT=u.ROOT/'experiments/full-upstream-reproduction'
SETUP={'download','ingest','newsplit'}
PRIORITY=['vfeat1','mf-61__knn3','mf-61__asym-16','mf-60__knn3','als8-8__knn3',
 'bknbrx-100','bknbrx-50','cfnade-12','attn-32','attn-32-t','aex-16','aex-64','aex-128',
 'dnn-24__dnn-16p50__asym-16p75__knn3b','dnn-24__dnn-16p50__asym-16p75__knn3c',
 'dnn-24__asym-16p75__knn3','dnn-24__dnn-16p50__asym-16p50','dnn-24__dnn-24p50',
 'dnn-24__dnn-16__knn3','rbmx2-0','rbmx2-8e','tsvdx4-d0tb','tsvdx4-f1',
 'tsvdx4-q16tb','tsvdx4-l32nb','tsvdx4-60o']
# Requires a storage-compatible implementation on this 32-GiB host; not smaller recipes.
LARGE={'tsvdx5-120','tsvdx5-150','tsvdx5-120o','tsvdx5-150o','tsvdx5-300',
 'tsvdx5-400','tsvdx5-1000','tsvdx6-300','tsvdx6-300-v2','tsvdx4-160o',
 'tsvdx4-4800lm'}


def event(**row):print(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),**row),flush=True)


def receipt_valid(name, jobs):
    path=u.BASE/'fresh-receipts'/(name+'.json')
    if not path.exists():return False
    record=u.read(path)
    if record['source_commit']!=u.PIN or record['job']!=name:raise ValueError('Wrong native receipt '+name)
    if record['command']!=jobs[name]['command']:raise ValueError('Native recipe differs '+name)
    if not set(jobs[name]['outputs']).issubset(record['outputs']):raise ValueError('Incomplete native receipt '+name)
    for filename,digest in record['outputs'].items():
        if not (u.FRESH/filename).is_file() or u.sha(u.FRESH/filename)!=digest:
            raise ValueError('Changed native output '+name+': '+filename)
    return True


def plan():
    registry=u.read(u.BASE/'registry.json');jobs=registry['jobs'];columns=[];required=[]
    for name in registry['names']:
        producer,reason=producer_for(name,jobs)
        order=[] if producer is None else [n for n in u.dependency_order(jobs,producer) if n not in SETUP]
        required.extend(order)
        columns.append(dict(name=name,producer=producer,producer_resolution=reason,dependencies=order,
                            legacy_epoch_coefficients='__epochs' in name))
    # Include all declared source jobs, so side products and later blend layers are regenerated.
    ordered=[]
    for endpoint in PRIORITY+list(dict.fromkeys(required))+sorted(jobs):
        if endpoint not in jobs:continue
        for job in u.dependency_order(jobs,endpoint):
            if job not in SETUP and job not in ordered:ordered.append(job)
    complete=[name for name in ordered if receipt_valid(name,jobs)]
    result=dict(source_commit=u.PIN,registry_sha256=u.sha(u.BASE/'registry.json'),
        jobs_total=len(jobs),final_columns=len(columns),columns=columns,order=ordered,
        complete_native_jobs=complete,unresolved_columns=[r['name'] for r in columns if r['producer'] is None],
        memory_backend_required=sorted(LARGE),
        policy='Fresh native outputs only. Pinned source hyperparameters. ProbeX may fit source-defined calibration/blend layers. Qualifying outcomes remain zero throughout fitting; audit only after freezing final predictions.',
        legacy_policy='Published source epoch weights are reference recipes; retain separately and refit intermediate epoch weights on fresh development outputs before strict locally-fitted production attribution.')
    u.write(REPORT/'inventory.json',result)
    return result,jobs


def group_rss(pid):
    out=subprocess.check_output(['ps','-axo','pgid=,rss='],text=True)
    return sum(int(p[1]) for line in out.splitlines() if len(p:=line.split())==2 and int(p[0])==pid)*1024


def stop(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def render(state):
    lines=['# Full published-pipeline reproduction: active','',
      'Goal: verified recovered qualifying RMSE below 0.85725 using freshly trained outputs. No other laptop is assumed available.','',
      f"Current state: **{state['state']}**; current native job: `{state.get('current')}`.",
      f"Completed native jobs on this machine: {len(state['completed'])}. Registry: 201 jobs / 188 final input columns.",'',
      'The full exact source DAG is the work list. Completed remote ProbeX exports do not stand in for missing native fulltrain/qual or parent-training artifacts. Resource-limited jobs remain in the inventory for the memory-compatible backend; no smaller factor count is substituted.','',
      'State: `experiments/full-upstream-reproduction/status.json`. Inventory and unresolved historical aliases: `inventory.json`. Logs/receipts: `artifacts/upstream-repro/fresh-receipts/`.','',
      'The best existing fresh-only development checkpoint is 0.8621943362. It does not establish the qualifying target. The goal remains active until production predictions are frozen and both recovered groups independently pass.','']
    (u.ROOT/'FULL_REPRODUCTION_PROGRESS.md').write_text('\n'.join(lines))


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inv,jobs=plan();completed=set(inv['complete_native_jobs']);failed={}
        state=dict(state='running',current=None,completed=sorted(completed),failed=failed,
                   source_commit=u.PIN,runner_sha256=u.sha(Path(__file__)))
        initial=REPORT/'initial-mf61-launch.json'
        if initial.exists():
            pid=u.read(initial)['pid']
            while True:
                check=subprocess.run(['ps','-o','command=','-p',str(pid)],capture_output=True,text=True)
                if check.returncode or 'upstream_repro.py --run-job mf-61__knn3' not in check.stdout:break
                state.update(state='waiting_for_owned_initial_chain',current='mf-61__knn3')
                u.write(REPORT/'status.json',state);render(state);time.sleep(30)
            for name in ['mf-61','mf-61__knn3']:
                if receipt_valid(name,jobs):completed.add(name)
        child=None
        try:
            for name in inv['order']:
                if name in completed:continue
                if receipt_valid(name,jobs):completed.add(name);continue
                deps=[n for n in u.dependency_order(jobs,name) if n not in SETUP and n!=name]
                if name in LARGE or any(n in LARGE for n in deps):continue
                if any(n not in completed for n in deps):continue
                missing=[f for f in jobs[name]['inputs'] if not (u.FRESH/f).exists()]
                if missing:
                    failed[name]=dict(state='missing_inputs',paths=missing);continue
                if any((u.FRESH/f).exists() for f in jobs[name]['outputs']):
                    failed[name]=dict(state='unreceipted_partial_outputs_preserved');continue
                disk=os.statvfs(u.ROOT);free=disk.f_bavail*disk.f_frsize
                if free<20*2**30:
                    state.update(state='waiting_for_disk_space',current=name,completed=sorted(completed))
                    u.write(REPORT/'status.json',state);render(state);break
                state.update(state='training',current=name,completed=sorted(completed))
                u.write(REPORT/'status.json',state);render(state);event(start=name)
                env=dict(os.environ,UPSTREAM_REPRO_THREADS='4',UPSTREAM_REPRO_BLAS_THREADS='1')
                with (REPORT/(name+'.wrapper.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','scripts/upstream_repro.py','--run-job',name],
                        cwd=u.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    high=0;why=None
                    while child.poll() is None:
                        rss=group_rss(child.pid);disk=os.statvfs(u.ROOT)
                        high=high+1 if rss>26*2**30 else 0
                        sample=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),rss_bytes=rss,
                            disk_free_bytes=disk.f_bavail*disk.f_frsize,pid=child.pid)
                        u.write(REPORT/'resource-current.json',sample)
                        if high>=3 or sample['disk_free_bytes']<12*2**30:
                            why='memory_or_disk_guard';stop(child);break
                        state.update(heartbeat_utc=sample['utc'],child_pid=child.pid)
                        u.write(REPORT/'status.json',state);time.sleep(30)
                if child.returncode==0 and receipt_valid(name,jobs):
                    completed.add(name);event(done=name)
                else:
                    failed[name]=dict(state=why or 'native_failure',exit_code=child.returncode)
                    event(failed=name,reason=failed[name])
                child=None
            if state['state']!='waiting_for_disk_space':
                state.update(state='pass_finished_remaining_jobs_require_recovery_or_memory_backend',current=None)
            state.update(completed=sorted(completed),failed=failed)
            u.write(REPORT/'status.json',state);render(state)
        except BaseException as error:
            if child is not None:stop(child)
            state.update(state='interrupted',error=repr(error),completed=sorted(completed))
            u.write(REPORT/'status.json',state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',action='store_true');a=ap.parse_args()
    if a.plan:
        p,_=plan();print({key:p[key] for key in ['jobs_total','final_columns','complete_native_jobs','unresolved_columns']})
    else:run()
