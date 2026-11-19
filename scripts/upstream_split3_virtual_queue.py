#!/usr/bin/env python3
"""Fixed own user64 extension, original NN recipe, actual replays and mean."""
import datetime as dt
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_split3_virtual_mps_context as model

u=model.u
REPORT=u.ROOT/'experiments/upstream-split3-virtual-queue'

def execute(script,logname):
    with (REPORT/logname).open('x') as log:
        subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def monitored(extra,logname):
    with (REPORT/logname).open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(model.__file__)),*extra],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=model.base.preceding.group_rss(child.pid);other=model.base.preceding.resources.other_rss({child.pid})
                free=shutil.disk_usage(u.ROOT).free;high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<8*2**30:raise RuntimeError('Split3 virtual resource guard; partial outputs preserved')
                time.sleep(5)
        except BaseException:model.base.preceding.stop(child);raise
        if child.returncode:raise RuntimeError('Split3 virtual child failed; outputs preserved')

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing split3_virtual declaration preserved')
        paths=[Path(__file__),Path(model.__file__),Path(model.data.__file__),u.ROOT/'scripts/upstream_rbm50_tree_split3_virtual_mps.py',u.ROOT/'scripts/verify_upstream_development_mean.py']
        hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
        u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),code_sha256=hashes,
            fixed_addition=model.data.NEW,virtual_tensor_inputs=True,
            raw_prediction_columns=111,learned_support_columns=154,inputs=402,hidden=128,context_rank=32,
            origins=['upstream-rbm50-tree','upstream-split3-virtual-mps-context'],weights=[.5,.5],
            reference_development_rmse=.8592309819864167,qualifying_labels_read=False,
            resource_policy=dict(memory_total_bytes=24*2**30,consecutive_high_samples=3,disk_floor_bytes=8*2**30),
            rule='Retain original five nested user folds, seeds, optimizer and stopping. Require all original native phases and full known projections, independent ridge replay and exact virtual tensor fingerprints before fitting. All15 actual NN model replays and separate stored fixed-mean score required. No automatic production or qualifying selection.'))
        try:
            while not (model.data.screen.REPORT/'ridge-verification.json').exists():
                u.write(REPORT/'status.json',dict(state='waiting_for_complete_original_split3_and_independent_ridge_proof',observed=dt.datetime.now(dt.timezone.utc).isoformat()));time.sleep(20)
            with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
                fcntl.flock(shared,fcntl.LOCK_EX)
                previous=model.base.memory.memory();good=0
                while good<2:
                    current=model.base.memory.memory();workers=model.base.memory.footprints()
                    good=good+1 if model.context_admitted(previous,current,sum(r['bytes'] for r in workers.values())) and shutil.disk_usage(u.ROOT).free>12*2**30 else 0
                    u.write(REPORT/'status.json',dict(state='waiting_for_split3_virtual_preparation_memory',good_samples=good,memory=current))
                    if good<2:previous=current;time.sleep(5)
                execute('upstream_split3_virtual_features.py','features.log');model.configure();model.prepare()
                monitored(['--worker'],'nn.log');monitored(['--replay'],'replay.log')
                model.configure();model.base.preceding.summarize()
                for name,digest in hashes.items():
                    if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared fixed study code changed')
                execute('upstream_rbm50_tree_split3_virtual_mps.py','mean.log')
                with (REPORT/'independent-mean.log').open('x') as log:
                    subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_development_mean.py'),'upstream-rbm50-tree-split3-virtual-mps'],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=u.read(u.ROOT/'experiments/upstream-rbm50-tree-split3-virtual-mps/verification.json')
                u.write(REPORT/'status.json',dict(state='complete_split3_virtual_actual_replayed_development_and_fixed_mean',rmse=result['rmse'],improves_previous=result['improves_previous'],goal_complete=False))
        except BaseException as error:
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise

if __name__=='__main__':run()
