#!/usr/bin/env python3
"""Recover a terminal disk-guard stop without changing any frozen fit recipe."""
import argparse
import datetime as dt
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_item154_mps_context as model

u=model.u
REPORT=u.ROOT/'experiments/upstream-item154-recovery'
ARCHIVE=u.ROOT/'artifacts/upstream-item154-recovery/partial-fold4'

def monitored(extra,logname):
    with (REPORT/logname).open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),*extra],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=model.base.preceding.group_rss(child.pid);other=model.base.preceding.resources.other_rss({child.pid})
                free=shutil.disk_usage(u.ROOT).free;high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<8*2**30:raise RuntimeError('Recovery resource guard; partial outputs preserved')
                time.sleep(5)
        except BaseException:model.base.preceding.stop(child);raise
        if child.returncode:raise RuntimeError('Recovery child failed; all outputs preserved')

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing recovery evidence preserved')
        ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
        for line in ps.splitlines():
            pid,command=line.strip().split(None,1)
            if any(Path(a).name=='upstream_item154_mps_context.py' for a in command.split()[1:]):
                raise ValueError('Original model process still live; do not recover')
        status=u.read(model.REPORT/'status.json')
        if status['state']!='incomplete_preserved' or 'resource guard' not in status.get('error',''):
            raise ValueError('Actual terminal resource-guard evidence required')
        model.configure();model.n.verify_prepared();completed={}
        for fold in range(4):
            path=model.REPORT/'nn'/f'{fold}.json';r=u.read(path)
            if r['manifest_sha256']!=u.sha(model.REPORT/'manifest.json'):raise ValueError('Completed fold recipe differs')
            for name,d in r['outputs'].items():
                if u.sha(model.RUN/'nn'/str(fold)/name)!=d:raise ValueError('Completed artifact changed')
            completed[str(path.relative_to(u.ROOT))]=u.sha(path)
        if (model.REPORT/'nn/4.json').exists() or not (model.RUN/'nn/4').is_dir() or ARCHIVE.exists():raise ValueError('Expected sole incomplete fold4 differs')
        partial={p.name:u.sha(p) for p in (model.RUN/'nn/4').iterdir()}
        u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),script_sha256=u.sha(Path(__file__)),
            original_manifest_sha256=u.sha(model.REPORT/'manifest.json'),completed_receipt_sha256=completed,partial_sha256=partial,
            reason='Actual original worker terminated when paired-bank preparation reduced disk below generic22GiB guard. Four complete folds preserved. Move incomplete fold4 to a separate archive and repeat only that fold with the exact frozen source/config/seeds/folds/stopping.',
            model_recipe_changed=False,resource_policy=dict(memory_total_bytes=24*2**30,consecutive_high_samples=3,disk_floor_bytes=8*2**30,
                reason='Inputs already fully allocated; remaining fold/model/replay outputs total far below1GiB. Preserve the original memory guard and allow8GiB disk reserve.'),qualifying_labels_read=False))
        ARCHIVE.parent.mkdir(parents=True,exist_ok=True);(model.RUN/'nn/4').rename(ARCHIVE)
        u.write(REPORT/'status.json',dict(state='waiting_for_shared_fit_slot'))
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            monitored(['--worker'],'nn.log');monitored(['--replay'],'replay.log')
            model.configure();result=model.base.preceding.summarize()
            for name,d in completed.items():
                if u.sha(u.ROOT/name)!=d:raise ValueError('Completed fold receipt changed during recovery')
            u.write(REPORT/'completion.json',dict(state='complete_same_recipe_recovery',original_manifest_sha256=u.sha(model.REPORT/'manifest.json'),
                recovery_plan_sha256=u.sha(REPORT/'plan.json'),all_five_actual_saved_model_replays=u.sha(model.REPORT/'nn-replay.json'),
                first_four_receipts_unchanged=True,partial_fold_preserved=True,qualifying_labels_read=False))
            with (REPORT/'mean.log').open('x') as log:subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/upstream_rbm50_tree_item154_mps.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            u.write(REPORT/'status.json',dict(state='complete_recovered_development_and_fixed_mean',records=result['records']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();model.configure()
    if a.worker:model.n.worker('nn')
    elif a.replay:model.base.verifier.replay('nn')
    else:run()
