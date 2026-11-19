#!/usr/bin/env python3
"""Adopt the unchanged live CPU tree worker without holding the GPU epoch lock."""
import datetime as dt
import fcntl
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import upstream_item154_tree as model

u=model.u
REPORT=u.ROOT/'experiments/upstream-item154-tree-supervisor'
OLD_CONTROLLER=60506
WORKER=60801

def identity(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat=,lstart=,command='],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None

def alive(pid,original):
    current=identity(pid)
    if current is None:return False
    if current.split()[0].startswith('Z'):return False
    if current.split(maxsplit=1)[1]!=original.split(maxsplit=1)[1]:raise ValueError('PID identity changed')
    return True

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing adoption declaration preserved')
        controller,worker=identity(OLD_CONTROLLER),identity(WORKER)
        if not controller or not worker or 'upstream_item154_tree.py' not in controller or '--worker' not in worker or 'upstream_item154_tree.py' not in worker:
            raise ValueError('Exact live CPU controller and worker required')
        if int(subprocess.check_output(['ps','-p',str(WORKER),'-o','ppid='],text=True))!=OLD_CONTROLLER:
            raise ValueError('Worker parent differs')
        # Python-created lock descriptors are non-inheritable. Verify the actual
        # worker has no shared epoch descriptor before replacing supervision.
        descriptors=subprocess.run(['lsof','-a','-p',str(WORKER),'-Fn'],capture_output=True,text=True,check=True).stdout
        shared=u.ROOT/'experiments/upstream-source-extension/run.lock'
        if str(shared) in descriptors or str(model.REPORT/'run.lock') in descriptors:
            raise ValueError('Worker inherited a controller lock; adoption cannot release it')
        model.configure();manifest=model.n.verify_prepared();manifest_path=model.REPORT/'manifest.json'
        bound={str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),str(manifest_path.relative_to(u.ROOT)):u.sha(manifest_path),
               str((model.REPORT/'plan.json').relative_to(u.ROOT)):u.sha(model.REPORT/'plan.json')}
        u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),controller_pid=OLD_CONTROLLER,controller_identity=controller,
            unchanged_worker_pid=WORKER,worker_identity=worker,source_sha256=bound,
            rule='Replace only idle Python parent supervision. Keep the exact running CPU worker and all original numerical inputs/configuration/models/folds/budgets. The replacement holds the model run lock, not the GPU epoch lock. Monitor the original24GiB total RSS/8GiB disk guards; require all five actual receipts, actual saved-tree replay and independent scoring before success.',
            public_prediction_values_read=False,qualifying_labels_read=False,worker_signal_sent=False))
        # SIGKILL is confined to the supervising parent so its exception cleanup
        # cannot terminate the independent-session native fitting child.
        if not alive(OLD_CONTROLLER,controller) or not alive(WORKER,worker):raise ValueError('Verified process changed before adoption')
        os.kill(OLD_CONTROLLER,signal.SIGKILL)
        with (model.REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not alive(WORKER,worker):raise ValueError('Actual tree worker unexpectedly stopped during adoption')
            u.write(REPORT/'handoff.json',dict(state='same_actual_cpu_worker_adopted',observed=dt.datetime.now(dt.timezone.utc).isoformat(),worker_pid=WORKER,
                worker_identity=identity(WORKER),original_manifest_sha256=u.sha(manifest_path),worker_signals_sent=False,original_fitting_recipe_unchanged=True))
            high=0
            while alive(WORKER,worker):
                actual=model.template.group_rss(WORKER);other=model.template.parent.resources.other_rss({WORKER});free=__import__('shutil').disk_usage(u.ROOT).free
                high=high+1 if actual+other>24*2**30 else 0
                u.write(REPORT/'status.json',dict(state='monitoring_same_original_cpu_tree_worker',pid=WORKER,rss_bytes=actual,other_rss_bytes=other,
                    disk_free_bytes=free,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<8*2**30:
                    os.kill(WORKER,signal.SIGTERM)
                    raise RuntimeError('Original tree resource guard reached; partial outputs preserved')
                time.sleep(5)
            for name,digest in bound.items():
                if u.sha(u.ROOT/name)!=digest:raise ValueError('Adopted worker source declaration changed')
            for fold in range(5):
                path=model.REPORT/'tree'/f'{fold}.json';r=u.read(path)
                if r['manifest_sha256']!=u.sha(manifest_path) or r['fold']!=fold or r['family']!='tree':raise ValueError('Actual completed fold receipt missing')
                for name,digest in r['outputs'].items():
                    if u.sha(model.RUN/'tree'/str(fold)/name)!=digest:raise ValueError('Actual saved tree artifact differs')
            model.guarded([sys.executable,str(Path(model.__file__)),'--replay'],REPORT/'saved-tree-replay.log')
            model.configure();result=model.template.summarize()
            u.write(REPORT/'completion.json',dict(state='complete_original_tree_fit_actual_replay_and_independent_scoring',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
                evaluation_sha256=u.sha(model.REPORT/'evaluation.json'),verification_sha256=u.sha(model.REPORT/'independent-verification.json'),
                replay_sha256=u.sha(model.REPORT/'tree-replay.json'),records=result['records'],qualifying_labels_read=False,goal_complete=False))
            u.write(model.REPORT/'status.json',dict(state='development_complete',records=result['records'],supervision='upstream-item154-tree-supervisor',goal_complete=False))

if __name__=='__main__':
    try:run()
    except BaseException as error:
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',observed=dt.datetime.now(dt.timezone.utc).isoformat(),error=repr(error)));raise
