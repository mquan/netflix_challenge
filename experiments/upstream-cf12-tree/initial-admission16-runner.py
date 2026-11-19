#!/usr/bin/env python3
"""Train the corresponding86-predictor tree and reuse the verified86-context NN."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_cf12_context as parent
from upstream_full_reproduce import group_rss,stop

u,n,x=parent.u,parent.n,parent.x
RUN=u.ROOT/'artifacts/upstream-cf12-tree'
REPORT=u.ROOT/'experiments/upstream-cf12-tree'
CONFIG=dict(parent.CONFIG)


def code_hashes():
    return {**parent.code_hashes(),str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))}


def configure():
    parent.configure()
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,CONFIG,code_hashes


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial matching-tree preparation preserved')
    parent.configure();pm=n.verify_prepared()
    evaluation=parent.REPORT/'evaluation.json';proof=u.read(parent.REPORT/'independent-verification.json')
    replay=u.read(parent.REPORT/'nn-replay.json')
    if proof['evaluation_sha256']!=u.sha(evaluation) or not proof['fixed_average_bitwise_equal']:
        raise ValueError('Parent complete score is not independently verified')
    if len(replay['records'])!=5 or not all(r['bitwise_equal'] for r in replay['records']):
        raise ValueError('Reused neural model replay incomplete')
    rows=u.read(evaluation)['records'];nn=next(r for r in rows if r['arm']=='nn')
    previous=next(r['rmse'] for r in rows if r['arm']=='fixed-half-tree-nn')
    if u.sha(parent.RUN/'nn.f32')!=nn['prediction_sha256']:raise ValueError('Reused neural predictions changed')
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [parent.REPORT/'manifest.json',evaluation,
        parent.REPORT/'independent-verification.json',parent.REPORT/'nn-replay.json']}
    for fold in range(5):
        p=parent.REPORT/'nn'/(str(fold)+'.json');receipt=u.read(p)
        if receipt['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json'):raise ValueError('Reused NN receipt changed')
        for name,digest in receipt['outputs'].items():
            if u.sha(parent.RUN/'nn'/str(fold)/name)!=digest:raise ValueError('Reused NN checkpoint changed')
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    configure();REPORT.mkdir(parents=True,exist_ok=True);RUN.mkdir()
    # Read-only feature/data caches are shared; the old tree vector is excluded.
    for name in pm['prepared_sha256']:
        if name!='tree.f32':u.link_or_copy(parent.RUN/name,RUN/name)
    u.link_or_copy(parent.RUN/'nn.f32',RUN/'nn.f32')
    z=np.load(RUN/'design.npy',mmap_mode='r');features=np.load(RUN/'features.npy',mmap_mode='r')
    if z.shape!=(pm['rows'],162) or features.shape!=(pm['rows'],222):raise ValueError('Matching-tree86 schema differs')
    manifest=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        rows=pm['rows'],prediction_columns=86,design_columns=162,nonlinear_features=222,
        baseline_rmse=pm['baseline_rmse'],previous_best_rmse=previous,reused_nn_prediction_sha256=nn['prediction_sha256'],
        arms=['control','tree','nn','fixed-half-tree-nn'],qualifying_labels_read=False,isolation=pm['isolation'],
        scope='Adaptive known-ProbeX matching-tree experiment. Original tree hyperparameters/user folds, now the same86-predictor ridge/features as the verified context NN. Reuses the actual neural OOF vector without a new neural fit or synthetic receipts. No qualifying outcomes or production changes.')
    u.write(REPORT/'manifest.json',manifest);return manifest


def summarize():
    manifest=n.verify_prepared();y,groups,baseline=[np.load(RUN/(k+'.npy')) for k in ['labels','groups','baseline']]
    folds=groups//2;tree=np.empty(len(y),dtype=np.float64)
    for fold in range(5):
        root=RUN/'tree'/str(fold);receipt=u.read(REPORT/'tree'/(str(fold)+'.json'))
        if receipt['manifest_sha256']!=u.sha(REPORT/'manifest.json'):raise ValueError('Matching tree receipt changed')
        for name,digest in receipt['outputs'].items():
            if u.sha(root/name)!=digest:raise ValueError('Matching tree model changed')
        values=np.fromfile(root/'prediction.f32',dtype='<f4');held=folds==fold
        if values.shape!=(int(held.sum()),) or not np.isfinite(values).all():raise ValueError('Invalid tree fold predictions')
        tree[held]=values
    nn=np.fromfile(RUN/'nn.f32',dtype='<f4').astype(np.float64)
    if nn.shape!=y.shape or u.sha(RUN/'nn.f32')!=manifest['reused_nn_prediction_sha256']:
        raise ValueError('Reused neural vector changed')
    predictions=dict(control=baseline,tree=tree,nn=nn)
    predictions['fixed-half-tree-nn']=(tree+nn)/2;records=[]
    for arm,values in predictions.items():
        path=RUN/(arm+'.f32')
        if arm!='nn':values.astype('<f4').tofile(path)
        stored=np.fromfile(path,dtype='<f4');score=x.metric(stored,y)
        records.append(dict(arm=arm,rmse=score,delta_vs_previous_best=score-manifest['previous_best_rmse'],
            fold_rmse=[x.metric(stored[folds==fold],y[folds==fold]) for fold in range(5)],prediction_sha256=u.sha(path)))
    result=dict(state='complete',manifest_sha256=u.sha(REPORT/'manifest.json'),records=records,
        previous_best_rmse=manifest['previous_best_rmse'],scope=manifest['scope'],isolation=manifest['isolation'])
    u.write(REPORT/'evaluation.json',result);parent.verifier.score();return result


def guarded(command,log_path):
    with log_path.open('a') as log:
        child=subprocess.Popen(['nice','-n','10']+command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own,other=group_rss(child.pid),parent.resources.other_rss({child.pid});fs=os.statvfs(u.ROOT);free=fs.f_bavail*fs.f_frsize
                high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<22*2**30:raise RuntimeError('Matching-tree resource guard; partial outputs preserved')
                time.sleep(5)
        except BaseException:stop(child);raise
        if child.returncode:raise RuntimeError('Matching-tree child failed; partial outputs preserved')


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while parent.resources.other_rss(set())>16*2**30:
            u.write(REPORT/'status.json',dict(state='waiting_for_memory'));time.sleep(15)
        prepare();u.write(REPORT/'status.json',dict(state='nested_tree_fitting'))
        guarded([sys.executable,str(Path(__file__)),'--worker'],REPORT/'tree.log')
        u.write(REPORT/'status.json',dict(state='saved_tree_replay'))
        guarded([sys.executable,str(Path(__file__)),'--replay'],REPORT/'tree-replay.log')
        result=summarize();u.write(REPORT/'status.json',dict(state='development_complete',records=result['records'],goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();configure()
    if args.worker:n.worker('tree')
    elif args.replay:parent.verifier.replay('tree')
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
