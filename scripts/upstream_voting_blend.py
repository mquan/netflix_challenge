#!/usr/bin/env python3
"""Add freshly reproduced published voting features to the fixed nested blender."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import upstream_remote_batch as parent
import upstream_nonlinear as n
import verify_upstream_nonlinear as verifier
from upstream_full_reproduce import receipt_valid

u,x=n.u,n.x
RUN=u.ROOT/'artifacts/upstream-voting-blend'
REPORT=u.ROOT/'experiments/upstream-voting-blend'
NEW=['mf-61','mf-61__knn3']


def code_hashes():
    return {**parent.code_hashes(),str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),
            'scripts/upstream_full_reproduce.py':u.sha(u.ROOT/'scripts/upstream_full_reproduce.py')}


def configure():
    n.RUN,n.REPORT,n.code_hashes=RUN,REPORT,code_hashes


def feature_inventory(receipt):
    files=sorted(p for p in receipt['outputs'] if p.startswith('preds_new/vf/') and p.endswith('.probex.npy'))
    if len(files)!=130:raise ValueError('Expected all 130 published voting features')
    if [Path(p).name[:5] for p in files]!=[f'vf{i:03}' for i in range(130)]:
        raise ValueError('Missing or ambiguous voting feature indices')
    for p in files:
        if p.removesuffix('.probex.npy')+'.qual.npy' not in receipt['outputs']:
            raise ValueError('Feature lacks fulltrain/qual counterpart')
    return files


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial feature preparation preserved; inspect before resuming')
    parent.configure();pm=n.verify_prepared();configure()
    parent_evaluation=parent.REPORT/'evaluation.json'
    if u.read(parent.REPORT/'independent-verification.json')['evaluation_sha256']!=u.sha(parent_evaluation):
        raise ValueError('Parent score verification changed')
    jobs=u.read(u.BASE/'registry.json')['jobs'];sources={}
    for name in ['vfeat1']+NEW:
        if not receipt_valid(name,jobs):raise ValueError('Fresh native job not complete: '+name)
        source=u.BASE/'fresh-receipts'/(name+'.json')
        sources[str(source.relative_to(u.ROOT))]=u.sha(source)
    vf=feature_inventory(u.read(u.BASE/'fresh-receipts/vfeat1.json'))
    old=np.load(parent.RUN/'design.npy',mmap_mode='r')
    old_features=np.load(parent.RUN/'features.npy',mmap_mode='r')
    y,users,groups=[np.load(parent.RUN/(key+'.npy')) for key in ['labels','users','groups']]
    native=u.FRESH/'data/probex'
    if not np.array_equal(y,np.load(native/'ratings.npy')) or not np.array_equal(users,np.load(native/'user_idxs.npy')):
        raise ValueError('Source development identities changed')
    for key in ['ratings','user_idxs','item_idxs','dates']:
        path=native/(key+'.npy');sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    added=np.clip(np.column_stack([np.load(u.FRESH/'preds_new'/(name+'.probex.npy')) for name in NEW]),0,6)
    if added.shape!=(len(y),2) or not np.isfinite(added).all():raise ValueError('Invalid new native prediction')
    values=[np.load(u.FRESH/path,mmap_mode='r') for path in vf]
    if any(a.shape!=y.shape or a.dtype!=np.dtype('float32') for a in values):raise ValueError('Invalid voting feature shape/type')
    REPORT.mkdir(parents=True,exist_ok=True)
    u.write(REPORT/'plan.json',dict(parent_manifest_sha256=u.sha(parent.REPORT/'manifest.json'),
        new_predictions=NEW,published_features=vf,config=n.CONFIG,code_sha256=code_hashes(),
        scope='Fixed addition of both newly trained MF61 predictions and all 130 freshly trained published voting features. No feature subset or blend-weight search on scoring folds.'))
    scored=parent.screen(old,added,y,groups,dict(control=[],main=[0,1]))
    if abs(x.metric(scored['control']['prediction'],y)-pm['baseline_rmse'])>1e-9:raise ValueError('Parent ridge control mismatch')
    main=scored['main'];RUN.mkdir(parents=True)
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(len(y),old.shape[1]+2))
    features=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,
        shape=(len(y),old_features.shape[1]+2+len(values)))
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));v=np.column_stack([a[lo:hi] for a in values])
        if not np.isfinite(v).all():raise ValueError('Nonfinite published voting feature')
        z[lo:hi]=parent.augmented_block(old,added,lo,hi)
        features[lo:hi]=np.column_stack([old_features[lo:hi],added[lo:hi]-3.6,v])
    z.flush();features.flush()
    np.savez(RUN/'stats.npz',gram=main['gram'],rhs=main['rhs'])
    for name,value in [('labels',y),('users',users),('groups',groups),('baseline',main['prediction'])]:np.save(RUN/(name+'.npy'),value)
    for path in [parent.REPORT/'manifest.json',parent.REPORT/'evaluation.json',REPORT/'plan.json']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    manifest=dict(config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        baseline_rmse=x.metric(main['prediction'],y),parent_baseline_rmse=pm['baseline_rmse'],
        rows=len(y),prediction_columns=85,design_columns=z.shape[1],nonlinear_features=features.shape[1],
        published_features=vf,new_native_predictions=NEW,
        isolation='Voting statistics and latent features fit only on trainx ratings; query rows are MaskedDataset inputs in pinned save_vfeat1. Blender ridge/scalers/budget selection exclude outer scoring users. Qualifying outcomes are zero throughout training.',
        scope='Adaptive ProbeX development experiment using freshly trained predictors and published source feature code; not an official or untouched qualifying score.',arms=pm['arms'])
    u.write(REPORT/'manifest.json',manifest);return manifest


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare();child=None
        try:
            # One blender family at a time leaves RAM for both native training trees.
            for family in ['tree','nn']:
                u.write(REPORT/'status.json',dict(state='training',family=family))
                with (REPORT/(family+'.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',family],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    child.wait()
                if child.returncode:raise RuntimeError('Voting blender worker failed')
                subprocess.run([sys.executable,str(Path(__file__)),'--verify-family',family],check=True)
            result=n.summarize()
            previous=u.read(parent.REPORT/'evaluation.json')['records'][-1]['rmse']
            result['parent_best_rmse']=previous
            for row in result['records']:row['delta_vs_previous_best']=row['rmse']-previous
            u.write(REPORT/'evaluation.json',result)
            verifier.score()
            u.write(REPORT/'status.json',dict(state='complete',best=min(r['rmse'] for r in result['records'])))
        except BaseException as error:
            if child is not None and child.poll() is None:
                from upstream_full_reproduce import stop
                stop(child)
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--worker',choices=['tree','nn'])
    ap.add_argument('--verify-family',choices=['tree','nn'])
    ap.add_argument('--prepare',action='store_true');args=ap.parse_args();os.nice(10);configure()
    if args.worker:n.worker(args.worker)
    elif args.verify_family:verifier.replay(args.verify_family)
    elif args.prepare:prepare()
    else:run()
