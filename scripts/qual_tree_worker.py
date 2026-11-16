#!/usr/bin/env python3
"""Isolated LightGBM worker: never initialize PyTorch's second OpenMP runtime."""
import argparse
import json
from pathlib import Path
import sys
import lightgbm as lgb
import numpy as np
from gbdt_blend import load,blend_fold
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True)
    a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());root=Path(plan['output'])
    if 'torch' in sys.modules: raise RuntimeError('tree worker must not load PyTorch')
    b=json.loads(Path(plan['bank']+'.inputs.json').read_text())
    if digest(b['config']['data'])!=b['data_sha256']: raise ValueError('development data changed')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    raw=np.load(plan['bank']+'.x.npy',mmap_mode='r');qr=np.load(plan['qual']+'.x.npy',mmap_mode='r')
    base=np.load(root/'development-base.npy',mmap_mode='r')
    ib=np.load(root/'development-inner-base.npy',mmap_mode='r')
    qb=np.load(root/'qualifying-base.npy',mmap_mode='r')
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    fit,valid=~inner,inner
    tree=np.zeros(len(qr));rounds=[]
    for seed in plan['seeds']:
        params=dict(objective='regression',learning_rate=.02,num_leaves=63,min_data_in_leaf=200,
                    feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,
                    num_threads=plan['threads'],verbose=-1,seed=seed)
        dt=lgb.Dataset(raw[fit],y[fit]-ib[fit])
        dv=lgb.Dataset(raw[valid],y[valid]-ib[valid],reference=dt)
        model=lgb.train(params,dt,5000,valid_sets=[dv],callbacks=[lgb.early_stopping(100,verbose=False)])
        n=model.best_iteration;rounds.append(n);del dt,dv,model
        print(json.dumps(dict(state='tree_refit',seed=seed,rounds=n)),flush=True)
        model=lgb.train(params,lgb.Dataset(raw,y-base),n)
        model.save_model(str(root/f'tree-{seed}.txt'))
        for lo in range(0,len(qr),65536):
            tree[lo:lo+65536]+=model.predict(qr[lo:lo+65536],num_threads=plan['threads'])/len(plan['seeds'])
        del model
        print(json.dumps(dict(state='tree_seed_complete',seed=seed,tree_rounds=rounds)),flush=True)
    tree+=qb
    np.save(root/'tree-predictions.npy',tree,allow_pickle=False)
    tree.astype('<f4').tofile(root/'tree.f32')
    report=dict(state='complete',tree_rounds=rounds,torch_loaded='torch' in sys.modules,
                lightgbm_version=lgb.__version__,worker_sha256=digest(__file__),qualifying_labels_read=False)
    (root/'tree-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
