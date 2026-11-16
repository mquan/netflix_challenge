#!/usr/bin/env python3
"""Train development-only tree blenders before qualifying base exports exist."""
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
    if 'torch' in sys.modules:raise RuntimeError('tree fitting must be isolated from PyTorch')
    b=json.loads(Path(plan['bank']+'.inputs.json').read_text())
    bank_sha=digest(plan['bank']+'.inputs.json')
    if plan.get('bank_sha256',bank_sha)!=bank_sha:raise ValueError('development bank changed')
    for name,expected in plan.get('inputs',{}).items():
        if digest(root/name)!=expected:raise ValueError('development baseline changed')
    if digest(b['config']['data'])!=b['data_sha256']:raise ValueError('development data changed')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    raw=np.load(plan['bank']+'.x.npy',mmap_mode='r')
    base=np.load(root/'development-base.npy',mmap_mode='r');ib=np.load(root/'development-inner-base.npy',mmap_mode='r')
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    fit,valid=~inner,inner;rounds=[]
    ignored=plan.get('ignored_columns',[])
    if any(i<0 or i>=raw.shape[1] for i in ignored):raise ValueError('invalid ignored feature')
    for seed in plan['seeds']:
        params=dict(objective='regression',learning_rate=.02,num_leaves=63,min_data_in_leaf=200,
                    feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,
                    num_threads=plan['threads'],verbose=-1,seed=seed)
        if ignored:
            contribution=np.ones(raw.shape[1]);contribution[ignored]=0
            params['feature_contri']=contribution.tolist()
        dt=lgb.Dataset(raw[fit],y[fit]-ib[fit]);dv=lgb.Dataset(raw[valid],y[valid]-ib[valid],reference=dt)
        model=lgb.train(params,dt,5000,valid_sets=[dv],callbacks=[lgb.early_stopping(100,verbose=False)])
        n=model.best_iteration;rounds.append(n);del model,dt,dv
        print(json.dumps(dict(state='tree_refit',seed=seed,rounds=n)),flush=True)
        model=lgb.train(params,lgb.Dataset(raw,y-base),n)
        if ignored and np.any(model.feature_importance(importance_type='split')[ignored]):
            raise ValueError('ignored feature was used in a split')
        model.save_model(str(root/f'tree-{seed}.txt'));del model
        print(json.dumps(dict(state='tree_seed_complete',seed=seed,rounds=rounds)),flush=True)
    report=dict(state='complete',tree_rounds=rounds,qualifying_inputs_provided=False,
                qualifying_labels_read=False,torch_loaded=False,lightgbm_version=lgb.__version__,
                bank_sha256=bank_sha,plan_sha256=digest(a.plan),ignored_columns=ignored,
                outputs={p.name:digest(p) for p in root.glob('tree-*.txt')})
    (root/'tree-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
