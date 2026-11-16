#!/usr/bin/env python3
"""Fit/freeze a qualifying ensemble using development labels exclusively.

All stopping decisions use an inner user-disjoint development partition.
Qualifying labels are not an argument, opened file, score, or training target.
The final prediction hash is written before a separate audit can run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from gbdt_blend import load, blend_fold
from streaming_blend import digest, predict_ridge
from streaming_nn_blend import standardized, train_device, predict_device
from blend_features import nn_features


def solve(stats, selected, alpha=1e-5):
    gram,rhs,count=stats
    a=gram[selected].sum(0)
    penalty=np.ones(len(a))*count[selected].sum()*alpha; penalty[0]=0
    a.flat[::len(a)+1]+=penalty
    return np.linalg.solve(a,rhs[selected].sum(0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True)
    ap.add_argument('--qual',required=True,help='prepared qualifying feature prefix')
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--tree-seeds',nargs='+',type=int,default=[2026,7,13])
    ap.add_argument('--nn-seeds',nargs='+',type=int,default=[2026,7,13])
    a=ap.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    if a.device=='mps' and not torch.backends.mps.is_available(): ap.error('MPS unavailable')
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['state']!='complete' or q['qualifying_labels_read'] is not False: raise ValueError('unverified qualifying inputs')
    if q['bank_sha256']!=digest(a.bank+'.inputs.json'): raise ValueError('different feature schemas')
    for suffix in ('.x.npy','.context.npy'):
        key='x_sha256' if suffix=='.x.npy' else 'context_sha256'
        if digest(a.qual+suffix)!=q[key]: raise ValueError('qualifying feature hash mismatch')
    d=load(b['config']['data'])
    y=d['probe']['rating'].astype(np.float64); users=d['probe']['user'].copy()
    if not np.all((y>=1)&(y<=5)): raise ValueError('development labels required; refuse unknown targets')
    if digest(b['config']['data'])!=b['data_sha256']: raise ValueError('development dataset changed')
    del d
    raw=np.load(a.bank+'.x.npy',mmap_mode='r')
    qr=np.load(a.qual+'.x.npy',mmap_mode='r')
    context=np.load(b['config']['meta'])['context']
    qc=np.load(a.qual+'.context.npy',mmap_mode='r')
    saved=np.load(a.bank+'.stats.npz'); stats=tuple(saved[k] for k in ('gram','rhs','count'))
    if stats[2].sum()!=len(y): raise ValueError('development statistics row mismatch')
    start=time.monotonic()
    report=dict(state='starting',config=vars(a),bank_sha256=digest(a.bank+'.inputs.json'),
                qual_manifest_sha256=digest(a.qual+'.json'),
                stats_sha256=digest(a.bank+'.stats.npz'),development_rows=len(y),qualifying_rows=len(qr),
                development_metric=b['config'].get('metric_label','Original Probe'),
                qualifying_labels_read=False,metric=None,
                blend_rule='Fixed equal average of three-seed residual tree and three-seed NN2',
                target_rmse=.85725)

    def status(state,**values):
        report.update(values,state=state,elapsed_seconds=time.monotonic()-start)
        temp=root/'report.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(root/'report.json')
        print(json.dumps(dict(state=state,**values)),flush=True)

    status('ridge')
    all_groups=np.ones(10,dtype=bool)
    full_w=solve(stats,all_groups)
    inner_w=solve(stats,np.arange(10)%2==0)
    np.savez(root/'ridge-weights.npz',full=full_w,inner=inner_w)
    p=raw[:,:b['columns']]
    base=predict_ridge(p,context,full_w)
    inner_base=predict_ridge(p,context,inner_w)
    qb=predict_ridge(qr[:,:b['columns']],qc,full_w)
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    fit,valid=~inner,inner
    np.save(root/'development-base.npy',base,allow_pickle=False)
    np.save(root/'development-inner-base.npy',inner_base,allow_pickle=False)
    np.save(root/'qualifying-base.npy',qb,allow_pickle=False)
    plan=dict(bank=a.bank,qual=a.qual,output=str(root),seeds=a.tree_seeds,threads=a.threads)
    (root/'tree-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    status('tree_worker')
    # A fresh interpreter isolates Homebrew LightGBM's libomp from PyTorch's
    # bundled libomp. Import order alone fails for both orders on this machine.
    try:
        subprocess.run([sys.executable,str(Path(__file__).with_name('qual_tree_worker.py')),
                        '--plan',str(root/'tree-plan.json')],check=True)
    except BaseException as error:
        status('failed',error=repr(error));raise
    worker=json.loads((root/'tree-report.json').read_text())
    if worker['state']!='complete' or worker['torch_loaded']: raise ValueError('invalid isolated tree result')
    tree_rounds=worker['tree_rounds']
    tree=np.load(root/'tree-predictions.npy')
    status('tree_complete',tree_rounds=tree_rounds,isolated_tree_worker=True)
    features=nn_features(raw,b['columns'],str(root/'nn-features.npy'))
    normal,scaler=standardized(features,np.ones(len(y),dtype=bool),
                              output=str(root/'nn-normalized.npy'),return_scaler=True)
    np.savez(root/'nn-scaler.npz',**scaler)
    cfg=SimpleNamespace(hidden=128,layers=2,dropout=.2,lr=.0003,weight_decay=.01,
                        batch=4096,epochs=60,patience=5,device=a.device)
    models=[]; nn_epochs=[]
    for seed in a.nn_seeds:
        _,epochs=train_device(normal[fit],y[fit]-inner_base[fit],cfg,seed,60,
                              valid=(normal[valid],y[valid]-inner_base[valid]))
        nn_epochs.append(epochs)
        status('nn_refit',seed=seed,epochs=epochs,nn_epochs=nn_epochs)
        model,_=train_device(normal,y-base,cfg,seed,epochs)
        torch.save(dict(state=model.state_dict(),config=vars(cfg),epochs=epochs,seed=seed),root/f'nn-{seed}.pt')
        models.append(model)
    del normal,features
    neural=np.zeros(len(qr))
    for lo in range(0,len(qr),65536):
        hi=min(lo+65536,len(qr))
        block=np.array(qr[lo:hi],dtype=np.float32,order='F')
        f=nn_features(block,b['columns']).astype(np.float64)
        f=((np.clip(f,scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std']).astype(np.float32)
        for model in models: neural[lo:hi]+=predict_device(model,f,a.device)/len(models)
    neural+=qb;neural.astype('<f4').tofile(root/'nn.f32')
    final=np.clip((tree+neural)/2,1,5).astype('<f4')
    if not np.isfinite(final).all() or len(final)!=q['rows']: raise ValueError('invalid final predictions')
    final.tofile(root/'qualifying-final.f32')
    hashes={p.name:digest(p) for p in root.iterdir() if p.suffix in ('.f32','.pt','.txt','.npz')}
    status('complete',tree_rounds=tree_rounds,nn_epochs=nn_epochs,outputs=hashes,
           frozen_prediction_sha256=hashes['qualifying-final.f32'],
           prediction_min=float(final.min()),prediction_max=float(final.max()),
           qualifying_labels_read=False,metric=None)


if __name__=='__main__': main()
