#!/usr/bin/env python3
"""Freeze the independently validated NN2 component while tree fits continue.

Reuses only immutable raw-feature encodings and development-fitted ridge
baselines. No cross-fitted predictions or qualifying labels train this model.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from gbdt_blend import load,blend_fold
from fit_qual_blend import nn_features
from streaming_blend import digest
from streaming_nn_blend import standardized,train_device,predict_device


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True)
    ap.add_argument('--qual',required=True)
    ap.add_argument('--base-dir',required=True)
    ap.add_argument('--features',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps')
    ap.add_argument('--resident-features',action='store_true')
    ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--weight-decay',type=float,default=.01)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual+'.json').read_text())
    parent=json.loads((Path(a.base_dir)/'report.json').read_text())
    if parent['config']['bank']!=a.bank or parent['config']['qual']!=a.qual: raise ValueError('different parent bank')
    if parent['qualifying_labels_read'] is not False or q['qualifying_labels_read'] is not False: raise ValueError('labels not isolated')
    if q['bank_sha256']!=digest(a.bank+'.inputs.json'): raise ValueError('feature schema differs')
    for suffix,key in [('.x.npy','x_sha256'),('.context.npy','context_sha256')]:
        if digest(a.qual+suffix)!=q[key]: raise ValueError('qualifying feature hash mismatch')
    if digest(b['config']['data'])!=b['data_sha256']: raise ValueError('development data changed')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    if not np.all((y>=1)&(y<=5)): raise ValueError('development labels required')
    sample_weight=None
    if b['config'].get('sample_weights'):
        weight_path=b['config']['sample_weights']
        if digest(weight_path)!=b['sample_weights_sha256']:raise ValueError('training weights changed')
        sample_weight=np.load(weight_path,allow_pickle=False)
        if sample_weight.shape!=y.shape:raise ValueError('weight row count differs')
    raw=np.load(a.bank+'.x.npy',mmap_mode='r');qr=np.load(a.qual+'.x.npy',mmap_mode='r')
    features=np.load(a.features,mmap_mode='r')
    if features.shape!=(len(y),raw.shape[1]+1): raise ValueError('feature shape differs')
    rows=np.linspace(0,len(y)-1,min(256,len(y)),dtype=int)
    expected=nn_features(np.array(raw[rows],order='F'),b['columns'])
    if not np.array_equal(features[rows],expected): raise ValueError('cached encodings do not match raw bank')
    base_dir=Path(a.base_dir)
    paths={k:base_dir/v for k,v in [('full','development-base.npy'),('inner','development-inner-base.npy'),('qual','qualifying-base.npy')]}
    base,ib,qb=(np.load(paths[k],mmap_mode='r') for k in ('full','inner','qual'))
    if len(base)!=len(y) or len(ib)!=len(y) or len(qb)!=len(qr): raise ValueError('ridge baseline shape differs')
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    fit,valid=~inner,inner
    start=time.monotonic()
    report=dict(state='starting',config=vars(a),bank_sha256=digest(a.bank+'.inputs.json'),
                qual_manifest_sha256=digest(a.qual+'.json'),ridge_inputs={k:digest(v) for k,v in paths.items()},
                nn_features_sha256=digest(a.features),development_rows=len(y),qualifying_rows=len(qr),
                development_metric=b['config'].get('metric_label','Original Probe'),
                blend_rule='Three fixed residual-neural seeds (2026,7,13), contextual ridge baseline, equal seed average',
                sample_weights_sha256=b.get('sample_weights_sha256'),
                qualifying_labels_read=False,metric=None,target_rmse=.85725)

    def status(state,**values):
        report.update(state=state,elapsed_seconds=time.monotonic()-start,**values)
        p=root/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(root/'report.json')
        print(json.dumps(dict(state=state,**values)),flush=True)

    status('normalization')
    normal,scaler=standardized(features,np.ones(len(y),dtype=bool),
                              output=str(root/'normalized.npy'),return_scaler=True)
    np.savez(root/'scaler.npz',**scaler)
    cfg=SimpleNamespace(hidden=a.hidden,layers=a.layers,dropout=a.dropout,lr=a.lr,weight_decay=a.weight_decay,
                        batch=4096,patience=5,device=a.device,resident_features=a.resident_features)
    models=[];epochs_used=[]
    for seed in [2026,7,13]:
        _,epochs=train_device(normal[fit],y[fit]-ib[fit],cfg,seed,60,
                              valid=((normal[valid],y[valid]-ib[valid]) if sample_weight is None else
                                     (normal[valid],y[valid]-ib[valid],sample_weight[valid])),
                              sample_weights=None if sample_weight is None else sample_weight[fit])
        epochs_used.append(epochs);status('refit',seed=seed,epochs=epochs)
        model,_=train_device(normal,y-base,cfg,seed,epochs,sample_weights=sample_weight)
        torch.save(dict(state=model.state_dict(),config=vars(cfg),seed=seed,epochs=epochs),root/f'nn-{seed}.pt')
        models.append(model);status('seed_complete',seed=seed,epochs_used=epochs_used)
    del normal,features
    result=np.array(qb,copy=True)
    for lo in range(0,len(qr),65536):
        hi=min(lo+65536,len(qr))
        block=np.array(qr[lo:hi],dtype=np.float32,order='F')
        f=nn_features(block,b['columns']).astype(np.float64)
        f=((np.clip(f,scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std']).astype(np.float32)
        for model in models: result[lo:hi]+=predict_device(model,f,a.device)/3
    final=np.clip(result,1,5).astype('<f4')
    if not np.isfinite(final).all() or len(final)!=q['rows']: raise ValueError('invalid predictions')
    final.tofile(root/'qualifying-final.f32')
    hashes={p.name:digest(p) for p in root.iterdir() if p.suffix in ('.pt','.npz','.f32')}
    status('complete',outputs=hashes,nn_epochs=epochs_used,frozen_prediction_sha256=hashes['qualifying-final.f32'],
           prediction_min=float(final.min()),prediction_max=float(final.max()))


if __name__=='__main__': main()
