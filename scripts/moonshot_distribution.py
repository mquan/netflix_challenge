#!/usr/bin/env python3
"""Matched mean and rating-distribution neural heads on frozen union features."""
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
import torch
from gbdt_blend import load, blend_fold
from moonshot_nn_blend import design, ridge_fit, fit_nn, infer


def distribution_predict(model, x, base):
    levels=torch.arange(1,6,dtype=x.dtype)
    logits=model(x)-.5*(levels[None,:]-base[:,None]).square()
    return logits, (logits.softmax(1)*levels).sum(1)


def calibrated_centers(base):
    """Choose prior logits whose bounded expected rating matches the ridge mean."""
    levels=torch.arange(1,6,dtype=base.dtype)
    target=base.clamp(1.0001,4.9999); center=target.clone()
    for _ in range(16):
        prob=(-.5*(levels[None,:]-center[:,None]).square()).softmax(1)
        mean=(prob*levels).sum(1)
        variance=(prob*(levels[None,:]-mean[:,None]).square()).sum(1)
        center=(center+(target-mean)/variance.clamp_min(1e-5)).clamp(-20,20)
    return center


def fit_distribution(x,y,baseline,*,epochs,seed,validation=None,calibrated=False):
    torch.manual_seed(seed)
    mean=x.mean(0,dtype=np.float64); std=np.maximum(x.std(0,dtype=np.float64),.01)
    tx=torch.from_numpy(((x-mean)/std).astype(np.float32))
    ty=torch.from_numpy(y.astype(np.int64)-1)
    tb=torch.from_numpy(baseline.astype(np.float32))
    if calibrated: tb=calibrated_centers(tb)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,5))
    torch.nn.init.zeros_(model[-1].weight); torch.nn.init.zeros_(model[-1].bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best_epoch=0; best_score=float('inf')
    if validation is not None:
        vx,vy,vb=validation
        best_score=float(np.mean((np.clip(vb,1,5)-vy)**2))
        vx=torch.from_numpy(((vx-mean)/std).astype(np.float32))
        vb=torch.from_numpy(vb.astype(np.float32))
        if calibrated: vb=calibrated_centers(vb)
    for epoch in range(1,epochs+1):
        model.train()
        for batch in torch.randperm(len(tx)).split(8192):
            opt.zero_grad(set_to_none=True)
            logits,_=distribution_predict(model,tx[batch],tb[batch])
            torch.nn.functional.cross_entropy(logits,ty[batch]).backward(); opt.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                pred=np.concatenate([distribution_predict(model,vx[s:s+65536],vb[s:s+65536])[1].numpy()
                                     for s in range(0,len(vx),65536)])
            score=float(np.mean((pred-vy)**2))
            if score<best_score: best_epoch,best_score=epoch,score
            if epoch-best_epoch>=6: break
        else: best_epoch=epoch
    # Inner models are discarded; only their chosen duration is used in refitting.
    return model,mean,std,best_epoch


def fit_fold(predictions,context,metadata,y,users,fold,*,head,epochs=60,calibrated=False):
    torch.set_num_threads(1)
    rx,nx=design(predictions,context)
    nx=np.column_stack([nx,metadata]).astype(np.float32)
    train=blend_fold(users)!=fold; held=~train
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    fit,valid=train&~inner,train&inner
    wi=ridge_fit(rx[fit],y[fit]); inner_base=3.6+rx@wi
    seed=20260928+fold
    fitter=fit_distribution if head=='distribution' else fit_nn
    extra={'calibrated':calibrated} if head=='distribution' else {'hidden':32}
    _,_,_,chosen=fitter(nx[fit],y[fit],inner_base[fit],epochs=epochs,seed=seed,
                         validation=(nx[valid],y[valid],inner_base[valid]),**extra)
    w=ridge_fit(rx[train],y[train]); base=3.6+rx@w
    result=base[held]
    if chosen:
        model,mean,std,_=fitter(nx[train],y[train],base[train],epochs=chosen,seed=seed,**extra)
        if head=='distribution':
            model.eval()
            tx=torch.from_numpy(((nx[held]-mean)/std).astype(np.float32))
            tb=torch.from_numpy(base[held].astype(np.float32))
            if calibrated: tb=calibrated_centers(tb)
            with torch.no_grad():
                result=np.concatenate([distribution_predict(model,tx[s:s+65536],tb[s:s+65536])[1].numpy()
                                       for s in range(0,len(tx),65536)])
        else: result=result+infer(model,mean,std,nx[held])
    return result,chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--head',choices=['mean','distribution'],required=True)
    p.add_argument('--calibrated-prior',action='store_true')
    p.add_argument('--epochs',type=int,default=60); a=p.parse_args()
    if pathlib.Path(a.output+'.json').exists(): p.error('output already exists')
    bank=json.loads(pathlib.Path('experiments/UNION-round-manifest.json').read_text())['reports']['union']['inputs']
    support=json.loads(pathlib.Path('experiments/BOOST-support-inputs.json').read_text())
    columns=[]
    for name,expected in bank.items():
        raw=pathlib.Path(name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError(name)
        columns.append(np.frombuffer(raw,dtype='<f4'))
    pred=np.column_stack(columns)
    with np.load('artifacts/boost-union-features.npz') as cache:
        expected=hashlib.sha256(json.dumps(dict(bank,**support),sort_keys=True).encode()).hexdigest()
        if str(cache['bank_hash'])!=expected: raise ValueError('feature cache mismatch')
        context=cache['context']; metadata=cache['x'][:,len(columns):]
    d=load(a.data); y=d['probe']['rating'].astype(np.float64); user=d['probe']['user']
    cv=np.empty(len(y)); folds=blend_fold(user); records=[]
    for f in range(5):
        start=time.monotonic(); held=folds==f
        cv[held],chosen=fit_fold(pred,context,metadata,y,user,f,head=a.head,epochs=a.epochs,calibrated=a.calibrated_prior)
        record=dict(fold=f,chosen_epochs=chosen,rmse=float(np.sqrt(np.mean((np.clip(cv[held],1,5)-y[held])**2))),
                    seconds=time.monotonic()-start)
        records.append(record); print(json.dumps(record),flush=True)
    cv.astype('<f4').tofile(a.output+'.f32')
    report=dict(config=vars(a),inputs=dict(bank,**support),folds=records,
                rmse=float(np.sqrt(np.mean((np.clip(cv,1,5)-y)**2))))
    pathlib.Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
