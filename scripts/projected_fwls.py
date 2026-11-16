#!/usr/bin/env python3
"""Projected FWLS: explicit prediction/context products without a giant matrix.

Predictor disagreement is reduced without labels to eight principal directions.
Products with every source support/meta feature extend the four-context ridge.
All label-dependent regularization choices use inner development users only.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from gbdt_blend import load,blend_fold
from streaming_blend import digest,rmse


def encoding(raw,columns,stats):
    p=np.asarray(raw[:,:columns],dtype=np.float64)
    mean=p.mean(1)
    pc=(p-mean[:,None])@stats['projection']
    pc=(pc-stats['pc_mean'])/stats['pc_std']
    axes=np.column_stack([(mean-stats['mean_mean'])/stats['mean_std'],pc])
    m=np.asarray(raw[:,columns:],dtype=np.float64)
    m=(np.clip(m,stats['lo'],stats['hi'])-stats['meta_mean'])/stats['meta_std']
    return np.column_stack([np.ones(len(p)),(p-stats['pmean'])/stats['pstd'],
                            m,(axes[:,:,None]*m[:,None,:]).reshape(len(p),-1)])


def feature_stats(raw,columns,source_stats):
    n=len(raw);g=source_stats['gram'][:,:columns+1,:columns+1].sum(0)
    means=g[0,1:]/n+3.6
    cov=g[1:,1:]/n-np.outer(means-3.6,means-3.6)
    j=np.eye(columns)-np.ones((columns,columns))/columns
    val,vec=np.linalg.eigh(j@cov@j);projection=vec[:,-8:]
    pcmean=means@j@projection
    pcstd=np.sqrt(np.maximum(val[-8:],1e-12))
    meanmean=means.mean();meanstd=np.sqrt(max(cov.mean(),1e-12))
    result=dict(projection=projection,pc_mean=pcmean,pc_std=pcstd,
                mean_mean=meanmean,mean_std=meanstd,pmean=means,
                pstd=np.sqrt(np.maximum(np.diag(cov),1e-12)))
    width=raw.shape[1]-columns
    for name in ['lo','hi','meta_mean','meta_std']:result[name]=np.empty(width)
    for start in range(0,width,8):
        m=np.asarray(raw[:,columns+start:columns+start+8],dtype=np.float64)
        lo,hi=np.percentile(m,[1,99],axis=0);m=np.clip(m,lo,hi)
        for name,v in [('lo',lo),('hi',hi),('meta_mean',m.mean(0)),('meta_std',np.maximum(m.std(0),1e-6))]:
            result[name][start:start+len(v)]=v
    return result


def solutions(g,r,n):
    mu=g[0,1:]/g[0,0];ym=r[0]/g[0,0]
    cov=g[1:,1:]-g[0,0]*np.outer(mu,mu)
    cross=r[1:]-g[0,0]*mu*ym
    val,vec=np.linalg.eigh(cov);projection=vec.T@cross
    result=[]
    for alpha in [1e-5,1e-4,.001,.01,.1,1.]:
        coef=vec@(projection/(np.maximum(val,0)+n*alpha))
        result.append((alpha,np.r_[ym-mu@coef,coef]))
    return result


def fit(stats,train,valid=None,alpha=None):
    g,r,n=stats
    choices=solutions(g[train].sum(0),r[train].sum(0),n[train].sum())
    if valid is not None:
        vg,vr=g[valid].sum(0),r[valid].sum(0)
        alpha=min(choices,key=lambda item:item[1]@vg@item[1]-2*item[1]@vr)[0]
    return next((a,w) for a,w in choices if a==alpha)


def predict(raw,columns,scaler,w,rows=None):
    rows=np.arange(len(raw)) if rows is None else rows
    out=np.empty(len(rows))
    for start in range(0,len(rows),4096):
        ix=rows[start:start+4096]
        out[start:start+len(ix)]=3.6+encoding(raw[ix],columns,scaler)@w
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    b=json.loads(Path(a.bank+'.inputs.json').read_text());q=json.loads(Path(a.qual+'.json').read_text())
    if q['qualifying_labels_read'] is not False or q['bank_sha256']!=digest(a.bank+'.inputs.json'):
        raise ValueError('unverified qualifying features')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    outer=blend_fold(users);inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    groups=outer*2+inner;raw=np.load(a.bank+'.x.npy',mmap_mode='r')
    start=time.monotonic();print(json.dumps(dict(stage='feature_statistics')),flush=True)
    scaler=feature_stats(raw,b['columns'],np.load(a.bank+'.stats.npz'))
    np.savez(root/'encoding.npz',**scaler)
    width=encoding(raw[:2],b['columns'],scaler).shape[1]
    gram=np.zeros((10,width,width));rhs=np.zeros((10,width));count=np.bincount(groups,minlength=10)
    for group in range(10):
        rows=np.flatnonzero(groups==group)
        for lo in range(0,len(rows),4096):
            ix=rows[lo:lo+4096];x=encoding(raw[ix],b['columns'],scaler)
            gram[group]+=x.T@x;rhs[group]+=x.T@(y[ix]-3.6)
        print(json.dumps(dict(stage='statistics',group=group,seconds=time.monotonic()-start)),flush=True)
    np.savez(root/'statistics.npz',gram=gram,rhs=rhs,count=count)
    stats=gram,rhs,count;ids=np.arange(10);out=np.empty(len(y));folds=[]
    for fold in range(5):
        train=ids//2!=fold
        alpha,_=fit(stats,train&(ids%2==0),train&(ids%2==1))
        _,w=fit(stats,train,alpha=alpha);rows=np.flatnonzero(outer==fold)
        out[rows]=predict(raw,b['columns'],scaler,w,rows)
        record=dict(fold=fold,alpha=alpha,rmse=rmse(out[rows],y[rows]));folds.append(record)
        print(json.dumps(record),flush=True)
    alpha,_=fit(stats,ids%2==0,ids%2==1)
    _,w=fit(stats,np.ones(10,dtype=bool),alpha=alpha)
    np.save(root/'weights.npy',w,allow_pickle=False)
    out.astype('<f4').tofile(root/'development-oof.f32')
    qr=np.load(a.qual+'.x.npy',mmap_mode='r')
    final=np.clip(predict(qr,b['columns'],scaler,w),1,5).astype('<f4')
    final.tofile(root/'qualifying-final.f32')
    report=dict(state='complete',config=vars(a),qualifying_labels_read=False,qualifying_rows=len(final),
                bank_sha256=digest(a.bank+'.inputs.json'),qual_manifest_sha256=digest(a.qual+'.json'),
                development_rmse=rmse(out,y),folds=folds,production_alpha=alpha,features=width,
                unsupervised_encoding='Development feature covariance and quantiles; no label dependence',
                frozen_prediction_sha256=digest(root/'qualifying-final.f32'))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
