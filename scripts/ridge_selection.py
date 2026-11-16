#!/usr/bin/env python3
"""Nested development-only selection of ridge strength and contextual features."""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load,blend_fold
from streaming_blend import digest,predict_ridge,rmse


def candidates(gram,rhs,count):
    result=[]
    for name,width in [('global',(len(rhs)//5)),('context',len(rhs))]:
        g=gram[:width,:width];r=rhs[:width]
        center=g[0,1:]/g[0,0];ym=r[0]/g[0,0]
        cov=g[1:,1:]-g[0,0]*np.outer(center,center)
        cross=r[1:]-g[0,0]*center*ym
        val,vec=np.linalg.eigh(cov)
        projection=vec.T@cross
        for alpha in [1e-7,1e-6,1e-5,1e-4,1e-3,.01]:
            coef=vec@(projection/(np.maximum(val,0)+count*alpha))
            w=np.zeros(len(rhs));w[1:width]=coef;w[0]=ym-center@coef
            result.append((dict(context=name,alpha=alpha),w))
    return result


def choose(stats,fit,valid):
    g,r,n=stats
    options=candidates(g[fit].sum(0),r[fit].sum(0),n[fit].sum())
    vg,vr=g[valid].sum(0),r[valid].sum(0)
    losses=[float(w@vg@w-2*w@vr) for _,w in options]
    return options[int(np.argmin(losses))][0]


def refit(stats,groups,recipe):
    g,r,n=stats
    return next(w for config,w in candidates(g[groups].sum(0),r[groups].sum(0),n[groups].sum())
                if config==recipe)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['qualifying_labels_read'] is not False or q['bank_sha256']!=digest(a.bank+'.inputs.json'):
        raise ValueError('unverified feature schema')
    s=np.load(a.bank+'.stats.npz');stats=tuple(s[k] for k in ('gram','rhs','count'))
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);outer=blend_fold(d['probe']['user']);del d
    x=np.load(a.bank+'.x.npy',mmap_mode='r');qx=np.load(a.qual+'.x.npy',mmap_mode='r')
    c=np.load(b['config']['meta'])['context'];qc=np.load(a.qual+'.context.npy',mmap_mode='r')
    groups=np.arange(10);out=np.empty(len(y));folds=[]
    for fold in range(5):
        train=groups//2!=fold
        recipe=choose(stats,train&(groups%2==0),train&(groups%2==1))
        w=refit(stats,train,recipe);held=outer==fold
        out[held]=predict_ridge(x[held,:b['columns']],c[held],w)
        record=dict(fold=fold,recipe=recipe,rmse=rmse(out[held],y[held]));folds.append(record)
        print(json.dumps(record),flush=True)
    recipe=choose(stats,groups%2==0,groups%2==1)
    w=refit(stats,np.ones(10,dtype=bool),recipe)
    np.save(root/'weights.npy',w,allow_pickle=False)
    out.astype('<f4').tofile(root/'development-oof.f32')
    final=np.clip(predict_ridge(qx[:,:b['columns']],qc,w),1,5).astype('<f4')
    final.tofile(root/'qualifying-final.f32')
    report=dict(state='complete',config=vars(a),qualifying_labels_read=False,qualifying_rows=len(final),
                development_rmse=rmse(out,y),development_folds=folds,production_recipe=recipe,
                bank_sha256=digest(a.bank+'.inputs.json'),qual_manifest_sha256=digest(a.qual+'.json'),
                frozen_prediction_sha256=digest(root/'qualifying-final.f32'))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
