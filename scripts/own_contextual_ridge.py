"""Vary own-model combination weights with outcome-free context coordinates."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import blas,solve
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_history_context as hist
import own_full_context as full

OUT=ne.ROOT/'own-contextual-ridge-20261126'
PENALTIES=(2500.,20000.,100000.,400000.)
ROWS=4096


def orthogonal(x,k):
    g=x.T@x/len(x)
    v,b=np.linalg.eigh(g)
    keep=np.flatnonzero(v>1e-5)[-k:][::-1]
    return np.ascontiguousarray(b[:,keep]/np.sqrt(np.maximum(v[keep],.01)))


class Design:
    def __init__(self,x,nmodels):
        self.k=nmodels
        rng=np.random.default_rng(20261126)
        ids=np.sort(rng.choice(len(x),90000,replace=False))
        sample=x[ids]
        self.ref=sample.mean(0,dtype=float).astype('f')
        self.scale=np.maximum(sample.std(0,dtype=float),.001).astype('f')
        z=np.clip((sample-self.ref)/self.scale,-5,5)
        preds=z[:,1:self.k+1]
        self.vp=orthogonal(preds,30)
        m=self.meta(z)
        self.knots=np.percentile(m,[20,50,80],axis=0).astype('f')
        phi=self.splines(m)
        self.center=np.mean(phi,axis=0)
        self.vm=orthogonal(phi-self.center,48)
        self.vm_bias=orthogonal(phi-self.center,128)
    def meta(self,z):
        return np.column_stack((z[:,0],z[:,self.k+1:]))
    def splines(self,m):
        return np.column_stack([m]+[np.maximum(m-k,0.) for k in self.knots])
    def __call__(self,x):
        z=np.clip((x-self.ref)/self.scale,-5,5)
        pred=np.clip(z[:,1:self.k+1]@self.vp,-6.,6.)
        phi=self.splines(self.meta(z))-self.center
        ctx=np.clip(phi@self.vm,-6,6)
        broad=np.clip(phi@self.vm_bias,-6,6)
        products=(pred[:,:,None]*ctx[:,None,:]).reshape(len(x),-1)
        return np.asfortranarray(np.column_stack((np.ones(len(x),'f'),pred,broad,products)))


def main():
    OUT.mkdir(exist_ok=True)
    y,folds,p,q=anchor.history_control()
    prep=json.loads((full.OUT/'probex-sources.json').read_text())
    x=np.load(full.OUT/'probex.npy',mmap_mode='r')
    model=Design(x,len(prep['columns']))
    dim=model(x[:2]).shape[1]
    print('dimensions',x.shape,dim,flush=True)
    moments=[]
    for f in range(5):
        ids=np.flatnonzero(folds==f)
        gram=np.zeros((dim,dim),dtype=float)
        rhs=np.zeros(dim)
        for lo in range(0,len(ids),ROWS):
            ix=ids[lo:lo+ROWS]
            a=model(x[ix])
            gram+=blas.ssyrk(1.,a,lower=1,trans=1)
            rhs+=a.T@(y[ix]-p[ix]).astype('f')
        moments.append((gram,rhs))
        print('fold statistics',f,flush=True)
    total_g=sum(a for a,b in moments)
    total_rhs=sum(b for a,b in moments)
    def solve_ridge(g,r,pen):
        a=g.copy(order='F')
        a.flat[::len(a)+1]+=pen
        a[0,0]-=pen-1
        return solve(a,r,lower=True,assume_a='pos',check_finite=False)
    held=np.empty((len(x),len(PENALTIES)),dtype='f')
    for f in range(5):
        w=np.column_stack([solve_ridge(total_g-moments[f][0],total_rhs-moments[f][1],alpha) for alpha in PENALTIES]).astype('f')
        ids=np.flatnonzero(folds==f)
        for lo in range(0,len(ids),ROWS):
            ix=ids[lo:lo+ROWS]
            held[ix]=model(x[ix])@w
    np.save(OUT/'oof.npy',held)
    # Each fit already has a shrinkage prior; compare its unscaled correction.
    def metrics(v):return dict(pooled=ne.rmse(v,y),folds=[ne.rmse(v[folds==f],y[folds==f]) for f in range(5)])
    stats={'control':metrics(p)}
    for j,alpha in enumerate(PENALTIES):
        stats[str(j)]=metrics(ne.clipped(p+held[:,j]))
    good=[k for k in stats if k!='control' and stats[k]['pooled']<stats['control']['pooled']
          and sum(a<b for a,b in zip(stats[k]['folds'],stats['control']['folds']))>=4]
    sel=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    report=dict(selected=sel,known=stats,penalties=PENALTIES,features=dim,qualifying_hashes={},
                qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False,
                baseline_sha256=ne.sha(hist.OUT/'selected.f32'))
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if sel!='control':
        w=solve_ridge(total_g,total_rhs,PENALTIES[int(sel)]).astype('f')
        qx=np.load(full.OUT/'qual.npy',mmap_mode='r')
        assert prep['columns']==json.loads((full.OUT/'qual-sources.json').read_text())['columns']
        delta=np.empty(len(q),dtype='f')
        for lo in range(0,len(q),ROWS):
            sl=slice(lo,lo+ROWS)
            delta[sl]=model(qx[sl])@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if sel!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
