"""Estimate a same-user Gaussian residual update from observable original Probe ratings."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import own_neighbor_experiment as ne
import own_retained_history as retained
import own_anchor_context as anchor
import own_history_context as hist

OUT=ne.ROOT/'own-anchor-covariance-20261126'
PARENT=ne.ROOT/'own-overnight-blend-20261126/first'


def original():
    u,i,d,joint=anchor.old_probe()
    histcols=[]
    for name in retained.MODELS:
        report=json.loads((retained.QUAL/name/'report.json').read_text())
        path=Path(report['column'])
        assert ne.sha(path)==report['column_sha256']
        histcols.append(np.fromfile(path,dtype='<f4'))
    recipe=json.loads(Path('experiments/QUAL-B2-fixed128-recipe-20261116.json').read_text())
    bp=Path(recipe['column'])
    assert ne.sha(bp)==recipe['verified_files'][str(bp.relative_to(Path.cwd()))]
    bayes=np.fromfile(bp,dtype='<f4')
    old=.20*joint+.20*bayes+.36*histcols[0]+.24*np.mean(histcols[1:],axis=0)
    return u.astype('i'),i.astype('i'),d.astype('i'),old


def movie_geometry():
    path=ne.ROOT/'own-session-latent-20261126/graph-coordinates.npy'
    assert ne.sha(path)==json.loads((path.parent/'graph.json').read_text())['embedding_sha256']
    g=np.load(path)
    return g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)


def kernels(ia,da,pa,ib,db,pb,g):
    dt=np.abs(da-db).astype(float)
    sim=np.einsum('...k,...k->...',g[ia],g[ib])
    date=np.exp(-dt/14.)
    return np.stack([np.ones_like(dt),np.exp(-dt/120.),date,(dt==0).astype(float),sim**2,
                     date*sim**2,np.clip(pa-3.5,-2.,2.)*np.clip(pb-3.5,-2.,2.)],axis=-1)


def estimate(u,i,d,prior,ratings,g):
    # All pair moments involve ratings in the original Netflix training file.
    # Cxx controls how strongly multiple observed anchors duplicate evidence.
    # The ProbeX arm has at most one training anchor, so Cxx off-diagonals
    # never enter any ProbeX forecast.
    order=np.argsort(u,kind='stable')
    off=np.r_[0,np.bincount(u,minlength=480189).cumsum()]
    e=ratings-prior
    gram=np.zeros((7,7));rhs=np.zeros(7)
    for lag in range(1,int(np.diff(off).max())):
        a,b=order[:-lag],order[lag:]
        keep=u[a]==u[b]
        a,b=a[keep],b[keep]
        if not len(a):continue
        x=kernels(i[a],d[a],prior[a],i[b],d[b],prior[b],g)
        gram+=x.T@x;rhs+=x.T@(e[a]*e[b])
    gram.flat[::8]+=100
    l=np.linalg.cholesky(gram)
    coef=nnls(l.T,np.linalg.solve(l,rhs))[0]
    return coef,float(np.mean(e**2))


def features(train,query,p,g,coef,variance):
    u,i,d,prior=original()
    present,ix=anchor.training_indices(train,u,i,d)
    u,i,d,prior=[a[present] for a in (u,i,d,prior)]
    e=ne.history(train,'ratings')[ix]-prior
    order=np.argsort(u,kind='stable')
    starts=np.r_[0,np.bincount(u,minlength=480189).cumsum()]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    counts=np.diff(starts)
    weights=np.empty(len(u))
    for cnt in np.unique(counts):
        if not cnt:continue
        users=np.flatnonzero(counts==cnt)
        for lo in range(0,len(users),4096):
            block=users[lo:lo+4096]
            a=order[starts[block,None]+np.arange(cnt)]
            if cnt==1:
                weights[a[:,0]]=e[a[:,0]]/variance
            else:
                aa,bb=a[:,:,None],a[:,None,:]
                k=kernels(i[aa],d[aa],prior[aa],i[bb],d[bb],prior[bb],g)@coef
                diag=np.arange(cnt)
                # Use a positive independent noise term for every observed star.
                k[:,diag,diag]=np.maximum(variance,k[:,diag,diag]+.25)
                weights[a]=np.linalg.solve(k,e[a][...,None])[...,0]
    rp=ne.verified_rank(query)
    n=len(qu)
    signals=np.zeros((n,14),'f')
    history=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    for pos in range(int(counts.max())):
        ids=np.flatnonzero(counts[qu]>pos)
        for lo in range(0,len(ids),65536):
            j=ids[lo:lo+65536]
            a=order[starts[qu[j]]+pos]
            assert np.all(qi[j]!=i[a])
            k=kernels(qi[j],qd[j],rp[j],i[a],d[a],prior[a],g)
            # A separate coefficient for sessions with other observed ratings
            # lets the fit account for information already in the parent.
            available=history[j,13]>1.0
            weighted=k*weights[a,None]
            signals[j,:7]+=weighted
            signals[j,7:]+=weighted*available[:,None]
    return signals


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    p=np.load(PARENT/'known-selected.npy')
    br=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==br['qualifying_hashes']['selected']
    q=np.fromfile(PARENT/'selected.f32',dtype='<f4').astype(float)
    g=movie_geometry()
    # Cxx variance only scales the single-anchor feature and can therefore
    # be fixed for the development fit. Fit production covariance after selection.
    x=features('trainx','probex',p,g,np.zeros(7),1.)
    def solve(a,b):
        gram=a.T@a
        gram.flat[::len(gram)+1]+=500
        return np.linalg.solve(gram,a.T@b)
    x=x.astype(float)
    held=p.copy()
    for j in range(5):
        tr=f!=j
        held[~tr]=ne.clipped(p[~tr]+x[~tr]@solve(x[tr],(y-p)[tr]))
    w=solve(x,y-p)
    gains=[ne.rmse(p[f==j],y[f==j])-ne.rmse(held[f==j],y[f==j]) for j in range(5)]
    ok=sorted(gains)[1]>0 and ne.rmse(held,y)<ne.rmse(p,y)
    report=dict(selected='candidate' if ok else 'control',fold_gains=gains,known_rmse=ne.rmse(held,y),
                weights=w.tolist(),qualifying_hashes={},known_comparison_is_nested=False,qualifying_labels_used_for_fitting=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(OUT/'oof.npy',held-p)
    print(json.dumps(report),flush=True)
    if ok:
        u,i,d,prior=original()
        _,ix=anchor.training_indices('fulltrain',u,i,d)
        coef,var=estimate(u,i,d,prior,ne.history('fulltrain','ratings')[ix],g)
        z=features('fulltrain','qual',q,g,coef,var)
        delta=z@(w*var)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report.update(covariance=coef.tolist(),anchor_variance=var,qualifying_hashes={'selected':ne.sha(path)})
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
