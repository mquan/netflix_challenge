"""Fit a zero-self-weight residual autoencoder with ordinal errors, dated history, and implicit movie choices."""
import ctypes
import json
import subprocess
import time
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as transition
from own_empirical_neighborhood import fit
OUT=ne.ROOT/'own-multichannel-ease-20261126'
PARENT=blend.OUT/'schema-08'
K=1536


def generate(train,phase,lib):
    err,_=transition.loo(train);start=time.monotonic()
    u,i,d=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')];nu=len(uc);nm=len(ic)
    unknown_u,unknown_i=qu,qi
    if phase=='probex':
        unknown_u=np.r_[qu,ne.history('qual','user_idxs')];unknown_i=np.r_[qi,ne.history('qual','item_idxs')]
    old_u,_,old_d,_=anchor.old_probe();cut=np.full(nu,30000,'i2')
    np.minimum.at(cut,old_u,old_d);np.minimum.at(cut,ne.history('qual','user_idxs'),ne.history('qual','dates'))
    # Sources are popularity-ranked landmarks. Every movie remains eligible as a target.
    movies=np.argsort(-ic,kind='stable')[:K];mapper=np.full(nm,-1,'i2');mapper[movies]=np.arange(K)
    scale=(1/np.sqrt(22+uc)).astype('f')
    off=np.r_[0,uc.cumsum()].astype('i')
    y=sparse.csr_matrix((err*scale[u],i.astype('i'),off),shape=(nu,nm))
    mask=mapper[i]>=0
    lu=u[mask];li=mapper[i[mask]].astype('i');w=scale[lu]
    xr=sparse.csr_matrix((err[mask]*w,(lu,li)),shape=(nu,K))
    g=(1/(1+np.abs(d[mask].astype('i')-cut[lu])/60)).astype('f')
    xt=sparse.csr_matrix((err[mask]*w*g,(lu,li)),shape=(nu,K))
    xi=sparse.csr_matrix((w,(lu,li)),shape=(nu,K));imp=mapper[unknown_i]>=0
    xi+=sparse.csr_matrix((scale[unknown_u[imp]],(unknown_u[imp],mapper[unknown_i[imp]])),shape=(nu,K))
    del mask,lu,li,w,g,err
    x=sparse.hstack((xr,xi,xt),format='csr');del xr,xt,xi
    # User blocks bound intermediate sparse matrix multiplications.
    gram=np.zeros((3*K,3*K),'f');rhs=np.zeros((3*K,nm),'f')
    for lo in range(0,nu,60000):
        sl=slice(lo,min(nu,lo+60000));t=x[sl].T.tocsr()
        gram+=(t@x[sl]).toarray();rhs+=(t@y[sl]).toarray()
        print(phase,'normal equations',sl.stop,round(time.monotonic()-start),flush=True)
    del y
    qo=np.argsort(qu,kind='stable').astype('i');qoff=np.r_[0,np.bincount(qu,minlength=nu).cumsum()].astype('i')
    outputs=[]
    for pen in (18.,95.,420.):
        a=np.array(gram,order='F');a.flat[::len(a)+1]+=pen
        fac=cho_factor(a,overwrite_a=True,check_finite=False)
        inv=cho_solve(fac,np.eye(len(a),dtype='f',order='F'),overwrite_b=True,check_finite=False)
        weights=cho_solve(fac,np.array(rhs,order='F'),overwrite_b=True,check_finite=False)
        # Constrain every coefficient belonging to the target movie to zero, across all three channels.
        for k,m in enumerate(movies):
            idx=np.array([k,k+K,k+2*K]);pinv=inv[:,idx]
            weights[:,m]-=pinv@np.linalg.solve(pinv[idx],weights[idx,m]);weights[idx,m]=0
        result=np.empty((len(qi),3),'f');params=(x.indptr,x.indices,x.data,qoff,qo,qi,np.ascontiguousarray(weights.T),scale,result)
        lib.ease_score(nu,K,*[v.ctypes.data for v in params]);assert np.isfinite(result).all()
        outputs.append(result)
    v=np.column_stack(outputs);np.save(OUT/(phase+'.npy'),v);return v


def main(paired=False):
    global OUT,PARENT
    original=OUT
    if paired:
        OUT=ne.ROOT/'own-paired-ease-20261126';PARENT=blend.OUT/'paired-01'
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_multichannel_ease.cpp',
                                             '-o',str(OUT/'score.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'score.dylib').resolve()));lib.ease_score.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*9;lib.ease_score.restype=None
    y,f,_,_=anchor.history_control();r=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==r['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def raw(train,phase):
        if not paired:return generate(train,phase,lib)
        ready=original/(phase+'.npy')
        while not (original/'freeze.json').is_file() and (phase!='probex' or not (original/'selection.json').is_file()):time.sleep(8)
        if ready.is_file():return np.load(ready)
        return generate(train,phase,lib)
    v=raw('trainx','probex')
    def context(z,b,phase):
        return np.column_stack((z,b,full.cf96(phase)-b,ne.verified_rank(phase)-b)).astype('f')
    def model():return CatBoostRegressor(iterations=450,depth=5,l2_leaf_reg=80,learning_rate=.04,thread_count=2,
                                          random_seed=72457,verbose=False,allow_writing_files=False)
    x=context(v,b,'probex');_,_,td=anchor.crossfit(x,y,f,b,model);z=np.column_stack((v,td));held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-b)[tr])
    w=fit(z,y-b);ctrl=blend.metrics(b,y,f);known=blend.metrics(ne.clipped(b+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv=raw('fulltrain','qual');tree=model().fit(x,y-b)
        delta=np.column_stack((qv,tree.predict(context(qv,qb,'qual'),thread_count=2)))@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--paired',action='store_true');main(ap.parse_args().paired)
