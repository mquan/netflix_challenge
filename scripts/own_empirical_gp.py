"""Fit empirical movie-pair covariance on all training users and condition on each query user's history."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_full_context as full
import own_session_latent as latent
import own_overnight_blend as blend

OUT=ne.ROOT/'own-empirical-gp-20261126'


def features(train,phase,base):
    lib=ctypes.CDLL(str((OUT/'gp.dylib').resolve()));pt=ctypes.c_void_p;it=ctypes.c_int
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.learn_pairs.argtypes=[it]*3+[pt]*11
    lib.infer_pairs.argtypes=[it]*3+[pt]*13
    for fn in (lib.group_items,lib.learn_pairs,lib.infer_pairs):fn.restype=None
    u,m,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i');ni=len(ic)
    starts,rows=latent.groups(lib,m,ni)
    usr=(np.bincount(u,weights=r,minlength=len(uc))+15*r.mean())/(uc+15)
    film=(np.bincount(m,weights=r,minlength=ni)+20*r.mean())/(ic+20)
    re=r-ne.verified_rank(train);ce=r-full.cf96(train)
    raw=np.empty(len(r),'f')
    for j in range(0,len(r),500000):
        sl=slice(j,j+500000);raw[sl]=r[sl]-usr[u[sl]]-film[m[sl]]+3.6
    pools=[np.empty((ni,ni),'i2') for _ in range(3)]
    err=(re+ce)*.5
    def ptrs(*a):return [v.ctypes.data for v in a]
    args=ptrs(starts,rows,u,off,m,d,err,raw,*pools)
    def learn(lo):lib.learn_pairs(lo,min(lo+24,ni),ni,*args)
    with ThreadPoolExecutor(2) as ex:
        for k,_ in enumerate(ex.map(learn,range(0,ni,24))):
            if k%100==0:print(phase,'learn',k*24,flush=True)
    del err,rows
    for matrix in pools:
        probe=matrix[::43,::43].astype('i')
        assert np.abs(probe-probe.T).max()<20
    z=np.zeros((len(qi),42),'f')
    args=ptrs(qu,qi,qd,off,m,d,re,ce,raw,*pools,z)
    def infer(lo):lib.infer_pairs(lo,min(lo+512,len(qi)),ni,*args)
    with ThreadPoolExecutor(2) as ex:
        for k,_ in enumerate(ex.map(infer,range(0,len(qi),512))):
            if k%600==0:print(phase,'infer',k*512,flush=True)
    assert np.isfinite(z).all()
    return np.column_stack((base,ne.verified_rank(phase)-base,full.cf96(phase)-base,
                 usr[qu],film[qi],np.log1p(uc[qu]),np.log1p(ic[qi]),z)).astype('f')


def model():
    return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=50,thread_count=2,
        random_seed=1145,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC',
        'scripts/own_empirical_gp.cpp','scripts/own_neighbor_features.cpp','-framework','Accelerate','-o',str(OUT/'gp.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base)
    key,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,
             known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
