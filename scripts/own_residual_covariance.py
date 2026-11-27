"""Fit empirical movie error covariances and condition each terminal rating on its user's observed errors."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.cluster.vq import kmeans2
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as ac
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_transition as trans
import own_full_context as full
OUT=ne.ROOT/'own-residual-covariance-20261126'
PARENT=blend.OUT/'schema-03'


def groups():
    ic=ne.history('trainx','item_cnts');order=np.argsort(-ic.astype('i8'),kind='stable')
    geo=ac.movie_geometry()[order[2560:]]
    _,cluster=kmeans2(geo,256,iter=12,minit='++',seed=11246)
    mapping=np.empty(len(ic),'i');mapping[order[:2560]]=np.arange(2560);mapping[order[2560:]]=2560+cluster
    return mapping


def data(train,phase,base,lib,bins):
    err,_=trans.loo(train)
    i,d=[ne.history(train,k) for k in ('item_idxs','dates')]
    uc=ne.history(train,'user_cnts');ic=ne.history(train,'item_cnts');off=np.r_[0,uc.cumsum()].astype('i')
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    qr=np.argsort(qu,kind='stable').astype('i');qo=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()].astype('i')
    ou,oi,od,_=anchor.old_probe();u,m,day=[np.r_[a,ne.history('qual',k)] for a,k in zip((ou,oi,od),('user_idxs','item_idxs','dates'))]
    ix=np.argsort(u,kind='stable');co=np.r_[0,np.bincount(u,minlength=len(uc)).cumsum()].astype('i')
    ci=np.ascontiguousarray(m[ix],'i2');cd=np.ascontiguousarray(day[ix],'i2')
    dim=int(bins.max())+1;out=np.zeros((len(base),24),'f')
    ptr=lambda *args:[a.ctypes.data for a in args]
    for mode in range(2):
        gram=np.zeros((2*dim,2*dim),'f')
        lib.accumulate(len(uc),dim,mode,*ptr(off,i,d,err,bins,gram))
        def symmetric(a):return np.tril(a)+np.tril(a,-1).T
        count=symmetric(gram[:dim,:dim]);products=symmetric(gram[dim:,dim:]);cross=gram[dim:,:dim]
        mean=cross.diagonal()/count.diagonal().clip(1)
        den=count.clip(1)
        cov=((products-cross*cross.T/den)/(count+(100 if mode else 65))).clip(-.9,.9)
        variance=np.maximum(.3,products.diagonal()/count.diagonal().clip(1)-mean**2).astype('f')
        shift=(cross-mean[:,None]*count)/(count+(120 if mode else 80))
        cov=np.ascontiguousarray(cov);shift=np.ascontiguousarray(shift)
        del gram,count,products,cross,den
        args=ptr(off,i,d,err,bins,qo,qr,qi,qd,co,ci,cd,cov,shift,variance,out)
        def block(j):lib.predict(j,min(j+384,len(uc)),dim,mode,*args)
        with ThreadPoolExecutor(2) as pool:
            for j,_ in enumerate(pool.map(block,range(0,len(uc),384))):
                if j%400==0:print(train,'window',mode,'users',384*j,flush=True)
    assert np.isfinite(out).all()
    return np.column_stack((out,base,full.cf96(phase)-base,ne.verified_rank(phase)-base,np.log1p(uc[qu]),np.log1p(ic[qi]),qd/2000)).astype('f')


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-Wno-deprecated-declarations','-std=c++17','-shared','-fPIC','scripts/own_residual_covariance.cpp','-framework','Accelerate','-o',str(OUT/'cov.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'cov.dylib').resolve()));it,pt=ctypes.c_int,ctypes.c_void_p
    lib.accumulate.argtypes=[it]*3+[pt]*6;lib.accumulate.restype=None
    lib.predict.argtypes=[it]*4+[pt]*16;lib.predict.restype=None
    y,f,_,_=anchor.history_control();rec=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    bins=groups();x=data('trainx','probex',base,lib,bins)
    def model():return CatBoostRegressor(iterations=620,depth=6,learning_rate=.045,l2_leaf_reg=75,
            thread_count=2,random_seed=715,verbose=False,allow_writing_files=False)
    choice,stats,held=anchor.crossfit(x,y,f,base,model)
    rep=dict(parent=str(PARENT),selected=choice,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if choice!='control':
        qx=data('fulltrain','qual',qb,lib,bins);delta=model().fit(x,y-base).predict(qx,thread_count=2)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
