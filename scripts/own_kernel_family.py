"""Compare temporal kernels using factor, co-rating and training-residual movie similarities."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time

import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_history_context as hist
import own_neighbor_experiment as ne
import own_profile_regression as profile
import own_session_latent as latent
import own_overnight_blend as blend

OUT=ne.ROOT/'own-kernel-family-20261126'
GP=ne.ROOT/'own-user-gp-20261126'


def control():
    y,f,_,_=anchor.history_control()
    p=np.load(blend.OUT/'first/known-selected.npy')+np.load(GP/'oof.npy')
    record=json.loads((GP/'freeze.json').read_text())
    np.testing.assert_allclose(ne.rmse(p,y),record['known_rmse'],atol=1e-12,rtol=0)
    assert ne.sha(GP/'selected.f32')==record['qualifying_hashes']['selected']
    return y,f,p,np.fromfile(GP/'selected.f32',dtype='<f4').astype(float)


def gram(v):
    v=np.asarray(v,dtype='f')
    v=np.ascontiguousarray(v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-10))
    n=len(v)
    result=np.empty((n,n),dtype='i2')
    for lo in range(0,n,128):
        sl=slice(lo,lo+128)
        result[sl]=np.rint((v[sl]@v.T)*16000).astype('i2')
    return result


def generate(train,query):
    lib=ctypes.CDLL(str((OUT/'kernel.dylib').resolve()))
    lib.kernel_predict.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*17
    lib.kernel_predict.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    rp=ne.verified_rank(train);cp=full.cf96(train)
    re,ce=r-rp,r-cp
    trainp=(rp+cp)*.5
    queryp=(ne.verified_rank(query)+full.cf96(query))*.5
    residual_basis=profile.basis(i,off,ic,3.6+(re+ce)*.5,'explicit')
    ga=gram(ne.verified_rank(train,True))
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    gb=gram(np.load(path))
    gc=gram(residual_basis)
    user=np.ascontiguousarray((np.bincount(u,weights=r,minlength=len(uc))+10*float(r.mean()))/(uc+10),dtype='f')
    movie=np.zeros(len(ic))
    for a in range(0,len(i),1000000):
        sl=slice(a,a+1000000)
        movie+=np.bincount(i[sl],weights=r[sl]-user[u[sl]],minlength=len(ic))
    movie=np.ascontiguousarray(movie/(ic+30),dtype='f')
    output=np.zeros((len(qi),42),'f')
    args=[v.ctypes.data for v in (qu,qi,qd,off,i,d,r,re,ce,movie,user,ga,gb,gc,trainp,queryp,output)]
    def chunk(k):lib.kernel_predict(k,min(k+512,len(qi)),len(ic),*args)
    start=time.monotonic()
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(chunk,range(0,len(qi),512))):
            if j%500==0:print(query,j*512,round(time.monotonic()-start),flush=True)
    assert np.isfinite(output).all()
    np.save(OUT/(query+'.npy'),output)
    return output


def features(raw,query,p):
    x=raw.copy()
    x[:,2::7]-=p[:,None]
    old=np.load(GP/(query+'.npy')).copy()
    old[:,2::3]-=p[:,None]
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    return np.column_stack([x,old,p,h[:,1:9],h[:,13:16],h[:,37:40]]).astype('f')


def model():
    return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=40,random_seed=20261126,
                             thread_count=3,allow_writing_files=False,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC','scripts/own_kernel_family.cpp','-framework','Accelerate','-o',str(OUT/'kernel.dylib')],check=True)
    y,f,p,q=control()
    x=features(generate('trainx','probex'),'probex',p)
    selected,stats,held=anchor.crossfit(x,y,f,p,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,
                known_comparison_is_nested=False,baseline_sha256=ne.sha(GP/'selected.f32'))
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=features(generate('fulltrain','qual'),'qual',q)
        delta=model().fit(x,y-p).predict(qx,thread_count=3)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
