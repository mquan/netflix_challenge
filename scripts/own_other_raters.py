"""Predict a movie from its other raters with similar leave-movie-out history profiles."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_session_latent as latent
import own_full_context as full
import own_anchor_context as anchor
import own_history_context as hist
from own_kernel_family import control

OUT=ne.ROOT/'own-other-raters-20261126'


def geometry(train,ic):
    f=ne.verified_rank(train,True)
    _,v=np.linalg.eigh(f.T@f)
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    g=np.load(path)
    _,gv=np.linalg.eigh(g.T@g)
    coords=np.column_stack((f@v[:,-12:],g@gv[:,-12:]))
    mean=np.average(coords,weights=ic,axis=0)
    std=np.sqrt(np.average((coords-mean)**2,weights=ic,axis=0))
    return np.ascontiguousarray((coords-mean)/np.maximum(std,1e-8),dtype='f')


def features(train,query,p):
    lib=ctypes.CDLL(str((OUT/'raters.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.profiles.argtypes=[it]+[pt]*6
    lib.other.argtypes=[it,it]+[pt]*16
    for fn in (lib.group_items,lib.profiles,lib.other): fn.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    mm=np.ascontiguousarray((np.bincount(i,weights=r,minlength=len(ic))+20*r.mean())/(ic+20),dtype='f')
    coords=geometry(train,ic)
    prof=np.zeros((len(uc),48),'f')
    def ptr(*args):return [a.ctypes.data for a in args]
    lib.profiles(len(uc),*ptr(off,i,r,mm,coords,prof))
    re,ce=r-ne.verified_rank(train),r-full.cf96(train)
    s,rows=latent.groups(lib,i,len(ic))
    qs,qo=latent.groups(lib,qi,len(ic))
    rng=np.random.default_rng(20261126)
    perm=np.array([rng.permutation(48)[:24] for _ in range(3)],dtype='i')
    out=np.zeros((len(p),24),'f')
    args=ptr(s,rows,qs,qo,u,d,r,re,ce,qu,qd,mm,coords,prof,perm,out)
    def block(j):lib.other(j,min(j+8,len(ic)),*args)
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(block,range(0,len(ic),8))):
            if j%250==0:print(query,j*8,flush=True)
    out[:,2::4]-=p[:,None]
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    x=np.column_stack((p,h[:,1:9],h[:,13:16],out)).astype('f')
    assert np.isfinite(x).all()
    return x


def model():
    return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=40,
                             random_seed=20261126,thread_count=2,allow_writing_files=False,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-ffast-math','-shared','-fPIC','scripts/own_other_raters.cpp',
                    'scripts/own_neighbor_features.cpp','-o',str(OUT/'raters.dylib')],check=True)
    y,f,p,q=control()
    x=features('trainx','probex',p)
    selected,stats,held=anchor.crossfit(x,y,f,p,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,
                known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=features('fulltrain','qual',q)
        delta=model().fit(x,y-p).predict(qx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
