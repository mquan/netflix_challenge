"""Transfer empirical movie-pair and genre-cluster corrections across nearby rating dates."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time

import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_history_context as hist
import own_session_pairs as pairs
import own_session_latent as latent
import own_anchor_context as anchor
from own_kernel_family import control

OUT=ne.ROOT/'own-temporal-pairs-20261126'


def clusters():
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    x=np.load(path)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    rng=np.random.default_rng(20261126)
    c=x[rng.choice(len(x),128,replace=False)].copy()
    for _ in range(15):
        labels=np.argmax(x@c.T,axis=1)
        centers=np.zeros_like(c)
        np.add.at(centers,labels,x)
        c=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    return labels.astype('i')


def features(train,query,p):
    lib=ctypes.CDLL(str((OUT/'pairs.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.temporal_pairs.argtypes=[it]*4+[pt]*17
    lib.temporal_pairs.restype=None
    lib.group_items.argtypes=[it,pt,pt,it,pt];lib.group_items.restype=None
    ei,qsession,es,off,items,ratings,er,ec,keys=pairs.make_events(train,query,with_keys=True)
    # User identity sits in the high bits. Never cross account boundaries.
    base=(keys//4096)*4096
    lower=np.searchsorted(keys,np.maximum(base,keys-120)).astype('i')
    upper=np.searchsorted(keys,np.minimum(base+4095,keys+120),side='right').astype('i')
    ti=ne.history(train,'item_idxs')
    ni=len(ne.history(train,'item_cnts'))
    starts,rows=latent.groups(lib,ti,ni)
    qi=ne.history(query,'item_idxs')
    qs,qo=latent.groups(lib,qi,ni)
    label=clusters()
    out=np.zeros((len(qi),72),'f')
    args=[a.ctypes.data for a in (starts,rows,ei,qs,qo,qsession,es,off,keys,lower,upper,items,ratings,er,ec,label,out)]
    start=time.monotonic()
    def block(j):lib.temporal_pairs(j,min(j+16,ni),ni,128,*args)
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(block,range(0,ni,16))):
            if j%100==0:print(query,j*16,round(time.monotonic()-start),flush=True)
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    x=np.column_stack((p,h[:,1:8],h[:,13:16],out)).astype('f')
    assert np.isfinite(x).all()
    np.save(OUT/(query+'.npy'),x)
    return x


def model():
    return CatBoostRegressor(iterations=1200,depth=5,learning_rate=.04,l2_leaf_reg=35,random_seed=20261126,
        thread_count=3,allow_writing_files=False,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_temporal_pairs.cpp','scripts/own_neighbor_features.cpp','-o',str(OUT/'pairs.dylib')],check=True)
    y,f,p,q=control()
    x=features('trainx','probex',p)
    selected,stats,held=anchor.crossfit(x,y,f,p,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,days=120,clusters=128,qualifying_hashes={},qualifying_labels_used_for_fitting=False,
                known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=features('fulltrain','qual',q)
        delta=model().fit(x,y-p).predict(qx,thread_count=3)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
