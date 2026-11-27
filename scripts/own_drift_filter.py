"""Learn forward/backward user preference corrections over three drift timescales."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_full_context as full
import own_anchor_context as anchor
import own_history_context as hist
import own_other_raters as other
from own_kernel_family import control

OUT=ne.ROOT/'own-drift-filter-20261126'


def features(train,query,p):
    lib=ctypes.CDLL(str((OUT/'drift.dylib').resolve()))
    lib.drift.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*13
    lib.drift.restype=None
    item,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    qs=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()].astype('i')
    qo=np.argsort(qu,kind='stable').astype('i')
    coord=other.geometry(train,ic)
    x=np.column_stack((np.ones(len(ic)),coord[:,4:12],coord[:,17:24])).astype('f')
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    mm=np.ascontiguousarray((np.bincount(item,weights=r,minlength=len(ic))+20*r.mean())/(ic+20),dtype='f')
    re,ce=r-ne.verified_rank(train),r-full.cf96(train)
    out=np.zeros((len(p),24),'f')
    args=[v.ctypes.data for v in (off,qs,qo,item,d,r,re,ce,mm,qi,qd,x,out)]
    def block(j):lib.drift(j,min(j+512,len(uc)),*args)
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(block,range(0,len(uc),512))):
            if j%150==0:print(query,j*512,flush=True)
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    x=np.column_stack((p,mm[qi]-p,h[:,1:8],h[:,13:16],out)).astype('f')
    assert np.isfinite(x).all()
    return x


def model():
    return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=40,
                             random_seed=20261126,thread_count=2,allow_writing_files=False,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-ffast-math','-fPIC','-shared','scripts/own_drift_filter.cpp',
                    '-o',str(OUT/'drift.dylib')],check=True)
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
