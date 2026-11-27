"""Learn signed movie/rating embeddings from within-session co-ratings and decode user histories."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_full_context as full
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_history_context as hist

OUT=ne.ROOT/'own-session-spectrum-20261126'
D=24


def encode(v,s,prob):
    mean=(v[:,1:,:]*np.sqrt(prob)[:,:,None]).sum(1)
    z=(.6*v[:,:1,:]+v[:,1:,:]/np.sqrt(prob)[:,:,None]-mean[:,None,:])*s[:,None,None]
    return np.ascontiguousarray(z,dtype='f')


def basis(lib,off,order,i,d,cat,nitems):
    # Center each rating channel against the in-partition histogram for that film.
    counts=np.zeros((nitems,5))
    for j in range(0,len(i),1_000_000):
        sl=slice(j,j+1_000_000)
        ids=5*i[sl].astype('i')+cat[sl]
        counts+=np.bincount(ids,minlength=nitems*5).reshape(-1,5)
    prob=((counts+4)/(counts.sum(1)[:,None]+20)).astype('f')
    scale=(counts.sum(1)+50).astype('f')**-.37
    rng=np.random.default_rng(20261126)
    v=rng.normal(size=(nitems,6,D)).astype('f')
    split=(len(off)-1)//2
    for step in range(5):
        z=encode(v,scale,prob)
        out=[np.zeros(z.shape,'f') for _ in range(2)]
        def compute(j):
            lo,hi=((0,split),(split,len(off)-1))[j]
            lib.sweep(lo,hi,*[a.ctypes.data for a in (off,order,i,d,cat,z,out[j])])
        with ThreadPoolExecutor(2) as ex:list(ex.map(compute,range(2)))
        total=out[0]+out[1]
        summed=total.sum(1,keepdims=True)
        v=np.concatenate((.6*summed,(total-prob[:,:,None]*summed)/np.sqrt(prob)[:,:,None]),axis=1)*scale[:,None,None]
        v=np.linalg.qr(v.reshape(-1,D))[0].reshape(nitems,6,D)
        print('spectral iteration',step+1,flush=True)
    z=encode(v,np.ones(nitems,'f'),prob)
    z/=np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-7)
    return z


def features(train,phase,base):
    lib=ctypes.CDLL(str((OUT/'spectrum.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.chronological.argtypes=[it]+[pt]*3
    lib.sweep.argtypes=[it,it]+[pt]*7
    lib.decode.argtypes=[it,it]+[pt]*11
    for fn in (lib.chronological,lib.sweep,lib.decode):fn.restype=None
    i,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    qo=np.argsort(qu,kind='stable').astype('i')
    qs=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()].astype('i')
    order=np.empty(len(i),'i')
    lib.chronological(len(uc),*[a.ctypes.data for a in (off,d,order)])
    chunks=[]
    for mode in ('observed','residual'):
        if mode=='observed':cat=(r-1).astype('i1')
        else:
            cat=np.empty(len(r),'i1')
            a,b=ne.verified_rank(train),full.cf96(train)
            for j in range(0,len(r),1_000_000):
                sl=slice(j,j+1_000_000)
                cat[sl]=np.searchsorted([-.75,-.2,.2,.75],r[sl]-.5*(a[sl]+b[sl]))
        z=basis(lib,off,order,i,d,cat,len(ic))
        result=np.zeros((len(qi),28),'f')
        lib.decode(0,len(uc),*[a.ctypes.data for a in (off,order,i,d,cat,qs,qo,qi,qd,z,result)])
        chunks.append(result)
    h=np.load(hist.OUT/(phase+'.npy'),mmap_mode='r')
    x=np.column_stack((base,h[:,1:9],h[:,13:16],*chunks)).astype('f')
    assert np.isfinite(x).all()
    return x


def model():
    return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=45,
                             thread_count=2,allow_writing_files=False,random_seed=20261126,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-ffast-math','-std=c++17','-shared','-fPIC','scripts/own_session_spectrum.cpp',
                    '-o',str(OUT/'spectrum.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base)
    selected,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,parent=str(parent),qualifying_hashes={},
                qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        xx=features('fulltrain','qual',qb)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
