"""Learn conditional rating errors by where a film and its viewer are in their observed Netflix histories."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as transition
OUT=ne.ROOT/'own-film-lifecycle-20261126'
PARENT=blend.OUT/'paired-01'


def get(train,phase,b,lib):
    err,_=transition.loo(train)
    i,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc=ne.history(train,'user_cnts');n=len(uc);off=np.r_[0,uc.cumsum()].astype('i')
    def groups(u):return np.r_[0,np.bincount(u,minlength=n).cumsum()].astype('i'),np.argsort(u,kind='stable').astype('i')
    qoff,qr=groups(qu);iu,ii,idate=qu,qi,qd
    if phase=='probex':iu,ii,idate=[np.r_[a,ne.history('qual',k)] for a,k in zip((qu,qi,qd),('user_idxs','item_idxs','dates'))]
    ioff,order=groups(iu);ii=np.ascontiguousarray(ii[order],'i2');idate=np.ascontiguousarray(idate[order],'i2')
    assert d.min()>=0 and d.max()<2300 and idate.min()>=0 and idate.max()<2300
    raw=np.ascontiguousarray(full.cf96(phase),'f');year=np.ascontiguousarray(ne.history(train,'item_years'),'i2')
    x=np.empty((len(qi),48),'f');args=(off,i,d,r,err,year,qoff,qr,qi,qd,raw,ioff,ii,idate,x)
    lib.lifecycle(n,len(year),len(qi),*[v.ctypes.data for v in args])
    assert np.isfinite(x).all()
    return np.column_stack((x,b,raw-b,ne.verified_rank(phase)-b,year[qi],np.log1p(uc[qu]))).astype('f')


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_film_lifecycle.cpp','-o',str(OUT/'lifecycle.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'lifecycle.dylib').resolve()));lib.lifecycle.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*15;lib.lifecycle.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');x=get('trainx','probex',b,lib);held=np.empty(len(y),'f');models=[]
    for k in range(5):
        tr=f!=k;m=CatBoostRegressor(iterations=700,depth=6,learning_rate=.04,l2_leaf_reg=90,thread_count=2,
                                    random_seed=61831+k,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-b)[tr]);models.append(m);held[~tr]=m.predict(x[~tr],thread_count=2);print('lifecycle fold',k,flush=True)
    known={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in known if ret.qualifies(known[k],known['0'])];choice=min(good,key=lambda k:known[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=known,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx=get('fulltrain','qual',qb,lib);delta=sum(m.predict(qx,thread_count=2) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
