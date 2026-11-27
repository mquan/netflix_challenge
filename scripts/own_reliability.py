"""Estimate each user's and each film's conditional star frequencies from target-removed CF96 training scores."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_transition as transitions
import own_retained_history as ret
OUT=ne.ROOT/'own-reliability-20261126'
PARENT=blend.OUT/'schema-08'


def get(train,phase,b,lib):
    err,_=transitions.loo(train)
    nu,nm=[len(ne.history(train,k)) for k in ('user_cnts','item_cnts')]
    items,days,stars=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    u,i,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    def grouped(index,count):
        return np.r_[0,np.bincount(index,minlength=count).cumsum()].astype('i'),np.argsort(index,kind='stable').astype('i')
    off=np.r_[0,ne.history(train,'user_cnts').cumsum()].astype('i');qs,qr=grouped(u,nu);mf,mr=grouped(items,nm);qs2,qr2=grouped(i,nm)
    raw=np.ascontiguousarray(full.cf96(phase),'f');features=np.empty((len(b),64),'f')
    args=(off,items,days,stars,err,qs,qr,i,d,raw,mf,mr,qs2,qr2,features)
    lib.reliability(nu,nm,*[a.ctypes.data for a in args]);assert np.isfinite(features).all()
    return np.column_stack((features,b,raw-b,ne.verified_rank(phase)-b,b==1.05,b==4.95)).astype('f')


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_reliability.cpp','-o',str(OUT/'reliability.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'reliability.dylib').resolve()));lib.reliability.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*15;lib.reliability.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def model():return CatBoostRegressor(iterations=750,depth=6,learning_rate=.038,l2_leaf_reg=90,thread_count=2,
                                          random_seed=271813,verbose=False,allow_writing_files=False)
    x=get('trainx','probex',b,lib);held=np.empty(len(b),'f');models=[]
    for fold in range(5):
        tr=f!=fold;fit=model().fit(x[tr],(y-b)[tr]);held[~tr]=fit.predict(x[~tr],thread_count=2);models.append(fit)
        print('reliability fold',fold,flush=True)
    known={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in known if ret.qualifies(known[k],known['0'])];key=min(good,key=lambda k:known[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=known,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        q=get('fulltrain','qual',qb,lib);delta=sum(m.predict(q,thread_count=2) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
