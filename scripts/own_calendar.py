"""Estimate seasonality and demand shocks for each film from dated Netflix ratings and unlabeled movie addresses."""
import ctypes
from datetime import date,timedelta
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as transition
from own_query_sets import projection
OUT=ne.ROOT/'own-calendar-20261126'
PARENT=blend.OUT/'paired-01'


def generate(train,phase,b,lib):
    err,_=transition.loo(train);uc=ne.history(train,'user_cnts');off=np.r_[0,uc.cumsum()].astype('i')
    i,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')];u,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    qoff=np.r_[0,np.bincount(u,minlength=len(uc)).cumsum()].astype('i');qr=np.argsort(u,kind='stable').astype('i')
    ii,idate=qi,qd
    if phase=='probex':ii,idate=[np.r_[a,ne.history('qual',k)] for a,k in ((qi,'item_idxs'),(qd,'dates'))]
    assert min(d.min(),idate.min())>=0 and max(d.max(),idate.max())<2250
    days=[date(1999,11,11)+timedelta(days=k) for k in range(2250)]
    season=np.array([(date(2000,t.month,t.day)-date(2000,1,1)).days for t in days],'i');week=np.array([t.weekday() for t in days],'i')
    g=cov.movie_geometry();raw=projection(cf.checkpoint(train)['p'].mean(0),32);raw/=np.linalg.norm(raw,axis=1,keepdims=True).clip(.001)
    emb=np.column_stack((g,raw));nb=np.empty((len(g),16),'i')
    for j in range(0,len(g),160):
        sl=slice(j,min(len(g),j+160));sim=emb[sl]@emb.T;sim[np.arange(len(sim)),np.arange(sl.start,sl.stop)]=-10
        nb[sl]=np.argpartition(-sim,15,axis=1)[:,:16]
    x=np.full((len(u),49),np.nan,'f');args=(off,i,d,r,err,qoff,qr,qi,qd)
    lib.calendar(len(uc),len(g),len(u),*[v.ctypes.data for v in args],len(ii),*[v.ctypes.data for v in (ii,idate,nb,season,week,x)])
    assert np.isfinite(x).all()
    return np.column_stack((x,b,full.cf96(phase)-b,ne.verified_rank(phase)-b,np.log1p(uc[u]))).astype('f')


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_calendar.cpp','-o',str(OUT/'calendar.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'calendar.dylib').resolve()));lib.calendar.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*9+[ctypes.c_int]+[ctypes.c_void_p]*6;lib.calendar.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');x=generate('trainx','probex',b,lib);held=np.empty(len(y),'f');models=[]
    for k in range(5):
        tr=f!=k;m=CatBoostRegressor(iterations=700,depth=6,learning_rate=.042,l2_leaf_reg=95,thread_count=2,
                                    random_seed=17257+k,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-b)[tr]);models.append(m);held[~tr]=m.predict(x[~tr],thread_count=2);print('calendar fold',k,flush=True)
    stats={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];key=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx=generate('fulltrain','qual',qb,lib);delta=sum(m.predict(qx,thread_count=2) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
