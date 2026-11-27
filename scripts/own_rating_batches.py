"""Learn how a movie's place among the titles rated in the same session changes its expected rating."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as trans
from own_query_sets import projection

OUT=ne.ROOT/'own-rating-batches-20261126'
PARENT=blend.OUT/'paired-03'


def features(train,phase,b,lib):
    err,_=trans.loo(train);uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    it,day,star=[ne.history(train,k) for k in ('item_idxs','dates','ratings')];u,i,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    iu,ii,dd=u,i,d
    if phase=='probex':iu,ii,dd=[np.r_[a,ne.history('qual',k)] for a,k in zip((u,i,d),('user_idxs','item_idxs','dates'))]
    def grouped(users):return np.r_[0,np.bincount(users,minlength=len(uc)).cumsum()].astype('i'),np.argsort(users,kind='stable').astype('i')
    io,ir=grouped(iu);ii=np.ascontiguousarray(ii[ir],'i2');dd=np.ascontiguousarray(dd[ir],'i2');qo,qr=grouped(u)
    off=np.r_[0,uc.cumsum()].astype('i')
    assert min(day.min(),dd.min())>=0 and max(day.max(),dd.max())<2300
    year=np.clip((ne.history(train,'item_years').astype('f')-1985)/15,-5,2);pop=((np.log1p(ic)-8)/2).astype('f')
    geom=projection(cf.checkpoint(train)['p'].mean(0),8);geom=np.ascontiguousarray(geom/np.linalg.norm(geom,axis=1,keepdims=True).clip(.001))
    out=np.empty((len(b),38),'f');args=(off,it,day,star,err,io,ii,dd,qo,qr,i,d,pop,year,geom,out)
    lib.rating_batches(len(uc),len(ic),*[a.ctypes.data for a in args]);assert np.isfinite(out).all()
    return np.column_stack((out,b,full.cf96(phase)-b,ne.verified_rank(phase)-b)).astype('f')


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_rating_batches.cpp','-o',str(OUT/'batches.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'batches.dylib').resolve()));lib.rating_batches.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*16;lib.rating_batches.restype=None
    y,f,_,_=anchor.history_control();rec=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');x=features('trainx','probex',b,lib);held=np.empty(len(y),'f');models=[]
    for k in range(5):
        tr=f!=k;m=CatBoostRegressor(iterations=520,depth=5,l2_leaf_reg=80,learning_rate=.04,thread_count=2,random_seed=6737+k,allow_writing_files=False,verbose=False)
        m.fit(x[tr],(y-b)[tr]);held[~tr]=m.predict(x[~tr],thread_count=2);models.append(m);print('rating batch fold',k,flush=True)
    stats={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];choice=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx=features('fulltrain','qual',qb,lib);delta=sum(m.predict(qx,thread_count=2) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta);ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
