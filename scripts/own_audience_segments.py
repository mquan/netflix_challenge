"""Fit movie-specific calibration curves for audiences with similar unlabeled viewing histories."""
import ctypes
import json
import subprocess
import numpy as np
from scipy.cluster.vq import kmeans2
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_session_latent as latent
import own_transition as transition
from own_query_sets import projection
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-audience-segments-20261126'
PARENT=blend.OUT/'paired-03'


def segments(lib):
    uc,ic=[ne.history('trainx',k) for k in ('user_cnts','item_cnts')]
    geo=np.ascontiguousarray(projection(cf.checkpoint('trainx')['p'].mean(0),12))
    x=np.zeros((len(uc),24),'f');year=np.ascontiguousarray(ne.history('trainx','item_years'),'i2');pop=(np.log1p(ic)/13).astype('f')
    for phase in ('trainx','probex','qual'):
        u,i,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        lib.collect(len(i),*[v.ctypes.data for v in (u,i,d,geo,year,pop,x)])
    count=x[:,-1].copy();x/=count.clip(1)[:,None];x[:,-1]=np.log1p(count)
    x[:,13]-=x[:,12]**2;x[:,15]-=x[:,14]**2;x[:,17]-=x[:,16]**2
    x[:,21]-=x[:,12]*x[:,14];x[:,22]-=x[:,12]*x[:,16]
    x-=x.mean(0);x/=x.std(0).clip(.0001);np.clip(x,-5,5,out=x)
    clu=np.empty((len(x),3,2),'i1');mix=np.empty(clu.shape,'f');sample=np.random.default_rng(8188).choice(len(x),50000,replace=False)
    for h,a in enumerate((x[:,:12],x[:,12:],x)):
        a=np.ascontiguousarray(a);cent,_=kmeans2(a[sample],64,iter=18,minit='++',seed=8317+h)
        for j in range(0,len(a),8192):
            sl=slice(j,j+8192);diff=(cent**2).sum(1)[None,:]-2*a[sl]@cent.T
            ix=np.argsort(diff,axis=1)[:,:2];dist=np.take_along_axis(diff,ix,axis=1);w=np.exp(-.60*(dist-dist[:,0,None]))
            clu[sl,h]=ix;mix[sl,h]=w/w.sum(1,keepdims=True)
    return clu,mix


def features(train,phase,b,lib,segment):
    e,_=transition.loo(train);ic=ne.history(train,'item_cnts');u,i,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','ratings')]
    qu,qi=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
    mo,mr=latent.groups(lib,i,len(ic));qo,qr=latent.groups(lib,qi,len(ic))
    pred=np.ascontiguousarray(full.cf96(phase),'f');x=np.empty((len(b),24),'f');args=(mo,mr,u,r,e,qo,qr,qu,pred,*segment,x)
    lib.audience_cells(len(ic),*[v.ctypes.data for v in args]);assert np.isfinite(x).all()
    return np.column_stack((x,b,pred-b,ne.verified_rank(phase)-b)).astype('f')


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_audience_segments.cpp','scripts/own_neighbor_features.cpp','-o',str(OUT/'audience.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'audience.dylib').resolve()));pt=ctypes.c_void_p;it=ctypes.c_int
    lib.collect.argtypes=[it]+[pt]*7;lib.audience_cells.argtypes=[it]+[pt]*12;lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.collect.restype=lib.audience_cells.restype=lib.group_items.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');segment=segments(lib);x=features('trainx','probex',b,lib,segment)
    held=np.empty(len(y));trees=[];td=np.empty(len(y),'f')
    for k in range(5):
        tr=f!=k;m=CatBoostRegressor(iterations=450,depth=5,l2_leaf_reg=110,learning_rate=.036,random_seed=8315+k,thread_count=2,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-b)[tr]);td[~tr]=m.predict(x[~tr],thread_count=2);trees.append(m);print('audience calibration fold',k,flush=True)
    z=np.column_stack((x[:,0:24:4],td))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-b)[tr])
    w=fit(z,y-b);known=blend.metrics(ne.clipped(b+held),y,f);control=blend.metrics(b,y,f);ok=ret.qualifies(known,control)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=control,weights=w.tolist(),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del x
        qx=features('fulltrain','qual',qb,lib,segment);qd=sum(m.predict(qx,thread_count=2) for m in trees)/5
        delta=np.column_stack((qx[:,0:24:4],qd))@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
