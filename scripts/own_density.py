"""Estimate each user's rating and error distributions from class-conditional movie preferences with shrunk covariance."""
import ctypes
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
OUT=ne.ROOT/'own-density-20261126'
PARENT=blend.OUT/'paired-03'


def generate(train,phase,b,lib):
    err,_=transition.loo(train);uc=ne.history(train,'user_cnts');ic=ne.history(train,'item_cnts');off=np.r_[0,uc.cumsum()].astype('i')
    i,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    u,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    qo=np.r_[0,np.bincount(u,minlength=len(uc)).cumsum()].astype('i');qr=np.argsort(u,kind='stable').astype('i')
    s=cf.checkpoint(train);geos=(s['p'].mean(0),cov.movie_geometry())
    out=[];wt=(ic+20)/(ic+20).sum()
    for g in geos:
        v=g-np.average(g,axis=0,weights=wt);val,vec=np.linalg.eigh((v.T*wt)@v)
        x=np.ascontiguousarray((v@vec[:,-14:]/np.sqrt(val[-14:].clip(.0000001))).clip(-6,6),'f')
        pr=np.zeros((len(qi),20),'f');args=(off,i,d,r,err,qo,qr,qi,qd,x,pr)
        lib.density_scores(len(uc),*[a.ctypes.data for a in args]);assert np.isfinite(pr).all()
        np.testing.assert_allclose(pr.reshape(-1,5).sum(1),1,atol=3e-6);out.append(pr)
    return np.column_stack((*out,b,full.cf96(phase)-b,ne.verified_rank(phase)-b,np.log1p(uc[u]),np.log1p(ic[qi]))).astype('f')


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_density.cpp','-o',str(OUT/'density.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'density.dylib').resolve()));lib.density_scores.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*11;lib.density_scores.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');x=generate('trainx','probex',b,lib);held=np.empty(len(y),'f');models=[]
    for k in range(5):
        tr=f!=k;m=CatBoostRegressor(iterations=680,depth=6,learning_rate=.04,l2_leaf_reg=90,thread_count=2,
                                    random_seed=31673+k,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-b)[tr]);held[~tr]=m.predict(x[~tr],thread_count=2);models.append(m);print('density fold',k,flush=True)
    scores={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in scores if ret.qualifies(scores[k],scores['0'])];key=min(good,key=lambda k:scores[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=scores,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx=generate('fulltrain','qual',qb,lib);delta=sum(m.predict(qx,thread_count=2) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
