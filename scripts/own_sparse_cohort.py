"""Fit sparse signed movie neighborhoods and clustered movie-pair effects on chronological CF96 errors."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
from scipy.cluster.vq import kmeans2
import own_anchor_covariance as ac
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_suffix_cf as suffix
from own_query_sets import projection
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-sparse-cohort-20261126'


def run(train,phase,lib,joint=False):
    d=suffix.Data(train,phase,None);s=cf.checkpoint(train)
    def unit(x):return x/np.linalg.norm(x,axis=1,keepdims=True).clip(.0001)
    geo=unit(np.column_stack((unit(projection(s['p'].mean(0),32)),unit(projection(s['a'].mean(0),32)),ac.movie_geometry())))
    _,group=kmeans2(geo,128,minit='++',iter=12,seed=18165);group=np.ascontiguousarray(group,'i')
    top=np.empty((len(geo),128),'i')
    for j in range(0,len(geo),192):
        sl=slice(j,min(j+192,len(geo)));scores=geo[sl]@geo.T
        scores[np.arange(len(scores)),np.arange(j,j+len(scores))]=-10
        top[sl]=np.argpartition(-scores,127,axis=1)[:,:128]
    out=np.full((4,len(d.native)),np.nan,'f')
    args=(d.off,d.i,d.d,d.r,d.boundary,d.grid,d.fi,d.fd,d.obs,d.lookup,d.enc,d.eproj,d.eb,d.op,d.b,d.ub,d.ud,top,group,out)
    if joint:
        geometry=np.ascontiguousarray(np.column_stack((projection(s['a'].mean(0),16),projection(s['p'].mean(0),16))),'f')
        args=args[:-1]+(geometry,out)
    fn=lib.joint_cohort if joint else lib.sparse_cohort
    fn(d.nu,len(d.geo),len(d.native),d.sd,*[v.ctypes.data for v in args])
    assert np.isfinite(out).all();np.save(OUT/(phase+'.npy'),out.T)
    return out.T


def main(joint=False):
    global OUT
    if joint:OUT=ne.ROOT/'own-joint-neighborhood-20261126'
    source='scripts/own_joint_neighborhood.cpp' if joint else 'scripts/own_sparse_cohort.cpp'
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations',source,
                    '-framework','Accelerate','-o',str(OUT/'sparse.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'sparse.dylib').resolve()));fn=lib.joint_cohort if joint else lib.sparse_cohort
    fn.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*(21 if joint else 20);fn.restype=None
    y,f,_,_=anchor.history_control();parent=blend.OUT/('schema-08' if joint else 'schema-07')
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    def feats(v,b,t,phase):
        u,i=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        return np.column_stack((v,b,full.cf96(phase)-b,ne.verified_rank(phase)-b,np.log1p(ne.history(t,'user_cnts')[u]),np.log1p(ne.history(t,'item_cnts')[i]))).astype('f')
    def model():return CatBoostRegressor(iterations=650,depth=5,learning_rate=.045,l2_leaf_reg=70,thread_count=2,
                                          random_seed=527,verbose=False,allow_writing_files=False)
    v=run('trainx','probex',lib,joint);x=feats(v,base,'trainx','probex');_,_,td=anchor.crossfit(x,y,f,base,model)
    z=np.column_stack((v,td));held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-base)[tr])
    w=fit(z,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv=run('fulltrain','qual',lib,joint);t=model().fit(x,y-base).predict(feats(qv,qb,'fulltrain','qual'),thread_count=2)
        delta=np.column_stack((qv,t))@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--joint',action='store_true');main(parser.parse_args().joint)
