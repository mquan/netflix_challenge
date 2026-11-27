"""Learn four ordinal corrections to CF96 with explicit user factors and chronological asymmetric history factors."""
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
import own_suffix_cf as suffix
from own_empirical_neighborhood import fit
from own_query_sets import projection
OUT=ne.ROOT/'own-ordinal-factors-20261126'
PARENT=blend.OUT/'schema-04'


def run(train,phase,lib):
    d=suffix.Data(train,phase,None);s=cf.checkpoint(train)
    geo=np.ascontiguousarray(np.column_stack((projection(s['a'].mean(0),16),projection(s['p'].mean(0),16))))
    out=np.full((8,len(d.native)),np.nan,'f')
    prob=np.ascontiguousarray(np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')[:,7:12])
    np.testing.assert_allclose(prob@np.arange(1,6,dtype='f'),d.native,atol=.0004,rtol=0)
    args=(d.off,d.i,d.d,d.r,d.boundary,d.grid,d.fi,d.fd,d.obs,d.lookup,d.enc,d.eproj,d.eb,d.op,d.b,d.ub,d.ud,geo,prob,out)
    lib.train_ordinal_factors(d.nu,len(d.geo),len(d.native),d.sd,*[v.ctypes.data for v in args])
    assert np.isfinite(out).all();np.save(OUT/(phase+'.npy'),out.T)
    return out.T


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations','scripts/own_ordinal_factors.cpp','-framework','Accelerate','-o',str(OUT/'ordinal.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'ordinal.dylib').resolve()));lib.train_ordinal_factors.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*20;lib.train_ordinal_factors.restype=None
    y,f,_,_=anchor.history_control();rec=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def features(v,b,phase,train):
        u,m=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        return np.column_stack((v,b,full.cf96(phase)-b,ne.verified_rank(phase)-b,np.log1p(ne.history(train,'user_cnts')[u]),np.log1p(ne.history(train,'item_cnts')[m]))).astype('f')
    def model():return CatBoostRegressor(iterations=550,depth=5,learning_rate=.045,l2_leaf_reg=70,thread_count=2,random_seed=7287,verbose=False,allow_writing_files=False)
    v=run('trainx','probex',lib);x=features(v,base,'probex','trainx');_,_,td=anchor.crossfit(x,y,f,base,model)
    z=np.column_stack((v[:,::2],td));held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-base)[tr])
    w=fit(z,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    report=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),qualifying_hashes={},
                chronological_windows_per_user=7,thresholds=4,qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        v=run('fulltrain','qual',lib);t=model().fit(x,y-base).predict(features(v,qb,'qual','fulltrain'),thread_count=2)
        delta=np.column_stack((v[:,::2],t))@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');report['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
