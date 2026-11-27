"""Train rating-specific asymmetric latent factors on chronological CFNADE forecast errors."""
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
from own_query_sets import projection
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-suffix-factors-20261126'


def run(train,phase,lib):
    d=suffix.Data(train,phase,None);s=cf.checkpoint(train)
    geo=np.ascontiguousarray(np.column_stack((projection(s['a'].mean(0),16),projection(s['p'].mean(0),16))))
    out=np.full((4,len(d.native)),np.nan,'f')
    args=(d.off,d.i,d.d,d.r,d.boundary,d.grid,d.fi,d.fd,d.obs,d.lookup,d.enc,d.eproj,d.eb,d.op,d.b,d.ub,d.ud,geo,out)
    lib.terminal_factors(d.nu,len(d.geo),len(d.native),d.sd,*[v.ctypes.data for v in args])
    assert np.isfinite(out).all();np.save(OUT/(phase+'.npy'),out.T)
    return out.T


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations','scripts/own_suffix_factors.cpp',
                    '-framework','Accelerate','-o',str(OUT/'factors.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'factors.dylib').resolve()));lib.terminal_factors.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*19;lib.terminal_factors.restype=None
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    def feats(v,b,t,phase):
        u,i=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        return np.column_stack((v,b,full.cf96(phase)-b,ne.verified_rank(phase)-b,np.log1p(ne.history(t,'user_cnts')[u]),np.log1p(ne.history(t,'item_cnts')[i]))).astype('f')
    def model():return CatBoostRegressor(iterations=650,depth=5,learning_rate=.045,l2_leaf_reg=70,thread_count=2,
                                          random_seed=527,verbose=False,allow_writing_files=False)
    v=run('trainx','probex',lib);x=feats(v,base,'trainx','probex');_,_,td=anchor.crossfit(x,y,f,base,model)
    z=np.column_stack((v,td));held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-base)[tr])
    w=fit(z,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv=run('fulltrain','qual',lib);t=model().fit(x,y-base).predict(feats(qv,qb,'fulltrain','qual'),thread_count=2)
        delta=np.column_stack((qv,t))@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
