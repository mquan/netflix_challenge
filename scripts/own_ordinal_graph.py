"""Fit a symmetric five-star interaction graph on chronological cohorts with all cohort stars removed from the CF prior."""
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
OUT=ne.ROOT/'own-ordinal-graph-20261126'
PARENT=blend.OUT/'schema-08'


def run(train,phase,lib):
    d=suffix.Data(train,phase,None);s=cf.checkpoint(train)
    geo=np.ascontiguousarray(np.column_stack((projection(s['a'].mean(0),16),projection(s['p'].mean(0),16))),'f')
    out=np.full((6,len(d.native)),np.nan,'f')
    args=(d.off,d.i,d.d,d.r,d.boundary,d.grid,d.fi,d.fd,d.obs,d.lookup,d.enc,d.eproj,d.eb,d.op,d.b,d.ub,d.ud,geo,out)
    lib.ordinal_graph(d.nu,len(d.geo),len(d.native),d.sd,*[v.ctypes.data for v in args])
    assert np.isfinite(out).all() and out.min()>=.999 and out.max()<=5.001
    np.save(OUT/(phase+'.npy'),out.T)
    return out.T


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations','scripts/own_ordinal_graph.cpp',
                    '-framework','Accelerate','-o',str(OUT/'graph.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'graph.dylib').resolve()));lib.ordinal_graph.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*19
    lib.ordinal_graph.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def features(v,b,phase):
        raw=full.cf96(phase)
        return np.column_stack((v-raw[:,None],raw-b,b,ne.verified_rank(phase)-b)).astype('f')
    def model():return CatBoostRegressor(iterations=450,depth=5,learning_rate=.045,l2_leaf_reg=80,thread_count=2,
                                          random_seed=48193,verbose=False,allow_writing_files=False)
    v=run('trainx','probex',lib);x=features(v,b,'probex');_,_,td=anchor.crossfit(x,y,f,b,model)
    z=np.column_stack((x[:,:6],td));held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-b)[tr])
    w=fit(z,y-b);ctrl=blend.metrics(b,y,f);known=blend.metrics(ne.clipped(b+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),
        qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv=run('fulltrain','qual',lib);qx=features(qv,qb,'qual')
        t=model().fit(x,y-b).predict(qx,thread_count=2);delta=np.column_stack((qx[:,:6],t))@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
