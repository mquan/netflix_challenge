"""Fit film-specific franchise regressions with both target and context ratings explicitly removed for the context forecast."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_slate as slate
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as transition
from own_empirical_neighborhood import fit
OUT=ne.ROOT/'own-series-ridge-20261126'
PARENT=blend.OUT/'paired-03'


def generate(train,phase,lib):
    err,_=transition.loo(train);d=slate.Data(train,phase)
    original=ne.ROOT/'own-paired-title-neighbors-20261126';path=original/'catalog.npz'
    assert ne.sha(path)==json.loads((original/'catalog.json').read_text())['derived_sha256']
    inferred=ne.ROOT/'own-learned-series-20261126'/(phase+'-neighbors.npz');emp=np.load(inferred)
    cat=np.load(path);s=cat['neighbors'];w=cat['affinity'];g=cat['bands'];est=emp['neighbors'];v=emp['weight']
    nb=np.full((len(s),20),-1,'i');weights=np.zeros(nb.shape,'f')
    for m in range(len(s)):
        ids=np.flatnonzero(s[m]>=0);ids=ids[np.argsort(g[m,ids]-w[m,ids]*.5,kind='stable')[:12]]
        nb[m,:len(ids)]=s[m,ids];weights[m,:len(ids)]=np.sqrt(w[m,ids])
        keep=np.flatnonzero(est[m]>=0)[:8]
        nb[m,12:12+len(keep)]=est[m,keep];weights[m,12:12+len(keep)]=v[m,keep]
    qi,qd,u=[ne.history(phase,k) for k in ('item_idxs','dates','user_idxs')]
    uc=d.uc;off=np.r_[0,uc.cumsum()].astype('i');qo=np.r_[0,np.bincount(u,minlength=len(uc)).cumsum()].astype('i');qr=np.argsort(u,kind='stable').astype('i')
    result=np.zeros((len(qi),5),'f');args=(off,d.i,d.d,d.r,err,qo,qr,qi,qd,nb,weights,d.md,d.total,d.ep,d.op,d.eb,d.b,d.ub,d.drift,result)
    assert all(x.flags.c_contiguous for x in args)
    lib.series_ridge(len(uc),len(nb),len(qi),d.sd,*[x.ctypes.data for x in args])
    assert np.isfinite(result).all();np.save(OUT/(phase+'.npy'),result)
    return result,dict(catalog_sha256=ne.sha(path),inferred_graph_sha256=ne.sha(inferred))


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations','scripts/own_series_ridge.cpp',
                                             '-framework','Accelerate','-o',str(OUT/'series.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'series.dylib').resolve()));lib.series_ridge.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*20;lib.series_ridge.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');v,inputs=generate('trainx','probex',lib)
    def features(v,b,phase):return np.column_stack((v,b,full.cf96(phase)-b,ne.verified_rank(phase)-b)).astype('f')
    mask=v[:,4]>0;x=features(v,b,'probex')[mask]
    td=np.zeros(len(y),'f');models=[]
    for k in range(5):
        tr=f[mask]!=k;m=CatBoostRegressor(iterations=380,depth=4,l2_leaf_reg=80,learning_rate=.04,thread_count=2,random_seed=3821+k,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-b)[mask][tr]);models.append(m);td[mask&(f==k)]=m.predict(x[~tr],thread_count=2)
    z=np.column_stack((v[:,:3],td));held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-b)[tr])
    w=fit(z,y-b);ctrl=blend.metrics(b,y,f);known=blend.metrics(ne.clipped(b+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),inputs={'probex':inputs},
             support_probex=int(mask.sum()),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv,inputs=generate('fulltrain','qual',lib);rep['inputs']['qual']=inputs;okq=qv[:,4]>0
        qx=features(qv,qb,'qual')[okq];qtd=np.zeros(len(qb),'f');qtd[okq]=sum(m.predict(qx,thread_count=2) for m in models)/5
        delta=np.column_stack((qv[:,:3],qtd))@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
