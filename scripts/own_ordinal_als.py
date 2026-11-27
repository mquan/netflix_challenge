"""Jointly factor four cumulative rating thresholds and two own-model errors with user time drift."""
import ctypes
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
import own_session_latent as latent
from own_empirical_neighborhood import fit
from own_query_sets import projection

OUT=ne.ROOT/'own-ordinal-als-20261126'


def run(train,phase):
    lib=ctypes.CDLL(str((OUT/'als.dylib').resolve()));it=ctypes.c_int;pt=ctypes.c_void_p
    for name,nargs in [('ord_users',12),('ord_films',12),('ord_prior',5)]:
        fn=getattr(lib,name);fn.argtypes=[it,it]+[pt]*nargs;fn.restype=None
    lib.ord_predict.argtypes=[it]+[pt]*9;lib.ord_predict.restype=None
    lib.group_items.argtypes=[it,pt,pt,it,pt];lib.group_items.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i');starts,order=latent.groups(lib,i,len(ic))
    rank=ne.verified_rank(train);nade=full.cf96(train)
    md=np.ascontiguousarray(np.bincount(u,weights=d,minlength=len(uc))/np.maximum(1,uc),'f')
    def logits(a):return np.log(a.clip(.002,.998)/(1-a.clip(.002,.998)))
    up=np.empty((len(uc),4),'f');ip=np.empty((len(ic),4),'f')
    for k in range(4):
        mask=r>k+1;fraction=mask.mean()
        up[:,k]=logits((np.bincount(u,weights=mask,minlength=len(uc))+18*fraction)/(uc+18))-logits(np.array(fraction))
        ip[:,k]=logits((np.bincount(i,weights=mask,minlength=len(ic))+25*fraction)/(ic+25))
    coords=np.column_stack((projection(ne.verified_rank(train,True),18),projection(cf.checkpoint(train)['p'][2:].mean(0),10))).astype('f')
    rng=np.random.default_rng(851)
    film=np.stack([coords*.06+rng.normal(0,.027,coords.shape).astype('f') for _ in range(6)],axis=1)
    users=np.zeros((len(uc),28),'f');prior=np.zeros_like(users)
    def ptrs(*a):return [v.ctypes.data for v in a]
    lib.ord_prior(0,len(uc),*ptrs(off,i,r,coords,prior))
    ua=ptrs(off,i,d,r,md,up,ip,rank,nade,film,prior,users)
    ma=ptrs(starts,order,u,d,r,md,up,ip,rank,nade,users,film)
    def advance(fn,args,n,step,pool):
        def work(lo):fn(lo,min(lo+step,n),*args)
        list(pool.map(work,range(0,n,step)))
    start=time.monotonic();pred=[]
    with ThreadPoolExecutor(2) as pool:
        for ep in range(1,10):
            advance(lib.ord_users,ua,len(uc),640,pool)
            assert np.isfinite(users).all()
            advance(lib.ord_films,ma,len(ic),32,pool)
            assert np.isfinite(film).all()
            print(train,'sweep',ep,'seconds',round(time.monotonic()-start),flush=True)
            if ep in (3,6,9):
                z=np.empty((len(qi),8),'f')
                lib.ord_predict(len(qi),*ptrs(qu,qi,qd,md,up,ip,users,film,z))
                assert np.isfinite(z).all();pred.append(z)
    result=np.column_stack(pred);np.save(OUT/(phase+'.npy'),result)
    return result


def directions(raw,base):
    return np.column_stack([(raw[:,k+6]-base) if j==0 else raw[:,k+3+j] for k in (0,8,16) for j in range(3)])


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC','scripts/own_ordinal_als.cpp',
              'scripts/own_neighbor_features.cpp','-framework','Accelerate','-o',str(OUT/'als.dylib')],check=True)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    raw=run('trainx','probex');x=directions(raw,base).astype(float);held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);orig=blend.metrics(base,y,f);stats=blend.metrics(ne.clipped(base+held),y,f)
    ok=retained.qualifies(stats,orig);np.save(OUT/'oof.npy',held)
    rep=dict(selected='candidate' if ok else 'control',known=stats,control=orig,weights=w.tolist(),parent=str(parent),
             standalone=ne.rmse(raw[:,22],y),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        delta=directions(run('fulltrain','qual'),qb)@w
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
