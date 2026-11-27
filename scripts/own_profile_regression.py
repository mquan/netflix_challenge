"""Learn implicit and explicit movie coordinates and fit leave-one-rating-out user profiles."""
import ctypes
import gc
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from scipy import sparse
from scipy.linalg import blas,solve
import own_neighbor_experiment as ne
from own_empirical_neighborhood import fit,signals
from own_history_context import neighbor_control

OUT=ne.ROOT/'own-profile-regression-20261126'
D=48
NONLINEAR=160
PENALTIES=(64.,512.)


def basis(items,starts,counts,ratings,mode):
    nusers=len(starts)-1
    off=np.diff(starts)
    # Normalize rows by activity and columns by popularity before the
    # subspace iteration. CSR indices follow the native user-sorted data.
    row_scale=np.maximum(off,1).astype('f')**-.45
    col_scale=np.maximum(counts,1).astype('f')**-.4
    values=np.repeat(row_scale,off)
    for a in range(0,len(values),1_000_000):
        sl=slice(a,a+1_000_000)
        values[sl]*=col_scale[items[sl]]
        if mode=='explicit': values[sl]*=ratings[sl]-3.6
    x=sparse.csr_matrix((values,items.astype('i'),starts), shape=(nusers,len(counts)))
    xt=x.T.tocsr()
    rng=np.random.default_rng(20261126)
    v=rng.standard_normal((len(counts),D+8),dtype='f')
    for epoch in range(5):
        up=x@v
        v=np.ascontiguousarray(np.linalg.qr(xt@up)[0])
        print(mode,'subspace',epoch+1,flush=True)
    up=x@v
    eig,rot=np.linalg.eigh(up.T@up)
    embedding=(v@rot[:,-D:])*col_scale[:,None]
    embedding/=np.sqrt(np.average(embedding**2,axis=0,weights=counts)).astype('f')
    return np.ascontiguousarray(embedding)


def generate(train,query):
    lib=ctypes.CDLL(str((OUT/'features.dylib').resolve()))
    pt,it=ctypes.c_void_p,ctypes.c_int
    lib.profiles.argtypes=[it,it]+[pt]*8
    lib.profile_rows.argtypes=[it]*3+[pt]*9+[it,pt]
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.group_items.restype=lib.profiles.restype=lib.profile_rows.restype=None
    def ptrs(*a): return [v.ctypes.data for v in a]
    users,items,dates,ratings=[ne.history(train,key) for key in ('user_idxs','item_idxs','dates','ratings')]
    counts,uc=[ne.history(train,key) for key in ('item_cnts','user_cnts')]
    starts=np.r_[0,counts.cumsum()].astype('i')
    us=np.r_[0,uc.cumsum()].astype('i')
    imp=basis(items,us,counts,ratings,'implicit')
    exp=basis(items,us,counts,ratings,'explicit')
    gc.collect()
    e=ratings-ne.verified_rank(train)
    dim=6*D+8
    sums=np.zeros((len(uc),dim),'f')
    lib.profiles(len(uc),D,*ptrs(us,items,dates,ratings,e,imp,exp,sums))
    norms=np.sqrt(uc.astype('f')+15)
    std=np.std(sums[:,:6*D]/norms[:,None],axis=0)
    scale=np.ascontiguousarray(1/np.maximum(std,.001))
    order=np.empty(len(items),'i')
    lib.group_items(len(items),items.ctypes.data,starts.ctypes.data,len(counts),order.ctypes.data)
    iu,idates,ir,ie=[x[order] for x in (users,dates,ratings,e)]
    del order,e
    qi,qu,qd=[ne.history(query,key) for key in ('item_idxs','user_idxs','dates')]
    qo=np.argsort(qi,kind='stable')
    qs=np.r_[0,np.bincount(qi,minlength=len(counts)).cumsum()]
    qu,qd=qu[qo],qd[qo]
    zeros=np.zeros(1,'f')
    rng=np.random.default_rng(20261126)
    random=rng.normal(0,1/np.sqrt(dim),(dim+4,NONLINEAR)).astype('f')
    offset=rng.normal(0,.5,NONLINEAR).astype('f')
    def design(item,u,d,r,rs,training):
        x=np.empty((len(u),dim+4),'f')
        lib.profile_rows(len(u),D,item,*ptrs(u,d,r,rs,uc,sums,imp,exp,scale),int(training),x.ctypes.data)
        hidden=np.tanh(x@random+offset)
        return np.asfortranarray(np.column_stack((x,hidden)))
    result=np.zeros((len(qi),4),'f')
    def one(item):
        a,b=int(starts[item]),int(starts[item+1])
        g,h=int(qs[item]),int(qs[item+1])
        if a==b or g==h: return
        width=dim+4+NONLINEAR
        gram=np.zeros((width,width),'f',order='F')
        rhs=np.zeros((width,2),'f')
        for lo in range(a,b,8192):
            sl=slice(lo,min(b,lo+8192))
            x=design(item,iu[sl],idates[sl],ir[sl],ie[sl],True)
            gram=blas.ssyrk(1.,x,trans=1,lower=1,c=gram,beta=1.,overwrite_c=True)
            rhs+=x.T@np.column_stack((ie[sl],ir[sl]-3.6)).astype('f')
        ws=[]
        for penalty in PENALTIES:
            mat=gram.copy(order='F')
            mat.flat[::len(mat)+1]+=penalty
            ws.append(solve(mat,rhs,assume_a='pos',lower=True,check_finite=False,overwrite_a=True))
        w=np.column_stack(ws)
        for lo in range(g,h,8192):
            sl=slice(lo,min(h,lo+8192))
            result[qo[sl]]=design(item,qu[sl],qd[sl],zeros,zeros,False)@w
    start=time.monotonic()
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(one,range(len(counts)))):
            if j%1000==0:print(query,j,round(time.monotonic()-start),flush=True)
    assert np.isfinite(result).all()
    np.save(OUT/(query+'.npy'),result)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_profile_features.cpp','scripts/own_neighbor_features.cpp','-o',str(OUT/'features.dylib')],check=True)
    y,folds,p,q=neighbor_control()
    x=signals(generate('trainx','probex'),p)
    oof=np.empty_like(y)
    ws=[]
    for f in range(5):
        tr=folds!=f
        w=fit(x[tr],(y-p)[tr])
        oof[~tr]=ne.clipped(p[~tr]+x[~tr]@w)
        ws.append(w.tolist())
    np.save(OUT/'known-selected.npy',oof)
    w=fit(x,y-p)
    gains=[ne.rmse(p[folds==f],y[folds==f])-ne.rmse(oof[folds==f],y[folds==f]) for f in range(5)]
    ok=min(ne.rmse(p,y)-ne.rmse(oof,y),sorted(gains)[1])>0
    report=dict(selected='candidate' if ok else 'control',fold_gains=gains,known_rmse=ne.rmse(oof,y),
                weights=w.tolist(),fold_weights=ws,qualifying_hashes={},known_comparison_is_nested=False,
                qualifying_labels_used_for_fitting=False,baseline_sha256=ne.sha(ne.OUT/'selected.f32'))
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if ok:
        qx=signals(generate('fulltrain','qual'),q)
        path=OUT/'selected.f32'
        ne.clipped(q+qx@w).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok: ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
