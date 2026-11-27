"""Select co-rating neighbors on training rows and fit item-specific multivariate ridge."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.linalg import blas, solve
from scipy.optimize import lsq_linear
import own_neighbor_experiment as ne
from own_history_context import neighbor_control

OUT=ne.ROOT/'own-empirical-neighborhood-20261126'
CAP=160
PENALTIES=(96.,768.)


def generate(train, query, workers):
    lib=ctypes.CDLL(str((OUT/'features.dylib').resolve()))
    pt,it=ctypes.c_void_p,ctypes.c_int
    lib.pair_scores.argtypes=[it]*3+[pt]*8
    lib.design.argtypes=[it,pt,pt,it]+[pt]*7
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.pair_scores.restype=lib.group_items.restype=lib.design.restype=None
    items=ne.history(train,'item_idxs')
    users=ne.history(train,'user_idxs')
    dates=ne.history(train,'dates')
    ratings=ne.history(train,'ratings')
    residuals=ratings-ne.verified_rank(train)
    counts=ne.history(train,'item_cnts')
    starts=np.r_[0,counts.cumsum()].astype('i')
    ustarts=np.r_[0,ne.history(train,'user_cnts').cumsum()].astype('i')
    nr=len(counts)
    order=np.empty(len(items),'i')
    lib.group_items(len(items),items.ctypes.data,starts.ctypes.data,nr,order.ctypes.data)
    iu,idates,ir,ie=[a[order] for a in (users,dates,ratings,residuals)]
    qu,qi,qd=[ne.history(query,key) for key in ('user_idxs','item_idxs','dates')]
    qo=np.lexsort((qu,qi))
    qs=np.r_[0,np.bincount(qi,minlength=nr).cumsum()]
    qu,qd=qu[qo],qd[qo]
    factor=ne.find_neighbors(ne.verified_rank(train,True),counts)
    def ptrs(*arrays):
        return [a.ctypes.data for a in arrays]
    pair_args=ptrs(users,items,dates,ustarts,residuals,ratings)
    design_args=ptrs(starts,iu,idates,ir,ie)
    result=np.zeros((len(qi),2*len(PENALTIES)),dtype='f')
    related_ids=np.full((nr,CAP),-1,'i')
    def design(u,d,neighbors):
        k=len(neighbors)
        x=np.zeros((len(u),k*5+1),'f',order='F')
        lib.design(len(u),*ptrs(u,d),k,neighbors.ctypes.data,*design_args,x.ctypes.data)
        x[:,-1]=1
        return x
    def one(item):
        a,b=int(starts[item]),int(starts[item+1])
        g,h=int(qs[item]),int(qs[item+1])
        if a==b or g==h: return
        scores=np.empty((nr,3),'f')
        lib.pair_scores(nr,item,b-a,order[a:b].ctypes.data,*pair_args,scores.ctypes.data)
        # Round-robin selection retains complementary neighborhoods even when
        # their score scales and overlap vary by target film.
        lists=np.argsort(-scores,axis=0,kind='stable')[:CAP]
        candidates=dict.fromkeys(factor[item].tolist())
        for ids in lists:
            candidates.update(dict.fromkeys(ids.tolist()))
            if len(candidates)>=CAP: break
        related=np.array(list(candidates)[:CAP],dtype='i')
        assert item not in related
        related_ids[item]=related
        dim=len(related)*5+1
        gram=np.zeros((dim,dim),'f',order='F')
        rhs=np.zeros((dim,2),'f')
        for lo in range(a,b,8192):
            sl=slice(lo,min(lo+8192,b))
            x=design(iu[sl],idates[sl],related)
            gram=blas.ssyrk(1.,x,trans=1,lower=1,c=gram,beta=1.,overwrite_c=True)
            target=np.column_stack((ie[sl],ir[sl]-3.6)).astype('f')
            rhs+=x.T@target
        weights=[]
        for penalty in PENALTIES:
            mat=gram.copy(order='F')
            mat.flat[::dim+1]+=penalty
            weights.append(solve(mat,rhs,assume_a='pos',lower=True,overwrite_a=True,check_finite=False))
        w=np.column_stack(weights)
        for lo in range(g,h,8192):
            sl=slice(lo,min(lo+8192,h))
            result[qo[sl]]=design(qu[sl],qd[sl],related)@w
    start=time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        for j,_ in enumerate(ex.map(one,range(nr))):
            if j%1000==0: print(query,j,round(time.monotonic()-start),flush=True)
    np.save(OUT/(query+'-neighbors.npy'),related_ids)
    np.save(OUT/(query+'.npy'),result)
    return result


def signals(raw,base):
    x=raw.astype(float)
    # The second output of each regression models centered stars directly.
    x[:,1::2]+=3.6-base[:,None]
    return x


def fit(x,y):
    gram=x.T@x
    gram.flat[::len(gram)+1]+=25
    L=np.linalg.cholesky(gram)
    return lsq_linear(L.T,np.linalg.solve(L,x.T@y),bounds=(0.,1.)).x


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','scripts/own_neighbor_features.cpp','-o',str(OUT/'features.dylib')],check=True)
    y,folds,p,q=neighbor_control()
    x=signals(generate('trainx','probex',args.workers),p)
    residual=y-p
    held=np.empty_like(y)
    ws=[]
    for f in range(5):
        tr=folds!=f
        w=fit(x[tr],residual[tr])
        held[~tr]=ne.clipped(p[~tr]+x[~tr]@w)
        ws.append(w.tolist())
    w=fit(x,residual)
    np.save(OUT/'known-selected.npy',held)
    def metrics(pred):
        return dict(pooled=ne.rmse(pred,y),folds=[ne.rmse(pred[folds==f],y[folds==f]) for f in range(5)])
    old,new=metrics(p),metrics(held)
    wins=sum(a<b for a,b in zip(new['folds'],old['folds']))
    accept=new['pooled']<old['pooled'] and wins>=4
    report=dict(control=old,candidate=new,fold_weights=ws,weights=w.tolist(),selected='candidate' if accept else 'control',
                improved_folds=wins,qualifying_hashes={},neighbors=CAP,penalties=PENALTIES,
                baseline_sha256=ne.sha(ne.OUT/'selected.f32'),known_comparison_is_nested=False,qualifying_labels_used_for_fitting=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if accept:
        qx=signals(generate('fulltrain','qual',args.workers),q)
        path=OUT/'selected.f32'
        ne.clipped(q+qx@w).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if accept: ne.audit(OUT,ne.LABELS)


if __name__=='__main__':
    main()
