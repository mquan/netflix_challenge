"""Fit film-specific responses to latent representations of co-rated movies."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import subprocess
import time

import numpy as np
from scipy.linalg import blas,solve
import own_neighbor_experiment as ne
import own_full_context as full
import own_session_pairs as pairs
import own_anchor_context as anchor
import own_history_context as hist
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-session-latent-20261126'
D=48
RIDGES=(80.,600.)


def ptrs(*a):return [v.ctypes.data for v in a]


def groups(lib,items,nitems):
    c=np.bincount(items,minlength=nitems)
    starts=np.r_[0,c.cumsum()].astype('i')
    rows=np.empty(len(items),'i')
    lib.group_items(len(items),*ptrs(items,starts),nitems,rows.ctypes.data)
    return starts,rows


def graph(lib,items,es,off,nitems):
    path=OUT/'graph-coordinates.npy'
    if path.exists():
        ref=json.loads((OUT/'graph.json').read_text())
        assert ref['embedding_sha256']==ne.sha(path)
        return np.load(path)
    starts,rows=groups(lib,items,nitems)
    matrix=np.zeros((nitems,nitems),'f')
    def block(i):lib.session_graph(i,min(i+64,nitems),nitems,*ptrs(starts,rows,es,off,items,matrix))
    with ThreadPoolExecutor(2) as ex:list(ex.map(block,range(0,nitems,64)))
    mass=matrix.sum(axis=1,dtype=float)
    total=mass.sum()
    for i in range(0,nitems,128):
        sl=slice(i,i+128)
        v=matrix[sl]
        ratio=(v*total)/np.maximum(mass[sl,None]*mass[None,:],1)
        matrix[sl]=(np.maximum(np.log(np.maximum(ratio,1)),0)*v/(v+1.5)).astype('f')
    q=np.random.default_rng(20261126).standard_normal((nitems,D+16),dtype='f')
    for _ in range(5):q=np.ascontiguousarray(np.linalg.qr(matrix@q)[0])
    val,rot=np.linalg.eigh(q.T@(matrix@q))
    v=(q@rot[:,-D:])*np.sqrt(np.maximum(val[-D:],0))
    v/=np.maximum(np.sqrt(np.average(v**2,axis=0,weights=mass)),1e-6)
    v=np.ascontiguousarray(np.clip(v,-8,8),dtype='f')
    np.save(path,v)
    (OUT/'graph.json').write_text(json.dumps(dict(embedding_sha256=ne.sha(path),implicit_rows=len(items),
        sessions=len(off)-1,uses_rating_values=False,dims=D))+'\n')
    return v


def generate(train,query):
    lib=ctypes.CDLL(str((OUT/'latent.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.session_graph.argtypes=[it]*3+[pt]*6
    lib.session_sums.argtypes=[it]*3+[pt]*8
    lib.session_design.argtypes=[it]*3+[pt]*7+[it,pt]
    for fn in ['group_items','session_graph','session_sums','session_design']:getattr(lib,fn).restype=None
    ei,qsession,es,off,items,r,er,ec=pairs.make_events(train,query)
    counts=ne.history(train,'item_cnts')
    nitems=len(counts)
    v=graph(lib,items,es,off,nitems)
    multi=np.diff(off)>1
    mapping=np.where(multi,np.cumsum(multi,dtype='i')-1,-1).astype('i')
    sums=np.empty((np.count_nonzero(multi),4*D+9),dtype='float16')
    def block(i):lib.session_sums(i,min(i+65536,len(mapping)),D,*ptrs(off,mapping,items,r,er,ec,v,sums))
    with ThreadPoolExecutor(2) as ex:list(ex.map(block,range(0,len(mapping),65536)))
    print('session vectors',sums.shape,flush=True)
    ti=ne.history(train,'item_idxs')
    starts,rows=groups(lib,ti,nitems)
    idx=ei[rows]
    tr,tre,tce,ts=[arr[idx] for arr in (r,er,ec,es)]
    del idx,rows,ei,items,r,er,ec,es
    gc.collect()
    qi=ne.history(query,'item_idxs')
    qs,qo=groups(lib,qi,nitems)
    sid=qsession[qo]
    zeros=np.zeros(1,'f')
    width=4*D+10
    def design(m,s,r,re,ce,training):
        x=np.empty((len(s),width),'f')
        lib.session_design(len(s),D,m,*ptrs(s,mapping,sums,v,r,re,ce),int(training),x.ctypes.data)
        return np.asfortranarray(x)
    result=np.zeros((len(qi),3*len(RIDGES)),'f')
    def one(m):
        a,b=starts[m:m+2]
        g,h=qs[m:m+2]
        if b==a or h==g:return
        gram=np.zeros((width,width),'f',order='F')
        rhs=np.zeros((width,3),'f')
        for lo in range(a,b,8192):
            sl=slice(lo,min(lo+8192,b))
            x=design(m,ts[sl],tr[sl],tre[sl],tce[sl],True)
            gram=blas.ssyrk(1.,x,trans=1,lower=1,c=gram,beta=1.,overwrite_c=True)
            rhs+=x.T@np.column_stack((tre[sl],tce[sl],tr[sl]-3.6)).astype('f')
        ws=[]
        for pen in RIDGES:
            gmat=gram.copy(order='F')
            gmat.flat[::width+1]+=pen
            ws.append(solve(gmat,rhs,lower=True,assume_a='pos',check_finite=False,overwrite_a=True))
        w=np.column_stack(ws)
        for lo in range(g,h,8192):
            sl=slice(lo,min(lo+8192,h))
            result[qo[sl]]=design(m,sid[sl],zeros,zeros,zeros,False)@w
    started=time.monotonic()
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(one,range(nitems))):
            if j%2000==0:print(query,j,round(time.monotonic()-started),flush=True)
    assert np.isfinite(result).all()
    np.save(OUT/(query+'.npy'),result)
    return result.astype(float)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_session_latent.cpp','scripts/own_neighbor_features.cpp','-o',str(OUT/'latent.dylib')],check=True)
    y,folds,p,q=anchor.history_control()
    x=generate('trainx','probex')
    x[:,2::3]+=3.6-p[:,None]
    held=p.copy()
    for i in range(5):
        tr=folds!=i
        w=fit(x[tr],(y-p)[tr])
        held[~tr]=ne.clipped(p[~tr]+x[~tr]@w)
    w=fit(x,y-p)
    gains=[ne.rmse(p[folds==f],y[folds==f])-ne.rmse(held[folds==f],y[folds==f]) for f in range(5)]
    accept=ne.rmse(held,y)<ne.rmse(p,y) and sorted(gains)[1]>0
    report=dict(selected='candidate' if accept else 'control',known_rmse=ne.rmse(held,y),fold_gains=gains,weights=w.tolist(),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False,
                baseline_sha256=ne.sha(hist.OUT/'selected.f32'))
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    np.save(OUT/'oof.npy',held-p)
    if accept:
        z=generate('fulltrain','qual')
        z[:,2::3]+=3.6-q[:,None]
        delta=z@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if accept:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
