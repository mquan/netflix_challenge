"""Fit local Gaussian-process corrections in two independently trained movie geometries."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import own_neighbor_experiment as ne
import own_full_context as full
import own_anchor_context as anchor
import own_session_latent as latent
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-user-gp-20261126'


def geometry(train):
    p=ne.verified_rank(train,True)
    _,vec=np.linalg.eigh(p.T@p)
    rv=np.ascontiguousarray(p@vec[:,-48:])
    sv=np.load(latent.OUT/'graph-coordinates.npy')
    assert ne.sha(latent.OUT/'graph-coordinates.npy')==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    ans=[]
    for x in [rv,sv]:
        v=np.ascontiguousarray(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-10))
        n=len(v)
        sims=np.empty((n,n),dtype='i2')
        for j in range(0,n,128):
            sl=slice(j,j+128)
            sims[sl]=np.rint((v[sl]@v.T)*16000).astype('i2')
        ans.append((v,sims))
    return ans


def generate(train,query):
    lib=ctypes.CDLL(str((OUT/'user-gp.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.user_gp.argtypes=[it]*3+[pt]*16
    lib.user_gp.restype=None
    (rv,sr),(sv,ss)=geometry(train)
    tu,ti,td,tr=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    u,i,d=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    re=tr-ne.verified_rank(train)
    ce=tr-full.cf96(train)
    mu=float(tr.mean())
    user=np.ascontiguousarray((np.bincount(tu,weights=tr,minlength=len(uc))+10*mu)/(uc+10),dtype='f')
    movie=np.zeros(len(ic))
    for j in range(0,len(tu),1000000):
        sl=slice(j,j+1000000)
        movie+=np.bincount(ti[sl],weights=tr[sl]-user[tu[sl]],minlength=len(ic))
    movie=np.ascontiguousarray(movie/(ic+30),dtype='f')
    output=np.zeros((len(u),12),'f')
    argv=[v.ctypes.data for v in (u,i,d,off,ti,td,tr,re,ce,movie,user,sr,ss,rv,sv,output)]
    def chunk(k):lib.user_gp(k,min(k+1024,len(u)),len(ic),*argv)
    started=time.monotonic()
    with ThreadPoolExecutor(2) as ex:
        for j,_ in enumerate(ex.map(chunk,range(0,len(u),1024))):
            if j%200==0:print(query,j*1024,round(time.monotonic()-started),flush=True)
    assert np.isfinite(output).all()
    np.save(OUT/(query+'.npy'),output)
    return output.astype(float)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-ffp-contract=fast','-std=c++17','-shared','-fPIC','scripts/own_user_gp.cpp','-o',str(OUT/'user-gp.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    root=ne.ROOT/'own-overnight-blend-20261126/first'
    rec=json.loads((root/'freeze.json').read_text())
    assert ne.sha(root/'selected.f32')==rec['qualifying_hashes']['selected']
    p=np.load(root/'known-selected.npy')
    q=np.fromfile(root/'selected.f32',dtype='<f4').astype(float)
    x=generate('trainx','probex')
    x[:,2::3]-=p[:,None]
    held=p.copy()
    for j in range(5):
        tr=f!=j
        w=fit(x[tr],(y-p)[tr])
        held[~tr]=ne.clipped(p[~tr]+x[~tr]@w)
    w=fit(x,y-p)
    gains=[ne.rmse(p[f==j],y[f==j])-ne.rmse(held[f==j],y[f==j]) for j in range(5)]
    ok=sorted(gains)[1]>0 and ne.rmse(held,y)<ne.rmse(p,y)
    np.save(OUT/'oof.npy',held-p)
    report=dict(selected='candidate' if ok else 'control',fold_gains=gains,known_rmse=ne.rmse(held,y),weights=w.tolist(),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if ok:
        z=generate('fulltrain','qual')
        z[:,2::3]-=q[:,None]
        delta=z@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
