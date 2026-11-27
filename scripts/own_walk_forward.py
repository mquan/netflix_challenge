"""Backtest earlier terminal cohorts for each individual and transfer their historical forecast errors."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
from own_query_sets import projection

OUT=ne.ROOT/'own-walk-forward-20261126'


def features(train,phase,base):
    lib=ctypes.CDLL(str((OUT/'walk.dylib').resolve()));it,ft,pt=ctypes.c_int,ctypes.c_float,ctypes.c_void_p
    lib.walk_forward.argtypes=[it]*3+[ft]*2+[pt]*20;lib.walk_forward.restype=None
    u,m,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    sort=np.lexsort((m,d,u));items,days,stars=[v[sort] for v in (m,d,r)];del sort
    off=np.r_[0,uc.cumsum()].astype('i8')
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    qoff=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()].astype('i');order=np.argsort(qu,kind='stable').astype('i')
    ou,_,od,_=anchor.old_probe();boundary=np.full(len(uc),30000,'i2')
    np.minimum.at(boundary,ou,od);np.minimum.at(boundary,ne.history('qual','user_idxs'),ne.history('qual','dates'))
    s=cf.checkpoint(train);enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
    dec=np.ascontiguousarray((s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2))
    b,ub,ud=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
    geo=np.column_stack((projection(s['a'].mean(0),12),projection(s['p'].mean(0),12)))
    geo=np.ascontiguousarray(geo/np.maximum(np.linalg.norm(geo,axis=1,keepdims=True),1e-7))
    film=np.ascontiguousarray((np.bincount(m,weights=r,minlength=len(ic))+20*r.mean())/(ic+20),'f')
    avg=d.mean(dtype=float);sd=(sum(np.sum((d[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(d),1000000))/len(d))**.5
    pred=base.astype('f');out=np.empty((len(qi),64),'f')
    args=(off,items,days,stars,boundary,qoff,order,qi,qd,pred,enc,s['enc_proj'],s['enc_bias'],dec,b,ub,ud,geo,film,out)
    ptr=[v.ctypes.data for v in args]
    def work(j):lib.walk_forward(j,min(j+256,len(uc)),len(ic),avg,sd,*ptr)
    with ThreadPoolExecutor(2) as pool:
        for k,_ in enumerate(pool.map(work,range(0,len(uc),256))):
            if k%400==0:print(phase,k,flush=True)
    assert np.isfinite(out).all()
    return np.column_stack((out,full.cf96(phase)-base,ne.verified_rank(phase)-base,
                             np.log1p(uc[qu]),np.log1p(ic[qi]))).astype('f')


def model():return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=65,thread_count=2,
                          random_seed=53741,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','-Wno-deprecated-declarations','scripts/own_walk_forward.cpp',
                    '-framework','Accelerate','-o',str(OUT/'walk.dylib')],check=True)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base);key,known,held=anchor.crossfit(x,y,f,base,model)
    rep=dict(parent=str(parent),known=known,selected=key,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb);delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
