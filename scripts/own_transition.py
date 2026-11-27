"""Estimate how earlier and later movie choices change the conditional distribution of a target rating."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.cluster.vq import kmeans2
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_session_latent as latent
from own_query_sets import projection

OUT=ne.ROOT/'own-transition-20261126'


def loo(train):
    lib=ctypes.CDLL(str((OUT/'transitions.dylib').resolve()));it,pt,ft=ctypes.c_int,ctypes.c_void_p,ctypes.c_float
    lib.cf_totals.argtypes=[it,it]+[pt]*7;lib.cf_totals.restype=None
    lib.predict_rows.argtypes=[it,ft,ft]+[pt]*14;lib.predict_rows.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    s=cf.checkpoint(train);enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
    off=np.r_[0,uc.cumsum()].astype('i');total=np.empty((len(uc),96),'f');md=np.empty(len(uc),'f')
    def ptr(*v):return [x.ctypes.data if x is not None else None for x in v]
    lib.cf_totals(len(uc),len(ic),*ptr(off,i,d,r,enc,total,md))
    avg=d.mean(dtype=float);md[uc==0]=avg;sd=(sum(np.sum((d[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(d),1000000))/len(d))**.5
    total=total@s['enc_proj'].T;ep=np.ascontiguousarray((enc@s['enc_proj'].T).transpose(1,0,2))
    op=np.ascontiguousarray((s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2))
    b,ub,ud=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
    shared=ptr(uc,md,total,ep,op,s['enc_bias'],b,ub,ud)
    result=np.empty(len(r),'f')
    def predict(rows,mask,out):
        us=np.ascontiguousarray(u[rows],'i');mi=np.ascontiguousarray(i[rows]+1,'u2');dd=np.ascontiguousarray(d[rows]+314,'u2')
        rr=np.ascontiguousarray(r[rows],'u1') if mask else None
        lib.predict_rows(len(us),sd,avg,*ptr(us,mi,dd,rr),*shared,out.ctypes.data)
    rows=np.linspace(0,len(r)-1,13000,dtype='i');replay=np.empty(len(rows),'f');predict(rows,False,replay)
    err=float(np.max(abs(replay-full.cf96(train)[rows])));assert err<.0003,err
    def work(j):
        sl=slice(j,j+8192);predict(sl,True,result[sl]);result[sl]=r[sl]-result[sl]
    with ThreadPoolExecutor(2) as pool:
        for _ in pool.map(work,range(0,len(r),8192)):pass
    print(train,'target-removed residuals',len(r),'native replay error',err,flush=True)
    return result,projection(s['p'].mean(0),18)


def features(train,phase,base):
    residual,geo=loo(train)
    lib=ctypes.CDLL(str((OUT/'transitions.dylib').resolve()));it,pt=ctypes.c_int,ctypes.c_void_p
    lib.group_items.argtypes=[it,pt,pt,it,pt];lib.group_items.restype=None
    lib.transitions.argtypes=[it]*3+[pt]*18;lib.transitions.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    sort=np.lexsort((d,u));u,i,d,r,residual=[v[sort] for v in (u,i,d,r,residual)];del sort
    off=np.r_[0,uc.cumsum()].astype('i');moff,mrows=latent.groups(lib,i,len(ic));qoff,qr=latent.groups(lib,qi,len(ic))
    ou,oi,od,_=anchor.old_probe();vis,_=anchor.training_indices(train,ou,oi,od)
    cu,ci,cd=[np.r_[x,ne.history('qual',k)].astype(t) for x,k,t in zip((ou,oi,od),('user_idxs','item_idxs','dates'),('i','i2','i2'))]
    seen=np.r_[vis,np.zeros(len(cu)-len(vis),bool)].astype('i1')
    ix=np.argsort(cu,kind='stable');coff=np.r_[0,np.bincount(cu,minlength=len(uc)).cumsum()].astype('i');ci,cd,seen=[v[ix] for v in (ci,cd,seen)]
    c=kmeans2(geo,96,iter=18,minit='++',seed=751)[1].astype('i')
    out=np.empty((len(qi),63),'f');args=(moff,mrows,u,off,i,d,r,residual,qoff,qr,qu,qd,coff,ci,cd,seen,c,out)
    def work(j):lib.transitions(j,min(j+16,len(ic)),len(ic),*[a.ctypes.data for a in args])
    with ThreadPoolExecutor(2) as ex:
        for k,_ in enumerate(ex.map(work,range(0,len(ic),16))):
            if k%250==0:print(phase,'films',k*16,flush=True)
    assert np.isfinite(out).all()
    return np.column_stack((base,full.cf96(phase)-base,ne.verified_rank(phase)-base,np.log1p(uc[qu]),np.log1p(ic[qi]),out)).astype('f')


def model():return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.04,l2_leaf_reg=65,thread_count=2,
                          random_seed=7377,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','-Wno-deprecated-declarations','scripts/own_transition.cpp',
        'scripts/own_residual_bayes.cpp','scripts/own_cf_adapter.cpp','scripts/own_neighbor_features.cpp',
        '-framework','Accelerate','-o',str(OUT/'transitions.dylib')],check=True)
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
