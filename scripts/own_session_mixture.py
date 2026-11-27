"""Learn ordinal response styles shared across a user's ratings and same-day sessions."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend

OUT=ne.ROOT/'own-session-mixture-20261126'
K=8
B=25


def initialize():
    means=np.linspace(1,5,B)[:,None,None]
    stars=np.arange(1,6)[None,:,None]
    shift=np.array([-.8,-.4,0,.4,.8,0,0,0])
    spread=np.array([1.,1.,.9,.95,.9,.55,1.6,1.1])
    logits=-.5*((stars-means-shift)/spread)**2
    logits[:,:,7]+=.8*(abs(stars[:,:,0]-3)-1)
    p=np.exp(logits-logits.max(1,keepdims=True))
    return np.ascontiguousarray(p/p.sum(1,keepdims=True),'f')


def features(train,phase,base):
    u,it,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    nu=len(uc)
    off=np.r_[0,uc.cumsum()].astype('i')
    qs=np.r_[0,np.bincount(qu,minlength=nu).cumsum()].astype('i')
    qo=np.argsort(qu,kind='stable').astype('i')
    pt=ctypes.c_void_p;num=ctypes.c_int
    lib=ctypes.CDLL(str((OUT/'mixture.dylib').resolve()))
    lib.chronological.argtypes=[num]+[pt]*3
    lib.mix_step.argtypes=[num]*2+[pt]*6
    lib.mix_query.argtypes=[num]*2+[pt]*11
    for fn in (lib.chronological,lib.mix_step,lib.mix_query):fn.restype=None
    def ptrs(*a):return [v.ctypes.data for v in a]
    order=np.empty(len(r),'i')
    lib.chronological(nu,*ptrs(off,d,order))
    days=d[order]
    usr=np.ascontiguousarray((np.bincount(u,weights=r,minlength=nu)+15*r.mean())/(uc+15),'f')
    film=np.ascontiguousarray((np.bincount(it,weights=r,minlength=len(ic))+25*r.mean())/(ic+25),'f')
    rp=ne.verified_rank(train);cp=full.cf96(train)
    qr=ne.verified_rank(phase);qc=full.cf96(phase)
    result=[]
    bounds=list(zip(np.linspace(0,nu,9,dtype=int)[:-1],np.linspace(0,nu,9,dtype=int)[1:]))
    for kind in range(2):
        def discretize(v):return np.clip(np.rint((v-1)*6),0,24).astype('uint8')
        code=np.empty(len(r),'uint8')
        for j in range(0,len(r),500000):
            sl=slice(j,j+500000);rows=order[sl]
            if kind:
                n=uc[u[rows]]
                # Leave this observed rating out of its own user-score covariate.
                v=.85*(usr[u[rows]]*(n+15)-r[rows])/(n+14)+.8*film[it[rows]]-.65*3.6
            else:v=(rp[rows]+cp[rows])*.5
            code[sl]=discretize(v)*5+r[rows]-1
        qbin=discretize(.85*usr[qu]+.8*film[qi]-.65*3.6 if kind else .5*(qr+qc))
        prior=np.zeros((nu,K),'f')
        prob=initialize()
        with ThreadPoolExecutor(2) as pool:
            for epoch in range(9):
                likelihood=np.ascontiguousarray(np.log(prob),'f')
                def one(bound):
                    acc=np.zeros((B,5,K))
                    lib.mix_step(*bound,*ptrs(off,days,code,likelihood,prior,acc))
                    return acc
                a=sum(pool.map(one,bounds))+50*initialize()
                prob=np.ascontiguousarray(a/a.sum(1,keepdims=True),'f')
                print(phase,kind,epoch+1,flush=True)
            likelihood=np.ascontiguousarray(np.log(prob),'f')
            x=np.empty((len(qi),22),'f')
            def infer(bound):
                lib.mix_query(*bound,*ptrs(off,days,code,likelihood,prob,prior,qs,qo,qd,qbin,x))
            list(pool.map(infer,bounds))
        assert np.isfinite(x).all()
        np.testing.assert_allclose(x[:,:20].reshape(-1,4,5).sum(2),1,atol=3e-6)
        result.append(x)
    return np.column_stack((base,qr-base,qc-base,usr[qu],film[qi],np.log1p(uc[qu]),np.log1p(ic[qi]),*result)).astype('f')


def model():
    return CatBoostRegressor(iterations=1100,depth=5,learning_rate=.04,l2_leaf_reg=45,
        thread_count=2,verbose=False,allow_writing_files=False,random_seed=3153)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_session_mixture.cpp',
                    'scripts/own_session_spectrum.cpp','-o',str(OUT/'mixture.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base)
    key,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,
             known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
