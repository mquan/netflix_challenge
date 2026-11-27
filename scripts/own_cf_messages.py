"""Infer other hidden ratings from our frozen CF96 and re-encode their expected embeddings."""
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

OUT=ne.ROOT/'own-cf-messages-20261126'
STARS=np.arange(1,6,dtype='f')


def tilt(p,target):
    # Match a held-user prediction while preserving the ordinal base distribution.
    t=np.zeros(len(target),'f')
    for _ in range(4):
        q=p*np.exp(t[:,None]*(STARS-3))
        q/=q.sum(1,keepdims=True)
        mu=q@STARS
        var=np.maximum(q@(STARS**2)-mu**2,.05)
        t=np.clip(t+np.clip((target-mu)/var,-1,1),-4,4)
    q=p*np.exp(t[:,None]*(STARS-3))
    return q/q.sum(1,keepdims=True)


def peer_sum(value,group):
    _,ix,counts=np.unique(group,return_inverse=True,return_counts=True)
    order=np.argsort(ix,kind='stable')
    sums=np.add.reduceat(value[order],np.r_[0,counts.cumsum()[:-1]],axis=0)
    return sums[ix]-value,counts[ix,None]-1


def features(train,phase,base):
    uc=ne.history(train,'user_cnts')
    ic=ne.history(train,'item_cnts')
    ti,td,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    n=len(qi)
    if phase=='probex':
        qu,qi,qd=[np.concatenate((a,ne.history('qual',k))) for a,k in zip((qu,qi,qd),('user_idxs','item_idxs','dates'))]
    sort=np.argsort(qu,kind='stable')
    off=np.r_[0,uc.cumsum()].astype('i')
    qoff=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()]
    s=cf.checkpoint(train)
    enc=np.ascontiguousarray(np.cumsum(s['a'],axis=0,dtype=float),'f')
    totals=np.empty((len(uc),96),'f');md=np.empty(len(uc),'f')
    lib=ctypes.CDLL(str((OUT/'totals.dylib').resolve()))
    lib.cf_totals.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*7
    lib.cf_totals.restype=None
    lib.cf_totals(len(uc),len(ic),*[a.ctypes.data for a in (off,ti,td,r,enc,totals,md)])
    avg=td.mean(dtype=float);md[uc==0]=avg
    variance=0.;counts=np.zeros((len(ic),5),'int64')
    for j in range(0,len(td),1000000):
        sl=slice(j,j+1000000);z=td[sl].astype(float)-avg
        variance+=np.dot(z,z)
        idx=ti[sl].astype('i')*5+r[sl]-1
        counts+=np.bincount(idx,minlength=len(ic)*5).reshape(-1,5)
    prior=(counts+1)/(counts.sum(1,keepdims=True)+5)
    typical=np.einsum('mr,rmk->mk',prior.astype('f'),enc)
    sd=(variance/len(td))**.5
    native=full.cf96(phase)
    result=np.empty((n,44),'f');maxerr=0.
    for lo in range(0,len(uc),512):
        ix=sort[qoff[lo]:qoff[min(len(uc),lo+512)]]
        keep=ix<n
        if not keep.any():continue
        us,mi,day=qu[ix],qi[ix],qd[ix]
        cnt=uc[us,None].astype('f');obs=totals[us]
        diff=day-md[us];dev=np.sign(diff)*np.power(np.abs(diff)/sd,.4)
        def decode(z):
            return cf.decode(z[:,None,:],s,mi,us,dev)[0][:,0]
        raw=decode(obs/np.sqrt(np.maximum(1,cnt)))
        maxerr=max(maxerr,float(np.max(abs((raw[keep]@STARS)-native[ix[keep]]))))
        assert maxerr<.0003,maxerr
        label_free=raw.copy()
        label_free[keep]=tilt(raw[keep],base[ix[keep]])
        outputs=[raw]
        for probs in (raw,label_free):
            e=np.einsum('nr,rnk->nk',probs,enc[:,mi])
            others,k=peer_sum(e,us)
            cx,ck=peer_sum(e-typical[mi],us)
            dx,dk=peer_sum(e,us.astype('int64')*65536+day)
            for msg,num,alpha in ((others,k,.4),(others,k,1.),(dx,dk,1.)):
                outputs.append(decode((obs+alpha*msg)/np.sqrt(np.maximum(1,cnt+alpha**2*num)).astype('f')))
            outputs.append(decode((obs+cx)/np.sqrt(np.maximum(1,cnt))))
        # Retain original probabilities and ratings 1-4 of each shift. The fifth follows from normalization.
        z=np.column_stack([raw]+[(pr-raw)[:,:4] for pr in outputs[1:]]+[cnt,k,dk,raw@STARS,outputs[-1]@STARS,outputs[2]@STARS,outputs[3]@STARS])
        result[ix[keep]]=z[keep]
        if lo%65536==0:print(phase,lo,'native error',maxerr,flush=True)
    assert np.isfinite(result).all()
    return np.column_stack((base,result))


def model():
    return CatBoostRegressor(iterations=1050,depth=5,learning_rate=.045,l2_leaf_reg=50,
        thread_count=2,allow_writing_files=False,random_seed=1463,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cf_adapter.cpp','-o',str(OUT/'totals.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base)
    key,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},
        qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
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
