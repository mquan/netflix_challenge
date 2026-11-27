"""Jointly impute the fixed Probe and qualifying movie cohort with our trained CF96."""
import argparse
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_cf_messages as messages
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained

OUT=ne.ROOT/'own-cf-cohort-20261126'


def features(train,phase,base,teachers=None):
    m,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    orig=anchor.old_probe()
    old_u,old_i,old_d=orig[:3]
    observed,rows=anchor.training_indices(train,old_u,old_i,old_d)
    qu,qi,qd=[np.concatenate((a,ne.history('qual',k))) for a,k in zip(orig[:3],('user_idxs','item_idxs','dates'))]
    flags=np.zeros(len(qu),'i1');flags[np.flatnonzero(observed)]=r[rows]
    target=np.full(len(qu),-1,'i')
    if phase=='probex':target[retained.map_rows()]=np.arange(len(base))
    else:target[len(old_u):]=np.arange(len(base))
    assert np.all(flags[target>=0]==0)
    # Removing the observed original-Probe cohort yields the original Prize training row count in either arm.
    assert len(r)-observed.sum()==99072112
    qo=np.argsort(qu,kind='stable')
    qoff=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()]
    s=cf.checkpoint(train);enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
    total=np.empty((len(uc),96),'f');md=np.empty(len(uc),'f')
    lib=ctypes.CDLL(str((OUT/'totals.dylib').resolve()))
    lib.cf_totals.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*7;lib.cf_totals.restype=None
    lib.cf_totals(len(uc),len(ic),*[a.ctypes.data for a in (off,m,d,r,enc,total,md)])
    avg=d.mean(dtype=float);md[uc==0]=avg
    sd=(sum(np.sum((d[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(d),1000000))/len(d))**.5
    result=np.zeros((len(base),50 if teachers is None else 52),'f');largest=0.
    native=full.cf96(phase)
    for lo in range(0,len(uc),384):
        ix=qo[qoff[lo]:qoff[min(lo+384,len(uc))]];ret=target[ix];keep=ret>=0
        if not keep.any():continue
        u,i,day,known=qu[ix],qi[ix],qd[ix],flags[ix]
        visible=known>0
        grp,ixuser,counts=np.unique(u,return_inverse=True,return_counts=True)
        begin=np.r_[0,counts.cumsum()[:-1]]
        def grouped(x):return np.add.reduceat(x,begin,axis=0)[ixuser]
        actual=enc[np.maximum(known-1,0),i]*visible[:,None]
        seen=grouped(visible.astype('i'));num=uc[u]-seen
        h0=total[u]-grouped(actual)
        diff=day-md[u];dev=np.sign(diff)*np.abs(diff/sd)**.4
        def decode(v,n):
            z=v/np.sqrt(np.maximum(1,n)).astype('f')[:,None]
            return cf.decode(z[:,None,:],s,i,u,dev)[0][:,0]
        original=decode(total[u],uc[u])
        largest=max(largest,float(np.max(abs(original[keep]@messages.STARS-native[ret[keep]]))))
        assert largest<.0003,largest
        start=decode(h0,num)
        denom=num+counts[ixuser]-1
        def step(prob,anchors,sample=None):
            if sample is None:e=np.einsum('nr,rnk->nk',prob,enc[:,i])
            else:
                rv=(np.cumsum(prob,axis=1)<sample[:,None]).sum(1).clip(0,4)
                e=enc[rv,i]
            if anchors:e=np.where(visible[:,None],actual,e)
            return decode(h0+grouped(e)-e,denom)
        if teachers is not None:
            priors=np.column_stack([model[ix] for model in teachers]).clip(1,5)
            mu=priors.mean(1)
            models=[original,start]
            # Each external factor forecast excluded every competition-cohort rating at training time.
            for strength,anchor_observed in ((.55,True),(1.,True),(.75,False)):
                v=start
                for it in range(2):
                    anchored=messages.tilt(v,(1-strength)*(v@messages.STARS)+strength*mu)
                    v=step(anchored,anchor_observed)
                    models.append(v)
            actual_peer=(mu-start@messages.STARS)[:,None]
            others=grouped(actual_peer)-actual_peer
            near,sz=messages.peer_sum(actual_peer,u.astype('int64')*4096+day)
            features=np.column_stack([original]+[(v-original)[:,:4] for v in models[1:]]+
                         [v@messages.STARS for v in models]+[priors-(original@messages.STARS)[:,None],np.std(priors,axis=1),others/(1+counts[ixuser,None]-1),
                          near/(1+sz),np.log1p(seen),np.log1p(num),dev])
            result[ret[keep]]=features[keep]
            if lo%(384*300)==0:print(phase,lo,'native error',largest,flush=True)
            continue
        out=[original,start]
        # Observe all available real peer ratings in one path and omit their values in a matched identity-only path.
        for anchors in (False,True):
            v=start
            for it in range(3):
                v=step(v,anchors)
                if it in (0,2):out.append(v)
        rng=np.random.default_rng(73411+lo)
        # Pair each random stream with its complement to reduce integration noise.
        avgprob=np.zeros_like(start)
        uniforms=[rng.random(len(i),dtype='f') for _ in range(4)]
        for anti in (False,True):
            v=start
            for t,unif in enumerate(uniforms):
                v=step(v,True,1-unif if anti else unif)
                if t>=1:avgprob+=v/6
        out.append(avgprob)
        ar=out[5]
        # Different assumed degree of dependence on the co-selected titles.
        mean=np.einsum('nr,rnk->nk',ar,enc[:,i])
        filled=np.where(visible[:,None],actual,mean)
        addition=grouped(filled)-filled-(grouped(actual)-actual)
        other=(counts[ixuser]-1-seen+visible).clip(0)
        out.append(decode(total[u]+.5*addition,uc[u]+.25*other))
        z=np.column_stack([original]+[(v-original)[:,:4] for v in out[1:]]+[v@messages.STARS for v in out]+
                          [np.log1p(seen),np.log1p(num),np.log1p(other),np.log1p(counts[ixuser]),visible.astype('f'),
                           (avgprob@messages.STARS)**2,(start@messages.STARS)**2,dev,
                           np.sum(original*np.log(np.maximum(original,1e-12)),axis=1)])
        result[ret[keep]]=z[keep]
        if lo%(384*300)==0:print(phase,lo,'native error',largest,flush=True)
    assert np.isfinite(result).all()
    return np.column_stack((base,native-base,ne.verified_rank(phase)-base,result)).astype('f')


def model():
    return CatBoostRegressor(iterations=1100,depth=5,learning_rate=.043,l2_leaf_reg=55,thread_count=2,
        random_seed=927,verbose=False,allow_writing_files=False)


def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--teacher',action='store_true');args=ap.parse_args()
    teachers=None
    if args.teacher:
        from own_joint_cohort_source import OUT as source,MODELS
        OUT=ne.ROOT/'own-cf-teacher-20261126'
        receipt=json.loads((source/'manifest.json').read_text())
        assert not receipt['query_outcomes_read'] and set(receipt['models'])==set(MODELS)
        teachers=[]
        for name in MODELS:
            row=receipt['models'][name]
            assert ne.sha(row['vector'])==row['vector_sha256']
            teachers.append(np.memmap(row['vector'],mode='r',dtype='<f4'))
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cf_adapter.cpp','-o',str(OUT/'totals.dylib')],check=True)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base,teachers)
    key,stats,held=anchor.crossfit(x,y,f,base,model);np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb,teachers)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
