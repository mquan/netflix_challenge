"""Train on catalog series to identify related films from observed co-ratings, then forecast the target user's residual."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_title_neighbors as titles
import own_transition as transitions
from own_query_sets import projection
OUT=ne.ROOT/'own-learned-series-20261126'
PARENT=blend.OUT/'paired-01'
WIDTH=100


def inputs(train,phase,base,lib,lexical):
    err,_=transitions.loo(train);nb,band,_,props=lexical
    s=cf.checkpoint(train);directions=[cov.movie_geometry(),projection(s['a'].mean(0),32),projection(s['p'].mean(0),32)]
    directions=[np.ascontiguousarray(v/np.linalg.norm(v,axis=1,keepdims=True).clip(.001),'f') for v in directions]
    n=len(nb);candidates=np.full((n,WIDTH),-1,'i');types=np.full((n,WIDTH),-1,'i1')
    for j in range(0,n,128):
        sl=slice(j,min(j+128,n));sims=[g[sl]@g.T for g in directions];cols=[np.argpartition(-v,49,axis=1)[:,:50] for v in sims]
        for row,m in enumerate(range(sl.start,sl.stop)):
            # Include lexical positives to train the related-title classifier, with the same physical statistics as other edges.
            known={k:int(t) for k,t in zip(nb[m],band[m]) if k>=0}
            all_ids=np.unique(np.concatenate([list(known)]+[a[row] for a in cols])).astype('i');all_ids=all_ids[all_ids!=m]
            score=.48*sims[0][row,all_ids]+.23*sims[1][row,all_ids]+.29*sims[2][row,all_ids]
            order=sorted(set(known)|set(all_ids[np.argsort(-score)[:WIDTH]]),key=lambda k:(k not in known,-float(.48*sims[0][row,k]+.23*sims[1][row,k]+.29*sims[2][row,k])))[:WIDTH]
            candidates[m,:len(order)]=order;types[m,:len(order)]=[known.get(k,-1) for k in order]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')];off=np.r_[0,uc.cumsum()].astype('i')
    item,day,star=[ne.history(train,k) for k in ('item_idxs','dates','ratings')];stats=np.zeros((n,WIDTH,10),'f')
    args=(off,item,day,star,err,candidates,stats)
    lib.edge_moments(len(uc),n,WIDTH,*[v.ctypes.data for v in args])
    ids=candidates.clip(0);years=ne.history(train,'item_years').astype('f')
    x=np.empty((n,WIDTH,17),'f')
    for j in range(0,n,128):
        sl=slice(j,min(j+128,n));ids0=ids[sl];z=stats[sl];count=z[:,:,0];src=ic[sl,None];dst=ic[ids0]
        ss=[np.einsum('nd,nad->na',g[sl],g[ids0]) for g in directions]
        values=[*ss,np.log1p(count),np.log1p(np.broadcast_to(src,count.shape)),np.log1p(dst),
            np.log((count+1)*len(uc)/(np.maximum(1,src)*np.maximum(1,dst).astype(float))),count/(12+np.minimum(src,dst)*.5),
            z[:,:,1]/(4+count),z[:,:,2]/(4+count),z[:,:,3]/(4+count),z[:,:,4]/np.sqrt((15+z[:,:,5])*(15+z[:,:,6])),
            z[:,:,7]/(6+count),z[:,:,8]/(6+count),z[:,:,9]/(6+count),
            np.abs(years[sl,None]-years[ids0]).clip(0,60)/20,(props[sl,None,0]+props[ids0,0])*.5]
        x[sl]=np.stack(values,axis=2)
    positive=(types==0)|(types==1)
    # Split by target film. Even the classifier's ratings-free supervision on film titles stays out of its held target films.
    fold=np.random.default_rng(16841).integers(0,4,n);likelihood=np.zeros((n,WIDTH),'f');valid=candidates>=0
    rng=np.random.default_rng(71881)
    for k in range(4):
        train_rows=(fold[:,None]!=k)&valid&(positive|(rng.random(valid.shape)<.25))
        model=CatBoostClassifier(iterations=320,depth=5,learning_rate=.055,l2_leaf_reg=35,thread_count=2,random_seed=4783+k,
                                 verbose=False,allow_writing_files=False)
        model.fit(x[train_rows],positive[train_rows]);va=(fold[:,None]==k)&valid
        likelihood[va]=model.predict_proba(x[va],thread_count=2)[:,1]
    chosen=np.full((n,40),-1,'i');group=np.zeros((n,40),'i1');weight=np.zeros((n,40),'f')
    for m in range(n):
        take=np.flatnonzero(valid[m]&(types[m]<0)&(likelihood[m]>.10)&(stats[m,:,0]>=5))
        ranked=take[np.argsort(-likelihood[m,take],kind='stable')[:40]];pr=likelihood[m,ranked];count=len(pr)
        chosen[m,:count]=candidates[m,ranked];group[m,:count]=np.where(pr>.65,0,np.where(pr>.3,1,2));weight[m,:count]=np.sqrt(pr)
    np.savez(OUT/(phase+'-neighbors.npz'),neighbors=chosen,group=group,weight=weight)
    print(phase,'new rating-derived neighbors',int((chosen>=0).sum()),flush=True)
    raw,props=titles.generate(train,phase,lib,(chosen,group,weight,props),err)
    mask=raw[:,[0,12,24,10,22,34]].max(1)>0
    return np.column_stack((raw[mask],props[mask],base[mask],full.cf96(phase)[mask]-base[mask],ne.verified_rank(phase)[mask]-base[mask])).astype('f'),mask


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_learned_series.cpp','-o',str(OUT/'series.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'series.dylib').resolve()));lib.edge_moments.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*7;lib.edge_moments.restype=None
    lib.title_neighbors.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*16;lib.title_neighbors.restype=None
    titles.OUT=OUT;lexical=titles.catalog();y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    x,mask=inputs('trainx','probex',b,lib,lexical);held=np.zeros(len(y),'f');models=[];fm=f[mask]
    for k in range(5):
        tr=fm!=k;m=CatBoostRegressor(iterations=560,depth=5,learning_rate=.04,l2_leaf_reg=92,thread_count=2,
                                     random_seed=3917+k,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-b)[mask][tr]);models.append(m);held[mask&(f==k)]=m.predict(x[~tr],thread_count=2)
    scores={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in scores if ret.qualifies(scores[k],scores['0'])];key=min(good,key=lambda k:scores[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=scores,covered_probe_rows=int(mask.sum()),catalog_sha256=ne.sha(OUT/'catalog.json'),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        qx,qmask=inputs('fulltrain','qual',qb,lib,lexical);delta=np.zeros(len(qb),'f');delta[qmask]=sum(m.predict(qx,thread_count=2) for m in models)/5
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
