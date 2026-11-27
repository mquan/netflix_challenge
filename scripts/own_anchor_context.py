"""Condition on original Probe ratings that are observable in the input training partition."""
import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from gbdt_blend import load
import own_neighbor_experiment as ne
import own_history_context as hist

OUT=ne.ROOT/'own-anchor-context-20261126'
JOINT=Path('experiments/J2-joint128-reg02.json')


def history_control():
    y,folds,p,q=hist.neighbor_control()
    report=json.loads((hist.OUT/'freeze.json').read_text())
    delta=np.load(hist.OUT/'oof.npy')
    name=report['selected']
    p=ne.clipped(p+float(name)*delta)
    np.testing.assert_allclose(ne.rmse(p,y),report['known'][name]['pooled'],atol=1e-12,rtol=0)
    path=hist.OUT/'selected.f32'
    assert ne.sha(path)==report['qualifying_hashes']['selected']
    return y,folds,p,np.fromfile(path,dtype='<f4').astype(float)


def old_probe():
    receipt=json.loads(JOINT.read_text())
    cmd=receipt['command']
    assert cmd[1]=='joint' and cmd[cmd.index('--data')+1]=='data/processed/netflix.bin' and '--init' not in cmd
    dataset=Path('data/processed/netflix.bin')
    assert ne.sha(dataset)=='08b417f7601339c95727cd02af4f839b6043c326a5fb0350245a1f6f2054c673'
    preds=receipt['predictions_artifact']
    assert ne.sha(preds['path'])==preds['sha256']
    rows=load(dataset)['probe']
    mapping=ne.ROOT/'qualifying/dataset/dense-to-raw-user.npy'
    meta=json.loads((mapping.parent/'manifest.json').read_text())
    assert ne.sha(mapping)==meta['outputs'][mapping.name]
    raw=np.load(mapping)
    dense=ne.history('trainx','uidx_to_uid')
    u=np.searchsorted(dense,raw[rows['user']])
    np.testing.assert_array_equal(dense[u],raw[rows['user']])
    i=rows['item'].astype(int)-1
    d=rows['day'].astype(int)-314
    keys=u.astype('int64')*32768+i
    assert len(np.unique(keys))==len(keys)
    return u,i,d,np.fromfile(preds['path'],dtype='<f4')


def training_indices(train,u,i,d,expected=None):
    # Search each probe key in its user's training history without reading
    # original probe labels. Only trainx/fulltrain provides observed stars.
    starts=np.r_[0,ne.history(train,'user_cnts').cumsum()]
    items=ne.history(train,'item_idxs')
    # Fulltrain is user-major but its films are not sorted within a user.
    uq=np.argsort(u,kind='stable')
    lookup=np.full(len(u),-1,dtype=np.int64)
    sorted_users=u[uq]
    for first in range(0,len(starts)-1,1024):
        end=min(first+1024,len(starts)-1)
        lo,hi=starts[first],starts[end]
        train_users=np.repeat(np.arange(first,end,dtype=np.int64),np.diff(starts[first:end+1]))
        keys=train_users*32768+items[lo:hi]
        order=np.argsort(keys,kind='stable')
        keys=keys[order]
        a,b=np.searchsorted(sorted_users,[first,end])
        targets=uq[a:b]
        wanted=u[targets].astype(np.int64)*32768+i[targets]
        pos=np.searchsorted(keys,wanted)
        ok=pos<len(keys)
        ok[ok]=keys[pos[ok]]==wanted[ok]
        lookup[targets[ok]]=lo+order[pos[ok]]
    present=lookup>=0
    np.testing.assert_array_equal(ne.history(train,'dates')[lookup[present]],d[present])
    if expected is None:
        expected=280231 if train=='trainx' else 1408395
    assert present.sum()==expected
    return present,lookup[present]


def anchor_features(train,query,p):
    u,i,d,old=old_probe()
    present,ix=training_indices(train,u,i,d)
    u,i,d,old=[a[present] for a in (u,i,d,old)]
    r=ne.history(train,'ratings')[ix].astype('f')
    parent=ne.verified_rank(train)[ix]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    nusers=len(ne.history(train,'user_cnts'))
    order=np.argsort(u,kind='stable')
    starts=np.r_[0,np.bincount(u,minlength=nusers).cumsum()]
    imp=ne.verified_rank(train,True).copy()
    imp/=np.maximum(np.linalg.norm(imp,axis=1,keepdims=True),1.e-12)
    sim=imp@imp.T
    counts=starts[qu+1]-starts[qu]
    out=np.zeros((len(qu),5,7),'f')
    for j in range(int(counts.max())):
        idx=np.flatnonzero(counts>j)
        other=order[starts[qu[idx]]+j]
        assert np.all(i[other]!=qi[idx])
        dd=np.abs(d[other]-qd[idx])
        simval=sim[qi[idx],i[other]]
        w=np.column_stack([np.ones(len(idx)),dd==0,1/(1+dd/14),np.maximum(simval,0)**2,
                           np.maximum(simval,0)**2/(1+dd/14)]).astype('f')
        vals=np.column_stack([np.ones(len(idx)),r[other]-old[other],r[other]-parent[other],
                              old[other]-p[idx],simval,np.log1p(dd),r[other]-3.6]).astype('f')
        out[idx]+=w[:,:,None]*vals[:,None,:]
    count=out[:,:,0].copy()
    out[:,:,1:]/=(.05+count[:,:,None])
    out[:,:,0]=np.log1p(count)

    # Use movie/date identifiers of all original evaluation events as
    # unlabeled context in both partitions. Exclude the target from totals.
    pu,pi,pd,_=old_probe()
    uu=np.r_[pu,ne.history('qual','user_idxs')].astype('i')
    ii=np.r_[pi,ne.history('qual','item_idxs')].astype('i')
    dd=np.r_[pd,ne.history('qual','dates')].astype('i')
    nitems=len(imp)
    pop=np.log1p(ne.history(train,'item_cnts'))
    years=ne.history(train,'item_years')
    def summarize(keys,qkeys):
        unique,rev=np.unique(keys,return_inverse=True)
        pos=np.searchsorted(unique,qkeys)
        assert np.array_equal(unique[pos],qkeys)
        n=np.bincount(rev)[pos]-1
        features=[np.log1p(n)]
        for v in (pop,years):
            total=np.bincount(rev,weights=v[ii])[pos]-v[qi]
            features.append(total/np.maximum(n,1))
        return features
    extra=summarize(uu,qu)+summarize(uu.astype('i8')*4096+dd,qu.astype('i8')*4096+qd)
    # Daily title demand among the supplied held-out movie identifiers.
    key=ii.astype('i8')*4096+dd
    unique,freq=np.unique(key,return_counts=True)
    extra.append(np.log(freq[np.searchsorted(unique,qi.astype('i8')*4096+qd)]))
    ctx=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    # Keep coverage, same-day residual, model disagreement and history size.
    x=np.column_stack([p,ctx[:,1:7],ctx[:,7],ctx[:,9],ctx[:,13],ctx[:,15],
                       out.reshape(len(qu),-1),*extra]).astype('f')
    assert np.isfinite(x).all()
    np.save(OUT/(query+'.npy'),x)
    return x


def crossfit(x,y,folds,p,make_model,weights=(.25,.5,.75,1.)):
    e=y-p
    held=np.empty_like(y)
    for f in range(5):
        tr=folds!=f
        m=make_model().fit(x[tr],e[tr])
        held[~tr]=m.predict(x[~tr],thread_count=3)
        print('fold',f,ne.rmse(ne.clipped(p[~tr]+.5*held[~tr]),y[~tr]),flush=True)
    def metrics(v):
        return dict(pooled=ne.rmse(v,y),folds=[ne.rmse(v[folds==f],y[folds==f]) for f in range(5)])
    stats={'control':metrics(p)}
    for w in weights: stats[str(w)]=metrics(ne.clipped(p+w*held))
    ok=[k for k in stats if k!='control' and stats[k]['pooled']<stats['control']['pooled']
        and sum(a<b for a,b in zip(stats[k]['folds'],stats['control']['folds']))>=4]
    return (min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'),stats,held


def model():
    return CatBoostRegressor(iterations=850,depth=5,learning_rate=.04,l2_leaf_reg=50,
                             random_seed=20261126,thread_count=3,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    y,folds,p,q=history_control()
    if (OUT/'selection.json').exists():
        report=json.loads((OUT/'selection.json').read_text())
        x=np.load(OUT/'probex.npy',mmap_mode='r')
        held=np.load(OUT/'oof.npy')
        selected=report['selected']
        if selected!='control':
            assert abs(ne.rmse(ne.clipped(p+float(selected)*held),y)-report['known'][selected]['pooled'])<1e-12
    else:
        x=anchor_features('trainx','probex',p)
        selected,stats,held=crossfit(x,y,folds,p,model)
        np.save(OUT/'oof.npy',held)
        report=dict(selected=selected,known=stats,known_comparison_is_nested=False,qualifying_hashes={},
                    qualifying_labels_used_for_fitting=False,original_probe_model=ne.sha(JOINT),
                    original_probe_model_epoch_selected_on_original_probe=True,
                    baseline_sha256=ne.sha(hist.OUT/'selected.f32'),observed_anchors_known=280231,observed_anchors_qual=1408395)
        (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=anchor_features('fulltrain','qual',q)
        m=model().fit(x,y-p)
        delta=m.predict(qx,thread_count=3)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control': ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
