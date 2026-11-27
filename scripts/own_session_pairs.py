"""Estimate same-day co-rating effects from train-only outcomes and unlabeled movie identities."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_full_context as full
import own_anchor_context as anchor
import own_history_context as hist

OUT=ne.ROOT/'own-session-pairs-20261126'


def make_events(train,query,with_keys=False):
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    re=r-ne.verified_rank(train)
    ce=r-full.cf96(train)
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    if query=='probex':
        extra=[ne.history('qual',k) for k in ('user_idxs','item_idxs','dates')]
        au,ai,ad=[np.r_[x,z] for x,z in zip((qu,qi,qd),extra)]
    else: au,ai,ad=qu,qi,qd
    n=len(r)+len(au)
    assert n==103297638
    items=np.empty(n,'i2')
    ratings=np.zeros(n,'i1')
    er,ec=[np.zeros(n,'f') for _ in range(2)]
    es=np.empty(n,'i')
    qsession=np.full(len(qu),-1,'i')
    event_for_train=np.empty(len(r),'i')
    us=np.r_[0,ne.history(train,'user_cnts').cumsum()]
    ids=np.argsort(au,kind='stable')
    au_sorted=au[ids]
    pointers=[]
    session_keys=[]
    ns=0
    for first in range(0,len(us)-1,1024):
        last=min(len(us)-1,first+1024)
        a,b=us[first],us[last]
        g,h=np.searchsorted(au_sorted,[first,last])
        ix=ids[g:h]
        start=a+g
        combined_u=np.r_[u[a:b],au[ix]].astype('int64')
        combined_d=np.r_[d[a:b],ad[ix]]
        combined_i=np.r_[i[a:b],ai[ix]]
        key=combined_u*4096+combined_d
        order=np.lexsort((combined_i,key))
        sorted_key=key[order]
        begins=np.flatnonzero(np.r_[True,sorted_key[1:]!=sorted_key[:-1]])
        if with_keys:session_keys.append(sorted_key[begins])
        pointers.append((start+begins).astype('i'))
        sids=np.cumsum(np.r_[True,sorted_key[1:]!=sorted_key[:-1]],dtype='i')+(ns-1)
        output=slice(start,start+len(order))
        items[output]=combined_i[order]
        es[output]=sids
        known=order<b-a
        pos=np.flatnonzero(known)+start
        rr=order[known]+a
        event_for_train[rr]=pos
        ratings[pos]=r[rr]
        er[pos]=re[rr]
        ec[pos]=ce[rr]
        others=ix[order[~known]-(b-a)]
        target=others<len(qu)
        qsession[others[target]]=sids[~known][target]
        ns+=len(begins)
    assert np.all(qsession>=0)
    off=np.r_[np.concatenate(pointers),np.int32(n)]
    print(train,'sessions',ns,'pair work',sum(np.diff(off).astype('int64')**2),flush=True)
    result=event_for_train,qsession,es,off,items,ratings,er,ec
    return result+(np.concatenate(session_keys),) if with_keys else result


def features(train,query,p):
    lib=ctypes.CDLL(str((OUT/'pairs.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.session_pairs.argtypes=[it]*3+[pt]*13
    lib.group_items.restype=lib.session_pairs.restype=None
    ei,qsession,es,off,items,ratings,er,ec=make_events(train,query)
    ic=ne.history(train,'item_cnts')
    starts=np.r_[0,ic.cumsum()].astype('i')
    ti=ne.history(train,'item_idxs')
    rows=np.empty(len(ti),'i')
    lib.group_items(len(ti),ti.ctypes.data,starts.ctypes.data,len(ic),rows.ctypes.data)
    qi=ne.history(query,'item_idxs')
    qo=np.argsort(qi,kind='stable').astype('i')
    qs=np.r_[0,np.bincount(qi,minlength=len(ic)).cumsum()].astype('i')
    pairs=np.empty((len(qi),27),'f')
    args=[v.ctypes.data for v in (starts,rows,ei,qs,qo,qsession,es,off,items,ratings,er,ec,pairs)]
    def block(i):lib.session_pairs(i,min(i+32,len(ic)),len(ic),*args)
    started=time.monotonic()
    with ThreadPoolExecutor(3) as pool:
        for j,_ in enumerate(pool.map(block,range(0,len(ic),32))):
            if j%100==0:print(query,j*32,round(time.monotonic()-started),flush=True)
    ctx=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    x=np.column_stack((p,ctx[:,1:8],ctx[:,13:16],pairs)).astype('f')
    assert np.isfinite(x).all()
    np.save(OUT/(query+'.npy'),x)
    return x


def model():
    return CatBoostRegressor(iterations=950,depth=5,learning_rate=.04,l2_leaf_reg=40,
                             verbose=False,thread_count=3,random_seed=20261126,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_session_pairs.cpp','scripts/own_neighbor_features.cpp','-o',str(OUT/'pairs.dylib')],check=True)
    y,folds,p,q=anchor.history_control()
    x=features('trainx','probex',p)
    selected,stats,held=anchor.crossfit(x,y,folds,p,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,
                known_comparison_is_nested=False,baseline_sha256=ne.sha(hist.OUT/'selected.f32'),
                implicit_rows=103297638,known_training_ratings=99352343,qual_training_ratings=100480507)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=features('fulltrain','qual',q)
        delta=model().fit(x,y-p).predict(qx,thread_count=3)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
