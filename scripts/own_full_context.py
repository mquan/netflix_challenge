"""Build verified own-model features and cross-fit a new residual layer on user folds."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_history_context as hist
import own_anchor_context as anchor

OUT=ne.ROOT/'own-full-context-20261126'
UP=Path('../netflix_upstream_repro/artifacts')
OWN=UP/'upstream-own17-remote-context-handoff'
CFG={'shallow':(5,1800,.035),'deep':(8,1600,.035)}
RECIPE_SHA='a209a4d951f8e747743074d29938fcf83595b009d5dfc2df307b27385eebf5f4'


def cf96(split):
    receipt=json.loads((OWN/'own-cf96-terminal-pair-20261124.json').read_text())
    name='cfnade-96.'+split+'.npy'
    target=next(a for a in receipt['assets'] if a['name']==name)
    sub='reference' if split in ('probex','qual') else 'preds_new'
    path=UP/'upstream-cf96-training-export'/sub/name
    assert ne.sha(path)==target['sha256']
    return np.load(path,mmap_mode='r')


def models(query,p):
    phase='known' if query=='probex' else 'qual'
    raw=subprocess.check_output(['git','show','5dfb573:experiments/remote-context-blend/bank-own17x.json'])
    assert hashlib.sha256(raw).hexdigest()==RECIPE_SHA
    bank=json.loads(raw)
    ownmeta=OWN/'own17-bank-columns.json'
    assert ne.sha(ownmeta)==bank['own17_assets']['columns_json']
    info=json.loads(ownmeta.read_text())
    ownpath=OWN/f'own17-bank-{phase}.npy'
    assert ne.sha(ownpath)==bank['own17_assets'][phase]
    x=np.load(ownpath,mmap_mode='r')
    names=[]
    columns=[]
    for j,c in enumerate(info['columns']):
        assert 'own17:'+c['name']==bank['columns'][j]['column']
        assert hashlib.sha256(np.ascontiguousarray(x[:,j])).hexdigest()==c[phase+'_sha256']
        names.append('own17-'+c['name'])
        columns.append((x[:,j]-p).astype('f'))
    for c in bank['columns'][17:]:
        if c['kind']!='split-new trainx->probex' or c['map'] is not None: continue
        known=Path(c['known'])
        qual=Path(c['qual'])
        if not known.is_file() or not qual.is_file():continue
        if any(ne.sha(Path(c[t]))!=c[t+'_sha256'] for t in ('known','qual')):
            print('Omitting non-matching source pair',c['column'],flush=True)
            continue
        path=Path(c[phase])
        names.append(c['column'])
        columns.append((np.load(path,mmap_mode='r')-p).astype('f'))
    from own17x_cf96_followup import manifest
    members=manifest(ne.BASE,'cf-members.sha256')
    for suffix in ['knn3','knn3d','knn-25c','knn-25m']:
        name='cfnade-96__'+suffix
        rel='cf96-chains/'+name+'.'+query+'.npy'
        path=ne.BASE/'cf'/rel
        assert ne.sha(path)==members[rel]
        names.append(name)
        columns.append((np.load(path)-p).astype('f'))
    names.append('cfnade-96')
    columns.append((cf96(query)-p).astype('f'))
    return names,np.column_stack(columns)


def derived(train,query):
    lib=ctypes.CDLL(str((OUT/'features.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.film_context.argtypes=[it,it]+[pt]*16
    lib.group_items.argtypes=[it,pt,pt,it,pt]
    lib.user_context.argtypes=[it]+[pt]*9+[it,pt]
    for f in [lib.film_context,lib.user_context,lib.group_items]:f.restype=None
    def ptrs(*vs):return [v.ctypes.data for v in vs]
    us,itms,ds,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    ur=np.bincount(us,weights=r,minlength=len(uc))
    ustarts=np.r_[0,uc.cumsum()].astype('i')
    starts=np.r_[0,ic.cumsum()].astype('i')
    order=np.empty(len(r),'i')
    lib.group_items(len(r),*ptrs(itms,starts),len(ic),order.ctypes.data)
    qs=np.r_[0,np.bincount(qi,minlength=len(ic)).cumsum()].astype('i')
    qo=np.argsort(qi,kind='stable').astype('i')
    rp,cp=ne.verified_rank(train),cf96(train)
    qr,qc=ne.verified_rank(query),cf96(query)
    film=np.empty((len(qi),63),'f')
    lib.film_context(len(ic),max(int(ds.max()),int(qd.max()))+1,
                     *ptrs(starts,order,qs,qo,us,ds,r,rp,cp,uc,ur,qu,qd,qr,qc,film))
    del order
    v=ne.verified_rank(train,True).copy()
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    sims=v@v.T
    ec=r-cp
    ctx=np.empty((len(qi),54),'f')
    def block(lo):
        sl=slice(lo,min(len(qi),lo+8192))
        lib.user_context(len(qi[sl]),*ptrs(qu[sl],qi[sl],qd[sl],ustarts,itms,ds,r,ec,sims),len(ic),ctx[sl].ctypes.data)
    with ThreadPoolExecutor(3) as ex:list(ex.map(block,range(0,len(qi),8192)))
    # User count/rating histograms already appear in the earlier context.
    keep=[6*j+k for j in (0,1,2,5,6,7,8) for k in (2,3)]
    return film,ctx[:,keep]


def prepare(query,p):
    train='trainx' if query=='probex' else 'fulltrain'
    names,pred=models(query,p)
    film,cfcontext=derived(train,query)
    history=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    anchorpath=anchor.OUT/(query+'.npy')
    a=np.load(anchorpath,mmap_mode='r') if anchorpath.exists() else anchor.anchor_features(train,query,p)
    x=np.lib.format.open_memmap(OUT/(query+'.npy'),mode='w+',dtype='f',shape=(len(p),1+pred.shape[1]+film.shape[1]+cfcontext.shape[1]+history.shape[1]+a.shape[1]-11))
    pos=0
    for chunk in [p[:,None],pred,film,cfcontext,history,a[:,11:]]:
        x[:,pos:pos+chunk.shape[1]]=chunk
        pos+=chunk.shape[1]
    assert pos==x.shape[1]
    x.flush()
    for j in range(0,len(x),32768): assert np.isfinite(x[j:j+32768]).all()
    (OUT/(query+'-sources.json')).write_text(json.dumps(dict(bank_sha256=RECIPE_SHA,columns=names,shape=x.shape,
        cf96_receipt=ne.sha(OWN/'own-cf96-terminal-pair-20261124.json'),feature_sha256=ne.sha(OUT/(query+'.npy'))),indent=2)+'\n')
    return x


def model(key,threads):
    depth,n,lr=CFG[key]
    return CatBoostRegressor(iterations=n,depth=depth,learning_rate=lr,l2_leaf_reg=40,
        random_seed=20261126,thread_count=threads,rsm=.75,border_count=96,verbose=False,allow_writing_files=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','shallow','deep'])
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    y,folds,p,q=anchor.history_control()
    if args.stage=='prepare':
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_film_context.cpp','scripts/own_neighbor_features.cpp','-o',str(OUT/'features.dylib')],check=True)
        for query,base in [('probex',p),('qual',q)]:
            receipt=OUT/(query+'-sources.json')
            if receipt.exists() and ne.sha(OUT/(query+'.npy'))==json.loads(receipt.read_text())['feature_sha256']:
                continue
            prepare(query,base)
            print('prepared',query,flush=True)
        return
    x=np.load(OUT/'probex.npy',mmap_mode='r')
    out=OUT/args.stage
    out.mkdir(exist_ok=True)
    def make():return model(args.stage,args.threads)
    selected,stats,held=anchor.crossfit(x,y,folds,p,make)
    np.save(out/'oof.npy',held)
    report=dict(selected=selected,known=stats,known_comparison_is_nested=False,qualifying_hashes={},
                qualifying_labels_used_for_fitting=False,baseline_sha256=ne.sha(hist.OUT/'selected.f32'))
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        m=make().fit(x,y-p)
        qx=np.load(OUT/'qual.npy',mmap_mode='r')
        delta=np.empty(len(q))
        for i in range(0,len(q),65536):
            sl=slice(i,i+65536)
            delta[sl]=m.predict(qx[sl],thread_count=args.threads)
        np.save(out/'qual-correction.npy',delta)
        path=out/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
