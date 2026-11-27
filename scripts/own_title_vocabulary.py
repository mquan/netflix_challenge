"""Predict user-specific preferences for words and phrases in the original Netflix catalog."""
import ctypes
import json
import subprocess
from collections import Counter
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_title_neighbors as titles
import own_transition as transition

OUT=ne.ROOT/'own-title-vocabulary-20261126'
PARENT=blend.OUT/'paired-03'


def vocabulary():
    records=[line.split(',',2) for line in titles.TITLES.read_text(encoding='latin1').splitlines()]
    np.testing.assert_array_equal([int(r[0]) for r in records],ne.history('trainx','iidx_to_iid'))
    docs=[]
    for _,_,title in records:
        terms=titles.words(title)
        keys={(0,t) for t in terms}|{(1,a+' '+b) for a,b in zip(terms,terms[1:])}
        normalized=' '.join(titles.words(title.split(':')[0]))
        if normalized:keys.add((1,normalized))
        for t in ('season','bonus','series','vol','live','special edition','documentary','concert','collection','workout','opera','ballet'):
            if t in title.lower():keys.add((2,t))
        docs.append(sorted(keys))
    freq=Counter(key for doc in docs for key in doc)
    vocab={key:i for i,key in enumerate(k for k,n in freq.items() if 2<=n<=3500)}
    terms=[[vocab[k] for k in doc if k in vocab] for doc in docs]
    return (np.r_[0,np.cumsum([len(t) for t in terms])].astype('i'),np.array([i for t in terms for i in t],'i'),
            np.array([k[0] for k in vocab],'i1'),np.array([np.log1p(len(records)/freq[k]) for k in vocab],'f'))


def features(train,phase,b,vocab,lib):
    err,_=transition.loo(train);uc=ne.history(train,'user_cnts');off=np.r_[0,uc.cumsum()].astype('i')
    i,r=[ne.history(train,k) for k in ('item_idxs','ratings')]
    qu,qi=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
    iu,ii=qu,qi
    if phase=='probex':iu=np.r_[iu,ne.history('qual','user_idxs')];ii=np.r_[ii,ne.history('qual','item_idxs')]
    def groups(u):return np.r_[0,np.bincount(u,minlength=len(uc)).cumsum()].astype('i'),np.argsort(u,kind='stable').astype('i')
    qo,qr=groups(qu);io,ir=groups(iu);im=np.ascontiguousarray(ii[ir],'i2')
    out=np.empty((len(b),24),'f');args=(off,i,r,err,*vocab,qo,qr,qi,io,im,out)
    assert all(x.flags.c_contiguous for x in args)
    lib.title_taste(len(uc),len(vocab[-1]),*[x.ctypes.data for x in args])
    mask=np.max(out[:,[0,4,8,12,16,20]],axis=1)>0
    return np.column_stack((out[mask],b[mask],(full.cf96(phase)[mask]-b[mask]),np.log1p(uc[qu[mask]]))).astype('f'),mask


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_title_vocabulary.cpp','-o',str(OUT/'taste.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'taste.dylib').resolve()));lib.title_taste.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*14;lib.title_taste.restype=None
    vocab=vocabulary();y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    x,mask=features('trainx','probex',b,vocab,lib);held=np.zeros(len(y),'f');target=(y-b)[mask];models=[]
    for k in range(5):
        tr=f[mask]!=k;m=CatBoostRegressor(iterations=450,depth=5,l2_leaf_reg=90,learning_rate=.035,verbose=False,thread_count=3,random_seed=7431+k,allow_writing_files=False)
        m.fit(x[tr],target[tr]);held[mask&(f==k)]=m.predict(x[~tr],thread_count=3);models.append(m);print('vocabulary fold',k,flush=True)
    stats={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];choice=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=stats,vocabulary_size=len(vocab[-1]),catalog_sha256=ne.sha(titles.TITLES),support_probex=int(mask.sum()),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx,qmask=features('fulltrain','qual',qb,vocab,lib);delta=np.zeros(len(qb),'f')
        delta[qmask]=sum(m.predict(qx,thread_count=3) for m in models)/5
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
