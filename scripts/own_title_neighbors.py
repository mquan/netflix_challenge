"""Learn residual relationships among series, sequels, and related titles using the original Netflix catalog and ratings."""
import ctypes
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from collections import Counter
import numpy as np
from scipy import sparse
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as transition
OUT=ne.ROOT/'own-title-neighbors-20261126'
PARENT=blend.OUT/'schema-08'
TITLES=Path('data/raw/extracted/download/movie_titles.txt')
K=40
STOP=set('a an the in on of for is at to and by from with as it its i s ii iii iv v vi vii viii ix x part vol volume season seasons disc disk series edition special collection bonus material complete episodes chapter widescreen extended director cut original best greatest hits live'.split())


def words(title):
    title=unicodedata.normalize('NFKD',title.casefold()).encode('ascii','ignore').decode()
    return [w for w in re.findall('[a-z]+',title) if w not in STOP and len(w)>1]


def catalog():
    records=[line.split(',',2) for line in TITLES.read_text(encoding='latin1').splitlines()]
    ids=np.array([int(r[0]) for r in records]);np.testing.assert_array_equal(ids,ne.history('trainx','iidx_to_iid'))
    rows=[];roots=[];fulls=[];props=[]
    for _,_,title in records:
        w=words(title);base=words(title.split(':')[0])
        sub=re.split(r'\b(?:season|vol(?:ume)?|disc|bonus material|special edition|part)\b',title,flags=re.I)[0]
        head=words(sub);st=' '.join(head)
        features={'w '+z:1. for z in w}
        for a,b in zip(w,w[1:]):features['b '+a+' '+b]=1.25
        if base:features['prefix '+' '.join(base)]=2.2
        if head:features['sub '+st]=2.5
        for matched in range(2,min(4,len(w))+1):features['first '+' '.join(w[:matched])]=1.5
        rows.append(features);roots.append(' '.join(base));fulls.append(st)
        props.append([('season' in title.lower()),('bonus' in title.lower()),bool(re.search(r'\b(?:part|vol|ii|iii|iv|[2-9])\b',title,re.I))])
    counts=Counter(k for row in rows for k in row);vocab={k:i for i,k in enumerate(k for k,v in counts.items() if 1<v<500)}
    col=[];row=[];val=[]
    for i,features in enumerate(rows):
        for k,v in features.items():
            if k in vocab:row.append(i);col.append(vocab[k]);val.append(v*np.log1p(len(rows)/counts[k]))
    x=sparse.csr_matrix((np.array(val,'f'),(row,col)),shape=(len(rows),len(vocab)))
    norm=np.sqrt(x.multiply(x).sum(1).A1).clip(1);x=sparse.diags(1/norm)@x
    mat=(x@x.T).tocsr();g=cov.movie_geometry();neighbors=np.full((len(rows),K),-1,'i');bands=np.zeros(neighbors.shape,'i1');aff=np.zeros(neighbors.shape,'f')
    for m in range(len(rows)):
        start,end=mat.indptr[m:m+2];ii=mat.indices[start:end];v=mat.data[start:end];keep=(ii!=m)&(v>.17)
        ii=ii[keep];v=v[keep]
        if not len(ii):continue
        exact=np.array([bool(fulls[m]) and fulls[m]==fulls[j] for j in ii])
        family=np.array([bool(roots[m]) and roots[m]==roots[j] for j in ii])
        paired=g[ii]@g[m];score=v*(.75+.25*paired.clip(-1,1))
        yes=exact|family|(score>.42)|((score>.23)&(paired>.30))
        ii,v,exact,family,score=[a[yes] for a in (ii,v,exact,family,score)]
        rank=np.argsort(-(score+2*exact+family),kind='stable')[:K];n=len(rank)
        neighbors[m,:n]=ii[rank];bands[m,:n]=np.where(exact[rank],0,np.where(family[rank]|(score[rank]>.64),1,2))
        aff[m,:n]=np.maximum(score[rank],.7*family[rank]+.2*exact[rank])
    np.savez(OUT/'catalog.npz',neighbors=neighbors,bands=bands,affinity=aff)
    report=dict(original_catalog=str(TITLES),catalog_sha256=ne.sha(TITLES),derived_sha256=ne.sha(OUT/'catalog.npz'),
        directed_relationships=int((neighbors>=0).sum()),titles_with_neighbors=int((neighbors[:,0]>=0).sum()),
        learned_geometry='own-session-latent-20261126/graph-coordinates.npy')
    (OUT/'catalog.json').write_text(json.dumps(report,indent=2)+'\n')
    return neighbors,bands,aff,np.array(props,'f')


def generate(train,phase,lib,catalog,errors=None):
    nb,band,aff,props=catalog
    err=transition.loo(train)[0] if errors is None else errors
    i,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')];uc=ne.history(train,'user_cnts');off=np.r_[0,uc.cumsum()].astype('i')
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    iu,ii=qu,qi
    if phase=='probex':iu=np.r_[iu,ne.history('qual','user_idxs')];ii=np.r_[ii,ne.history('qual','item_idxs')]
    year=np.ascontiguousarray(ne.history(train,'item_years'),'i2');sort=np.argsort(iu,kind='stable');items=np.ascontiguousarray(ii[sort],'i2')
    ioff=np.r_[0,np.bincount(iu,minlength=len(uc)).cumsum()].astype('i')
    qoff=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()].astype('i');qr=np.argsort(qu,kind='stable').astype('i')
    out=np.empty((len(qi),36),'f');args=(off,i,d,r,err,nb,band,aff,year,qoff,qr,qi,qd,ioff,items,out)
    lib.title_neighbors(len(uc),len(nb),*[v.ctypes.data for v in args]);assert np.isfinite(out).all()
    return out,props[qi]


def main(paired=False):
    global OUT,PARENT
    if paired:
        OUT=ne.ROOT/'own-paired-title-neighbors-20261126';PARENT=blend.OUT/'paired-01'
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_title_neighbors.cpp',
                                             '-o',str(OUT/'titles.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'titles.dylib').resolve()));lib.title_neighbors.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*16;lib.title_neighbors.restype=None
    cat=catalog();y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def features(train,phase,b):
        raw,props=generate(train,phase,lib,cat);ok=raw[:,[0,12,24,10,22,34]].max(1)>0
        return np.column_stack((raw[ok],props[ok],b[ok],full.cf96(phase)[ok]-b[ok],ne.verified_rank(phase)[ok]-b[ok])).astype('f'),ok
    def model():return CatBoostRegressor(iterations=650,depth=5,learning_rate=.04,l2_leaf_reg=85,thread_count=2,
                                          random_seed=71799,verbose=False,allow_writing_files=False)
    x,mask=features('trainx','probex',b);held=np.zeros(len(y),'f');models=[];fm=f[mask]
    for k in range(5):
        tr=fm!=k;m=model().fit(x[tr],(y-b)[mask][tr]);models.append(m);held[mask&(f==k)]=m.predict(x[~tr],thread_count=2)
    stats={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];key=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=stats,covered_probe_rows=int(mask.sum()),catalog_sha256=ne.sha(OUT/'catalog.json'),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx,qmask=features('fulltrain','qual',qb);delta=np.zeros(len(qb),'f');delta[qmask]=sum(m.predict(qx,thread_count=2) for m in models)/5
        rep['covered_qualifying_rows']=int(qmask.sum());np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--paired',action='store_true');main(parser.parse_args().paired)
