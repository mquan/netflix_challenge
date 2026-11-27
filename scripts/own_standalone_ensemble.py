"""Cross-fit a first-level rating predictor on raw local forecasters only, with a new partition of users."""
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_day_states as days
import own_extra_joint as extra
import own_full_context as full
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as peer_cf
import own_peer_replay as history
import own_peer_temporal as temporal
import own_rbm_joint as rbm
import own_retained_history as ret
OUT=ne.ROOT/'own-standalone-ensemble-20261126'
PARENT=blend.OUT/'paired-03'
STAR=np.arange(1,6,dtype='f')


class Role:
    def __init__(self,phase):
        self.phase=phase;which=phase=='qual';train='fulltrain' if which else 'trainx'
        self.index=np.arange(1408395,1408395+2817131) if which else ret.map_rows()
        self.mode=2 if which else 1;self.cf=full.cf96(phase);self.rank=ne.verified_rank(phase);self.files={}
        self.u,self.m,self.day=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc,self.ic=[np.log1p(ne.history(train,k)) for k in ('user_cnts','item_cnts')]
        self.year=ne.history(train,'item_years')
        def checked(path,sha):
            assert days.sha(path)==sha;self.files[str(path)]=sha
            return np.load(path,mmap_mode='r') if Path(path).suffix=='.npy' else np.memmap(path,'<f4','r')
        self.cprob=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')
        np.testing.assert_allclose(self.cprob[::31,7:12]@STAR,self.cf[::31],atol=.0003)
        self.raw=[]
        for family in ('temporal-attention','suffix-attention','cf-temporal-attention','temporal-continuation','hybrid-attention'):
            folder=ne.ROOT/('own-'+family+'-20261126');rec=json.loads((folder/'freeze.json').read_text())
            assert not rec['qualifying_labels_used_for_fitting'];path=folder/(phase+'.npy');digest=days.sha(path);self.raw.append(checked(path,digest))
        self.hist=checked(history.OUT/'effects.npy',json.loads((history.OUT/'manifest.json').read_text())['output_sha256'])
        self.p=checked(peer_cf.OUT/'probabilities.npy',json.loads((peer_cf.OUT/'manifest.json').read_text())['probabilities_sha256'])
        self.tm=checked(temporal.OUT/'means.npy',json.loads((temporal.OUT/'manifest.json').read_text())['means_sha256'])
        jr=json.loads((joint.OUT/'manifest.json').read_text());er=json.loads((extra.OUT/'manifest.json').read_text())
        self.j=[checked(v['vector'],v['vector_sha256']) for v in list(jr['models'].values())+list(er['inputs'].values())]
        info=json.loads((rbm.OUT/'freeze.json').read_text())['signals']
        self.rbm=[checked(rbm.OUT/(n+'.f32'),info[str(rbm.OUT/(n+'.f32'))]).reshape(-1,7) for n in rbm.MODELS]
        self.other=[]
        for family,stride in (('choice-factors',3),('suffix-factors',1),('sparse-cohort',1)):
            folder=ne.ROOT/('own-'+family+'-20261126');rec=json.loads((folder/'freeze.json').read_text());assert not rec['qualifying_labels_used_for_fitting']
            path=folder/(phase+'.npy');self.other.append(checked(path,days.sha(path))[:,::stride])
    def block(self,rows):
        idx=self.index[rows];mode=self.mode;w=self.cf[rows];j=np.column_stack([v[idx] for v in self.j]);center=j.mean(1)
        raw=np.column_stack([v[rows] for v in self.raw]);h=self.hist[idx,:,mode];h0=self.hist[idx,:,0];p0=self.p[idx,0].astype('f');p=self.p[idx,mode].astype('f')
        ts=self.tm[idx];rb=[v[idx] for v in self.rbm];cf=self.cprob[rows]
        parts=[w,self.rank[rows]-w,raw-w[:,None],raw.std(1),j-center[:,None],center-w,h-w[:,None],h-h0,p0,p-p0,
               ts[:,mode]-w,ts[:,mode]-ts[:,0],cf[:,7:12],cf[:,37:42],cf[:,2:7]-w[:,None],self.day[rows]/2000,
               self.uc[self.u[rows]],self.ic[self.m[rows]],self.year[self.m[rows]]/100]
        for a in rb:parts.extend((a[:,0]-w,a[:,1:5]-a[:,:1],a[:,5:]))
        return np.column_stack((*parts,*[a[rows] for a in self.other])).astype('f')
    def table(self):
        a=np.empty((len(self.cf),self.block(np.arange(2)).shape[1]),'f')
        for j in range(0,len(a),16384):
            sl=slice(j,j+16384);a[sl]=self.block(sl)
        return a


def mixed_folds(u):
    x=u.astype('uint64')+0x221234
    x=(x^(x>>16))*0x45d9f3b;x=(x^(x>>16))*0x45d9f3b
    return ((x^(x>>16))%5).astype('i1')


def main():
    OUT.mkdir(exist_ok=True);y,f,_,_=anchor.history_control();r=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==r['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    d=Role('probex');sources={'probex':d.files};x=d.table();hold=mixed_folds(d.u);raw=np.empty(len(y),'f');models=[]
    fold_counts=np.array([[int(((f==j)&(hold==k)).sum()) for k in range(5)] for j in range(5)]);assert fold_counts.min()>30000
    for k in range(5):
        tr=hold!=k;m=CatBoostRegressor(iterations=1050,depth=6,learning_rate=.05,l2_leaf_reg=72,random_strength=1.3,thread_count=2,
                                        random_seed=14371+k,verbose=False,allow_writing_files=False)
        m.fit(x[tr],(y-d.cf)[tr]);models.append(m);raw[~tr]=d.cf[~tr]+m.predict(x[~tr],thread_count=2)
        print('independent user fold',k,flush=True)
    delta=(raw-b);scale=np.zeros(len(y))
    def weight(mask):return float(np.clip(delta[mask]@(y-b)[mask]/(delta[mask]@delta[mask]+40),0,1))
    for k in range(5):scale[f==k]=weight(f!=k)
    w=weight(np.ones(len(y),bool));known=blend.metrics(ne.clipped(b+scale*delta),y,f);ctrl=blend.metrics(b,y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',weight=w,known=known,control=ctrl,
             standalone_probe=blend.metrics(raw,y,f),user_fold_cross_counts=fold_counts.tolist(),raw_sources=sources,
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del x,d
        d=Role('qual');sources['qual']=d.files;pred=np.empty(len(qb),'f')
        for j in range(0,len(qb),32768):
            sl=slice(j,j+32768);v=d.block(sl);pred[sl]=d.cf[sl]+sum(m.predict(v,thread_count=2) for m in models)/5
        qd=pred-qb;np.save(OUT/'oof.npy',scale*delta);np.save(OUT/'qual-correction.npy',w*qd)
        ne.clipped(qb+w*qd).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
