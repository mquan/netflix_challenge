"""Fit a held-user residual tree ensemble on frozen expert contributions and observable-peer forecast changes."""
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_day_states as daystates
import own_extra_joint as extra
import own_full_context as full
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as cf
import own_peer_replay as hn
import own_rbm_joint as rbm
import own_retained_history as ret
OUT=ne.ROOT/'own-cohort-boost-20261126'
PARENT=blend.OUT/'schema-07'


class Tables:
    def __init__(self):
        self.ix=ret.map_rows();self.n=1408395;self.nq=2817131
        self.pairs={}
        for root,fp,field in ((cf.OUT,'probabilities.npy','probabilities_sha256'),(hn.OUT,'effects.npy','output_sha256')):
            receipt=json.loads((root/'manifest.json').read_text());p=root/fp;assert daystates.sha(p)==receipt[field]
            self.pairs[fp]=np.load(p,mmap_mode='r')
        self.rb=[]
        sources=json.loads((rbm.OUT/'freeze.json').read_text())['signals']
        for name in rbm.MODELS:
            p=rbm.OUT/(name+'.f32');assert daystates.sha(p)==sources[str(p)]
            self.rb.append(np.memmap(p,'<f4','r').reshape(-1,7))
        self.day=np.memmap(daystates.OUT/'states.f32','<f4','r').reshape(-1,20)
        rec=json.loads((daystates.OUT/'freeze.json').read_text());assert daystates.sha(daystates.OUT/'states.f32')==rec['state_features_sha256']
        self.j=[]
        jr=json.loads((joint.OUT/'manifest.json').read_text());er=json.loads((extra.OUT/'manifest.json').read_text())
        for meta in list(jr['models'].values())+list(er['inputs'].values()):
            assert daystates.sha(meta['vector'])==meta['vector_sha256'];self.j.append(np.memmap(meta['vector'],'<f4','r'))
    def role(self,phase,p,q,base,control):
        known=phase=='probex';train='trainx' if known else 'fulltrain'
        self.kind=phase;self.original=(self.ix if known else np.arange(self.n,self.n+self.nq))
        self.mode=1 if known else 2;self.base=base;self.warm=full.cf96(phase)
        self.ixuser,self.ixmovie,self.date=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc=np.log1p(ne.history('trainx','user_cnts'));self.ic=np.log1p(ne.history('trainx','item_cnts'))
        self.families=[];k=0;weights=np.array(control['weights'])
        for name in control['columns']:
            z,proof=blend.direction(name,p,q,known)
            expected=control[('known_sources' if known else 'qual_sources')][name];assert proof['sha256']==expected['sha256']
            w=weights[k:k+z.shape[1]];k+=z.shape[1]
            self.families.append((z@w).astype('f'))
            if z.shape[1]>1:self.families.append(z.std(1).astype('f'))
        assert k==len(weights)
        self.native=[]
        for name in ('own-choice-factors-20261126','own-suffix-factors-20261126','own-suffix-impute-20261126','own-exact-attention-20261126'):
            root=ne.ROOT/name;r=json.loads((root/'freeze.json').read_text());assert r['qualifying_labels_used_for_fitting'] is False
            v=np.load(root/(phase+'.npy'),mmap_mode='r')
            if 'suffix-impute' in name:v=v[:,[0,4,5,7]]
            self.native.append(v)
    def block(self,rows):
        rr=self.original[rows];b=self.base[rows];u=self.ixuser[rows];m=self.ixmovie[rows]
        cp=self.pairs['probabilities.npy'];pp=cp[rr,self.mode].astype('f');p0=cp[rr,0].astype('f')
        hp=self.pairs['effects.npy'];h0=hp[rr,:,0];h=hp[rr,:,self.mode]
        ds=self.day[rr];raw=np.column_stack([v[rr] for v in self.j]);center=raw.mean(1)
        family=np.column_stack([v[rows] for v in self.families])
        parts=[b,self.warm[rows]-b,center-b,raw-center[:,None],p0,pp-p0,h0-b[:,None],h-h0,
               ds[:,0]-b,ds[:,1:4],ds[:,4::4],ds[:,5::4],ds[:,6::4],self.uc[u],self.ic[m],self.date[rows]/2000,
               family,family.std(1)]
        for v in self.rb:
            a=v[rr];parts.extend((a[:,0]-b,a[:,1:5]-a[:,:1],a[:,5:7]))
        for v in self.native:parts.append(v[rows])
        return np.column_stack(parts).astype('f')
    def known(self):
        x=np.empty((len(self.base),self.block(np.arange(2)).shape[1]),'f')
        for j in range(0,len(x),8192):
            rows=np.arange(j,min(j+8192,len(x)));x[rows]=self.block(rows)
        return x


def main():
    OUT.mkdir(exist_ok=True);y,f,p,q=anchor.history_control();r=json.loads((PARENT/'freeze.json').read_text())
    assert r['selected']=='candidate' and ne.sha(PARENT/'selected.f32')==r['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    table=Tables();table.role('probex',p,q,b,r);x=table.known()
    u=table.ixuser;nu=len(table.uc);kc=np.bincount(u,minlength=nu);qc=np.bincount(ne.history('qual','user_idxs'),minlength=nu)
    ratio=((qc[u]+4)/(kc[u]+2)).clip(.5,4);ratio/=ratio.mean();models=[];held=np.empty((len(y),2),'f')
    for j in range(5):
        tr=f!=j;va=~tr;results=[]
        for k in range(2):
            model=CatBoostRegressor(iterations=640 if k==0 else 510,depth=6+k,learning_rate=.039 if k==0 else .051,
                l2_leaf_reg=110,random_strength=1.8,thread_count=2,random_seed=12837+j*2+k,allow_writing_files=False,verbose=False)
            model.fit(x[tr],(y-b)[tr],sample_weight=None if k==0 else ratio[tr]);held[va,k]=model.predict(x[va],thread_count=2)
            model.save_model(str(OUT/f'fold-{j}-{k}.cbm'));results.append(model)
        models.append(results);print('fold',j,ne.rmse(b[va]+held[va].mean(1),y[va]),flush=True)
    delta=held.mean(1);stats={str(w):blend.metrics(ne.clipped(b+w*delta),y,f) for w in (0,.25,.5,.75,1)}
    ok=[w for w in stats if ret.qualifies(stats[w],stats['0'])];choice=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    report=dict(parent=str(PARENT),selected=choice,known=stats,width=x.shape[1],rating_sources=r['columns'],
        qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        del x;table.role('qual',p,q,qb,r);dq=np.empty(len(qb),'f')
        for j in range(0,len(qb),32768):
            rows=np.arange(j,min(len(qb),j+32768));x=table.block(rows)
            dq[rows]=sum(m.predict(x,thread_count=2) for group in models for m in group)/10
        np.save(OUT/'oof.npy',delta);np.save(OUT/'qual-correction.npy',dq)
        ne.clipped(qb+float(choice)*dq).astype('<f4').tofile(OUT/'selected.f32');report['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
