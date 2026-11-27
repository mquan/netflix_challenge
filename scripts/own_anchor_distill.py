"""Estimate held-out forecast errors for observed original Probe peers from our original ratings models."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_full_context as full

OUT=ne.ROOT/'own-anchor-distill-20261126'


def original_bank():
    sig=json.loads((joint.OUT/'manifest.json').read_text());raw=[];lineage={}
    for k,v in sig['models'].items():
        p=Path(v['vector']);assert ne.sha(p)==v['vector_sha256']
        raw.append(np.fromfile(p,'<f4'));lineage[k]=v['training_receipt']
    shared=np.column_stack(raw);original=[v[:1408395] for v in raw]
    for name in ('J1-joint64','RBM-C0-control-10e','RBM-C7-visible-10e','HN1-history','HN2-sparse','HN3-session'):
        rp=Path('experiments')/(name+'.json');r=json.loads(rp.read_text());cmd=r['command']
        assert cmd[1] in ('joint','rbm','scripts/history_model.py') and cmd[cmd.index('--data')+1]=='data/processed/netflix.bin'
        assert not any(k in cmd for k in ('--init-model','--inputs','--base','--base-predictions','--graph','--time-model'))
        a=r['predictions_artifact'];p=Path(a['path']);p=p if p.exists() else Path('../netflix_personas')/p
        assert ne.sha(p)==a['sha256'];v=np.fromfile(p,'<f4');assert len(v)==1408395
        original.append(v);lineage[name]=str(rp)
    ret.read_pairs()
    for k in ret.MODELS:
        rp=ret.QUAL/k/'report.json';v=json.loads(rp.read_text());p=Path(v['column'])
        assert ne.sha(p)==v['column_sha256'];original.append(np.fromfile(p,'<f4'));lineage[k]=str(rp)
    r=json.loads((ret.QUAL/'B2-fixed128/report.json').read_text());assert ne.sha(r['column'])==r['column_sha256']
    original.append(np.fromfile(r['column'],'<f4'));lineage['B2']=str(ret.QUAL/'B2-fixed128/report.json')
    for ep in (3,6,10,12):
        p=ne.ROOT/f'retained-neural/A2h-epoch{ep}.f32';rp=Path(str(p)+'.json');m=json.loads(rp.read_text())
        assert m['data_sha256']==sig['training_data_sha256'] and m['config']['objective']=='masked' and m['epochs']==[ep]
        assert ne.sha(p)==m['column_sha256'];original.append(np.fromfile(p,'<f4'));lineage[p.name]=str(rp)
    (OUT/'sources.json').write_text(json.dumps(lineage,indent=2)+'\n')
    return shared,np.column_stack(original).clip(1,5)


class Data:
    def __init__(self,y,fold,base,qb):
        self.shared,bank=original_bank()
        u,m,d,old=cov.original();self.n=len(u);self.prior=old
        self.u=np.r_[u,ne.history('qual','user_idxs')];self.m=np.r_[m,ne.history('qual','item_idxs')];self.d=np.r_[d,ne.history('qual','dates')]
        self.seen,rows=anchor.training_indices('trainx',u,m,d)
        self.ix=ret.map_rows()
        self.star=np.empty(len(u),'f');self.star[self.ix]=y;self.star[self.seen]=ne.history('trainx','ratings')[rows]
        assert (self.seen.sum()+len(self.ix))==len(u)
        self.base=np.empty(len(self.u),'f');self.base[self.ix]=base;self.base[len(u):]=qb
        self.bank=bank;self.geo=cov.movie_geometry()
        self.uc=ne.history('trainx','user_cnts');self.ic=ne.history('trainx','item_cnts')
        idx=np.argsort(u,kind='stable');cnt=np.bincount(u,minlength=len(self.uc));offset=np.r_[0,cnt.cumsum()]
        self.anchors=np.full((len(cnt),cnt.max()),-1,'i');self.anchors[u[idx],np.arange(len(idx))-offset[u[idx]]]=idx
        feats=np.column_stack((bank-old[:,None],old,self.shared[:self.n],
              np.log1p(self.uc[u]),np.log1p(self.ic[m]),d/2000,self.geo[m,::5])).astype('f')
        self.distill=np.zeros(len(u),'f')
        err=base-old[self.ix]
        for k in range(5):
            tr=fold!=k;va=~tr
            model=CatBoostRegressor(iterations=450,depth=6,learning_rate=.065,l2_leaf_reg=45,thread_count=2,
               random_seed=2886+k,verbose=False,allow_writing_files=False)
            model.fit(feats[self.ix[tr]],err[tr])
            self.distill[self.ix[va]]=old[self.ix[va]]+model.predict(feats[self.ix[va]],thread_count=2)
            self.distill[self.seen]+=(old[self.seen]+model.predict(feats[self.seen],thread_count=2))/5
        self.base[:self.n][self.seen]=self.distill[self.seen]
        self.reliable=self.distill.copy();self.reliable[self.ix]=base
        self.smean=self.shared.mean(1);self.mu_a=self.bank.mean(1)
        print('anchor models',bank.shape[1],'held-user distillation RMSE',ne.rmse(self.distill[self.ix],base),flush=True)
    def features(self,rows):
        u,m,d,b=self.u[rows],self.m[rows],self.d[rows],self.base[rows]
        ar=self.anchors[u];ai=ar.clip(min=0);allow=(ar>=0)&(ar!=rows[:,None]);actual=allow&self.seen[ai]
        astar=self.star[ai];old=self.prior[ai];ref=self.reliable[ai]
        dt=d[:,None]-self.d[ai];sim=np.einsum('bd,bad->ba',self.geo[m],self.geo[self.m[ai]])
        target=np.column_stack((self.shared[rows]-self.smean[rows,None],self.smean[rows]-b,b-3.5,np.log1p(self.uc[u])/7,
                   np.log1p(self.ic[m])/9,d/2000)).astype('f')
        rep=lambda x:np.broadcast_to(x[:,None],ai.shape)
        core=np.stack((astar-old,astar-ref,astar-self.distill[ai],old-ref,ref-self.distill[ai],astar-3.5,
              old-3.5,self.smean[ai]-ref,sim,1/(1+abs(dt)/20),np.sign(dt)*np.log1p(abs(dt))/5,
              (dt==0).astype('f'),rep(b)-ref,rep(self.smean[rows])-self.smean[ai],
              np.log1p(self.ic[self.m[ai]])/9,rep(b-3.5)),2)
        x=np.concatenate((core,self.bank[ai]-self.mu_a[ai][:,:,None],
                   np.broadcast_to(target[:,None,:],(*ai.shape,target.shape[1]))),2).astype('f')
        return x,allow,actual
    def known(self):
        parts=None
        for j in range(0,len(self.ix),16384):
            sl=slice(j,j+16384);group=self.features(self.ix[sl])
            if parts is None:parts=[np.empty((len(self.ix),*v.shape[1:]),v.dtype) for v in group]
            for dest,v in zip(parts,group):dest[sl]=v
        return tuple(torch.from_numpy(v) for v in parts)


class Net(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.15),nn.Linear(48,24),nn.SiLU())
        self.out=nn.Linear(24,8);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.decay=nn.Parameter(torch.tensor(.45))
    def effects(self,x):
        a=self.out(self.shared(x)).tanh()
        # Predict the innovation in the current blend and in the original-only model.
        primary=.20*a[:,:,0]*x[:,:,1]+.15*a[:,:,1]*x[:,:,0]+.30*a[:,:,2]*x[:,:,3]+.10*a[:,:,3]
        auxiliary=.30*a[:,:,4]*x[:,:,0]+.22*a[:,:,5]*x[:,:,1]+.12*a[:,:,6]+.15*a[:,:,7]*x[:,:,8]*x[:,:,0]
        return primary,auxiliary
    def total(self,v,mask):
        n=mask.sum(1);return (v*mask).sum(1)/(1+self.decay.sigmoid()*(n-1).clamp(min=0))


def train(x,allowed,actual,y,base,original,fold,k):
    torch.manual_seed(29177+k);model=Net(x.shape[2]);opt=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.035)
    idx=torch.from_numpy(np.flatnonzero(fold!=k));states=[]
    for ep in range(16):
        model.train()
        if ep==10:opt.param_groups[0]['lr']=.00025
        for ix in idx[torch.randperm(len(idx))].split(3072):
            v=x[ix];one,two=model.effects(v);mask=actual[ix];true=y[ix]
            primary=model.total(one,mask)
            # Extra observed stars enter only the auxiliary training objective on other users.
            more=allowed[ix]&(torch.rand(allowed[ix].shape)<torch.rand(len(ix),1)*.65+.33)
            aux=model.total(two,more)
            known=mask.any(1);support=more.any(1)
            loss=(primary[known]-(true-base[ix])[known]).square().mean()+.38*(aux[support]-(true-original[ix])[support]).square().mean()+.10*primary.square().mean()
            opt.zero_grad();loss.backward();opt.step()
        if ep in (4,9,15):states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    torch.save(states,OUT/f'fold-{k}.pt')
    models=[]
    for state in states:
        m=Net(x.shape[2]);m.load_state_dict(state);m.eval();models.append(m)
    return models


def infer(models,x,mask):
    with torch.inference_mode():return sum(m.total(m.effects(x)[0],mask) for m in models).numpy()/len(models)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    data=Data(y,f,base,qb);x,allow,mask=data.known();held=np.zeros(len(y),'f');models=[]
    for k in range(5):
        ensemble=train(x,allow,mask,torch.from_numpy(y.astype('f')),torch.from_numpy(base.astype('f')),torch.from_numpy(data.smean[data.ix]),f,k)
        models+=ensemble;rows=np.flatnonzero(f==k)
        for j in range(0,len(rows),8192):
            ix=rows[j:j+8192];held[ix]=infer(ensemble,x[ix],mask[ix])
        print('fold',k,ne.rmse(base[rows]+held[rows],y[rows]),flush=True)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    wins=[k for k in stats if ret.qualifies(stats[k],stats['0'])];key=min(wins,key=lambda k:stats[k]['pooled']) if wins else 'control'
    rep=dict(parent=str(parent),selected=key,known=stats,original_model_receipts_sha256=ne.sha(OUT/'sources.json'),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if wins:
        del x
        delta=np.empty(len(qb),'f')
        for j in range(0,len(qb),8192):
            sl=slice(j,min(j+8192,len(qb)));rows=np.arange(sl.start+data.n,sl.stop+data.n)
            v,allow,_=data.features(rows);delta[sl]=infer(models,torch.from_numpy(v),torch.from_numpy(allow))
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
