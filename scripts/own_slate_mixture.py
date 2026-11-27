"""Infer latent user rating styles from observed peers in the fixed nine-title competition cohorts."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_full_context as full
import own_joint_cohort_source as teacher
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_query_sets as sets
import own_retained_history as retained
from own17x_cf96_followup import user_folds

OUT=ne.ROOT/'own-slate-mixture-20261126'


class Dataset:
    def __init__(self,y,p,q):
        u,i,d,_=anchor.old_probe();n=len(u)
        ix=retained.map_rows()
        known,rows=anchor.training_indices('trainx',u,i,d)
        self.u,self.i,self.d=[np.concatenate((a,ne.history('qual',k))).astype('i') for a,k in zip(
            (u,i,d),('user_idxs','item_idxs','dates'))]
        self.nu=len(ne.history('trainx','user_cnts'));order=np.argsort(self.u,kind='stable')
        counts=np.bincount(self.u,minlength=self.nu);off=np.r_[0,counts.cumsum()]
        self.grid=np.full((self.nu,int(counts.max())),-1,'i')
        self.grid[self.u[order],np.arange(len(order))-off[self.u[order]]]=order
        assert self.grid.shape[1]<=12
        self.stars=np.zeros(len(self.u),'int64')
        self.stars[np.flatnonzero(known)]=ne.history('trainx','ratings')[rows]
        self.stars[ix]=y.astype('int64');assert self.stars[:n].min()>=1
        self.dev=np.full(len(self.u),-1,'i');self.dev[ix]=np.arange(len(ix))
        self.qual=np.full(len(self.u),-1,'i');self.qual[n:]=np.arange(len(q))
        self.visible=np.zeros(len(self.u),bool);self.visible[:n]=known
        rec=json.loads((teacher.OUT/'manifest.json').read_text());assert not rec['query_outcomes_read']
        raw=[]
        for name in teacher.MODELS:
            r=rec['models'][name];assert ne.sha(r['vector'])==r['vector_sha256']
            raw.append(np.memmap(r['vector'],mode='r',dtype='<f4'))
        self.teacher=np.column_stack(raw).clip(1,5)
        self.dev_base=p.astype('f');self.qual_base=q.astype('f')
        g=sets.geometry()
        self.geometry=sets.projection(g,32)
        self.pop=np.log1p(ne.history('trainx','item_cnts')).astype('f')/10.
        self.activity=np.log1p(ne.history('trainx','user_cnts')).astype('f')/6.
        self.fold=user_folds(np.arange(self.nu))
        self.users=np.flatnonzero(counts>0)

    def batch(self,users,mode,rng=None):
        ix=self.grid[users];valid=ix>=0;rows=ix.clip(min=0)
        movies=self.i[rows];dates=self.d[rows]
        stars=self.stars[rows].copy();stars[~valid]=0
        t=self.teacher[rows];avg=t.mean(2);base=avg.copy()
        development=mode!='qual'
        pos=(self.dev if development else self.qual)[rows]
        target=valid&(pos>=0)
        compare=self.dev_base if development else self.qual_base
        base[target]=compare[pos[target]]
        observed=valid&(self.visible[rows] if development else (stars>0))
        if mode=='train':
            # Extra anchor labels occur only for the training users of this fold.
            simulate=rng.random((len(users),1))<.55
            random_mask=(rng.random(rows.shape)<rng.uniform(.1,.7,(len(users),1)))&(stars>0)&valid
            observed=np.where(simulate,random_mask,observed)
        inputs=np.concatenate((t-avg[:,:,None],avg[:,:,None]-3.5,self.pop[movies,None],
                np.broadcast_to(self.activity[users,None,None],(*rows.shape,1)),dates[:,:,None]/2000.),axis=2)
        cv=lambda x:torch.from_numpy(np.ascontiguousarray(x))
        return (cv(inputs.astype('f')),cv(movies.astype('int64')),cv(dates),cv(valid),cv(base),cv(avg),
                cv(np.where(observed,stars,0))),cv(stars),cv(target&~observed),pos


class Net(nn.Module):
    def __init__(self,geo):
        super().__init__()
        self.register_buffer('geo',torch.from_numpy(geo))
        self.token=nn.Sequential(nn.Linear(32+9,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.local=nn.Sequential(nn.Linear(64,40),nn.SiLU(),nn.Dropout(.15),nn.Linear(40,8*5+2))
        self.prior=nn.Sequential(nn.Linear(32,24),nn.SiLU(),nn.Linear(24,8))
        self.gap=nn.Parameter(torch.tensor(.0))
        self.center=nn.Sequential(nn.Linear(32+3,20),nn.SiLU(),nn.Dropout(.12),nn.Linear(20,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.center[-1].weight);nn.init.zeros_(self.center[-1].bias)
        self.register_buffer('offset',torch.tensor([-.9,-.55,-.28,0.,.3,.55,.85,0.]))
        self.register_buffer('spread',torch.tensor([.90,.78,1.14,.67,.95,.80,.86,1.5]))
        self.register_buffer('star',torch.arange(1,6,dtype=torch.float32))
    def forward(self,x,m,d,valid,b,teacher_mu,visible):
        z=self.token(torch.cat((x,self.geo[m]),-1))
        pooled=(z*valid[:,:,None]).sum(1)/valid.sum(1).clamp(min=1)[:,None]
        a=self.local(torch.cat((z,pooled[:,None].expand_as(z)),2))
        logit=-.5*((self.star-(teacher_mu[:,:,None,None]+self.offset[:,None]))/self.spread[:,None]).square()
        logit=logit+.9*a[:,:,:40].reshape(*m.shape,8,5).tanh()
        logp=logit.log_softmax(-1);mean=logp.exp()@self.star
        likelihood=logp.gather(3,(visible-1).clamp(min=0)[:,:,None,None].expand(-1,-1,8,-1)).squeeze(-1)
        dt=(d[:,:,None]-d[:,None,:]).abs()
        strength=torch.sigmoid(self.gap)
        weights=(strength+(1-strength)/(1+dt.float()/30))*(visible>0)[:,None,:]
        weights=weights*(1-torch.eye(m.shape[1],device=d.device)[None])
        prior=self.prior(pooled).log_softmax(1)
        post=(prior[:,None,:]+weights@likelihood).softmax(2)
        effect=((post-prior.exp()[:,None])*mean).sum(2)
        direct=self.center(torch.cat((z,(b-teacher_mu)[:,:,None],(b-3.5)[:,:,None],(teacher_mu-3.5)[:,:,None]),2)).squeeze(2)
        delta=effect*(.5+.6*a[:,:,40].tanh())+.10*direct+.08*a[:,:,41]
        distribution=(post[:,:,:,None]*logp.exp()).sum(2).clamp(min=1e-8)
        return delta,distribution


def fit(data,ids,seed,path):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=Net(data.geometry);opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.04)
    states=[]
    for ep in range(11):
        if ep==7:opt.param_groups[0]['lr']=.00026
        net.train();rng.shuffle(ids)
        for start in range(0,len(ids),768):
            inp,stars,target,_=data.batch(ids[start:start+768],'train',rng)
            delta,prob=net(*inp)
            if not target.any():continue
            err=stars.float()-inp[4]
            mse=((delta-err).square()*target).sum()/target.sum()
            eligible=(stars>0)&(inp[-1]==0)
            ll=-torch.log(prob.gather(-1,(stars-1).clamp(min=0)[:,:,None]).squeeze(2))
            loss=mse+.35*(ll*eligible).sum()/eligible.sum().clamp(min=1)+.12*(delta.square()*target).sum()/target.sum()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
        if ep in (2,6,10):states.append({k:v.detach().clone() for k,v in net.state_dict().items()})
    torch.save(states,path)
    return states


def infer(data,users,states,mode,pred):
    net=Net(data.geometry);net.eval()
    with torch.inference_mode():
        for state in states:
            net.load_state_dict(state)
            for start in range(0,len(users),2048):
                inp,_,target,pos=data.batch(users[start:start+2048],mode)
                delta,_=net(*inp);mask=target.numpy()
                pred[pos[mask]]+=delta.numpy()[mask]/len(states)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent'];fr=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    data=Dataset(y,base,qb);held=np.zeros(len(y))
    assert np.array_equal(data.fold[data.u[np.flatnonzero(data.dev>=0)]],f[data.dev[data.dev>=0]])
    for k in range(5):
        tr=data.users[data.fold[data.users]!=k];va=data.users[data.fold[data.users]==k]
        states=fit(data,tr,6715+k,OUT/f'fold-{k}.pt');infer(data,va,states,'dev',held)
        print('fold',k,ne.rmse(base[f==k]+held[f==k],y[f==k]),flush=True)
    np.save(OUT/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    rep=dict(selected=key,known=stats,parent=str(parent),teacher_manifest_sha256=ne.sha(teacher.OUT/'manifest.json'),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        delta=np.zeros(len(qb))
        for k in range(5):
            one=np.zeros(len(qb))
            infer(data,data.users,torch.load(OUT/f'fold-{k}.pt',weights_only=True),'qual',one);delta+=one/5
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
