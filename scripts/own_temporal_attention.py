"""Predict jointly masked ratings using visible ratings and co-selected movie identities."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_masked_attention as first
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_session_latent as latent

T=14
H=192
WIDTH=64
HEADS=4


class Data(first.Split):
    def __init__(self,train,query,matched_days=False,cf_factors=False):
        super().__init__(train,query)
        self.matched_days=matched_days
        if cf_factors:
            from own_cf_context import checkpoint
            cf=checkpoint(train)
            dirs=np.column_stack((cf["p"][2:].mean(0),cf["a"][2:].mean(0)))
            _,vec=np.linalg.eigh(dirs.T@dirs)
            projection=dirs@vec[:,-32:]
            projection/=np.maximum(projection.std(0),1e-7)
            self.factors=np.column_stack((self.factors,np.clip(projection,-6,6))).astype("f")
        self.raw_starts=self.off
        # Chronological ordering is explicit here for both native partitions.
        order=np.lexsort((self.i,self.d,ne.history(train,'user_idxs')))
        self.i,self.r,self.d=[x[order] for x in (self.i,self.r,self.d)]
        uu,ii,dd=self.qu,self.qi,self.qd
        if query=='probex':
            uu,ii,dd=[np.r_[a,ne.history('qual',k)] for a,k in zip((uu,ii,dd),('user_idxs','item_idxs','dates'))]
        ids=np.lexsort((ii,dd,uu))
        self.ii=ii[ids]
        self.iday=dd[ids]
        self.ioff=np.r_[0,np.bincount(uu,minlength=len(self.counts)).cumsum()]
        path=latent.OUT/'graph-coordinates.npy'
        assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
        self.scene=np.load(path)

    def batch(self,us,rng,training,queries=None):
        b=len(us)
        hi,hr,hd=[np.zeros((b,H),np.int32) for _ in range(3)]
        length=T if training else max(map(len,queries))
        ti,td=[np.zeros((b,length),np.int32) for _ in range(2)]
        ty=np.full_like(ti,-1) if training else None
        for j,u in enumerate(us):
            lo,end=self.off[u:u+2]
            n=end-lo
            dates=self.d[lo:end]
            ai,bi=self.ioff[u:u+2]
            if training:
                t=min(T,max(1,round(n*.12)),n-1)
                a=rng.integers(max(0,n-min(n,60)) if rng.random()<.55 else 0,n)
                day=dates[a]
                if self.matched_days and bi>ai and rng.random()<.65:
                    wanted=int(self.iday[rng.integers(ai,bi)])
                    pos=np.searchsorted(dates,wanted)
                    left=max(0,pos-1)
                    right=min(n-1,pos)
                    day=dates[left if abs(int(dates[left])-wanted)<abs(int(dates[right])-wanted) else right]
                start,stop=np.searchsorted(dates,[int(day)-14,int(day)+15])
                near=rng.choice(np.arange(start,stop),min(t,stop-start),replace=False)
                if len(near)<t:
                    left=np.setdiff1d(np.arange(n),near,assume_unique=True)
                    near=np.r_[near,rng.choice(left,t-len(near),replace=False)]
                target=lo+near
                ti[j,:t],td[j,:t],ty[j,:t]=self.i[target],self.d[target],self.r[target]-1
                visible=np.ones(n,dtype=bool)
                visible[near]=False
                hide=(rng.random()<.25)
                if hide:visible[dates==day]=False
                available=np.flatnonzero(visible)
                unknown_i=np.r_[self.i[target],self.ii[ai:bi]]
                unknown_d=np.r_[self.d[target],self.iday[ai:bi]]
            else:
                target=queries[j]
                ti[j,:len(target)],td[j,:len(target)]=self.qi[target],self.qd[target]
                day=int(np.median(self.qd[target])) if len(target) else 0
                available=np.arange(n)
                unknown_i,unknown_d=self.ii[ai:bi],self.iday[ai:bi]
            # Unlabeled movies participate only through their identity and date.
            if len(unknown_i)>32:
                select=rng.choice(len(unknown_i),32,replace=False)
                unknown_i,unknown_d=unknown_i[select],unknown_d[select]
            nimp=len(unknown_i)
            cap=H-nimp
            if len(available)>cap:
                left,right=np.searchsorted(dates,[int(day)-14,int(day)+15])
                near=available[(available>=left)&(available<right)]
                if len(near)>cap//2:near=rng.choice(near,cap//2,replace=False)
                other=np.setdiff1d(available,near,assume_unique=True)
                available=np.r_[near,rng.choice(other,cap-len(near),replace=False)]
            vis=lo+available
            nv=len(vis)
            hi[j,:nv],hr[j,:nv],hd[j,:nv]=self.i[vis],self.r[vis],self.d[vis]
            hi[j,nv:nv+nimp],hr[j,nv:nv+nimp],hd[j,nv:nv+nimp]=unknown_i,6,unknown_d
        cv=lambda a:torch.as_tensor(a,device='mps')
        values=(hi,hr,hd,ti,td,self.counts[us].astype('f'),us.astype('i'))
        return tuple(map(cv,values)), None if ty is None else cv(ty).long()


class Net(nn.Module):
    def __init__(self,data):
        super().__init__()
        self.film=nn.Embedding.from_pretrained(torch.from_numpy(np.column_stack((data.factors,data.scene[:,:32])).astype('f')),freeze=False)
        self.register_buffer('pop',torch.from_numpy(data.pop))
        self.register_buffer('year',torch.from_numpy(data.year))
        dim=self.film.embedding_dim
        self.star=nn.Embedding(7,12,padding_idx=0)
        self.user=nn.Embedding(len(data.counts),16)
        nn.init.zeros_(self.user.weight)
        self.encoder=nn.Sequential(nn.Linear(2*dim+16,112),nn.LayerNorm(112),nn.GELU(),nn.Linear(112,2*WIDTH))
        self.query=nn.Linear(dim,WIDTH)
        self.hidden_gate=nn.Parameter(torch.tensor([-.5,.5,1.,-.4]))
        self.time_gate=nn.Parameter(torch.tensor([0.,.25,1.5,3.]))
        self.head=nn.Sequential(nn.Linear(dim+2*WIDTH+16+12,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(.12),
            nn.Linear(192,96),nn.GELU(),nn.Linear(96,5))
        self.bias=nn.Embedding(len(data.pop),5)
        nn.init.zeros_(self.bias.weight)

    def forward(self,hi,hr,hd,ti,td,counts,u):
        b,h=hi.shape
        n=ti.shape[1]
        he,qe=self.film(hi),self.film(ti)
        valid=hr!=0
        visible=valid&(hr<6)
        centered=(hr.float()-3.5)*visible/1.4
        extra=torch.stack((self.pop[hi],self.year[hi],hd/2000.,(hr==6).float()),-1)
        key,value=self.encoder(torch.cat([he,he*centered[:,:,None],self.star(hr),extra],-1)).chunk(2,-1)
        def heads(x):return x.reshape(b,-1,HEADS,WIDTH//HEADS).transpose(1,2)
        logits=heads(self.query(qe))@heads(key).transpose(-1,-2)/math.sqrt(WIDTH//HEADS)
        distance=(td[:,:,None]-hd[:,None,:]).abs().float()
        logits=logits-self.time_gate.clamp(0,5)[None,:,None,None]*torch.log1p(distance[:,None]/5.)
        logits=logits+self.hidden_gate[None,:,None,None]*(hr==6)[:,None,None,:]
        # A query movie can appear as an implicit token; leave its own identity
        # out of every attention sum, exactly as at inference.
        usable=valid[:,None,:]&(ti[:,:,None]!=hi[:,None,:])
        weights=logits.masked_fill(~usable[:,None],-10000.).softmax(-1)*usable[:,None]
        attended=(weights@heads(value)).transpose(1,2).reshape(b,n,WIDTH)
        nv=visible.sum(1).clamp(min=1)
        pooled=(value*visible[:,:,None]).sum(1)/nv[:,None]
        hist=torch.stack([(hr==k).sum(1)/nv for k in range(1,6)],-1)
        static=torch.cat([pooled,self.user(u),hist,torch.log1p(counts[:,None])/7.],-1)
        day=(distance==0)&usable
        dyn=torch.stack((td/2000.,self.pop[ti],self.year[ti],torch.log1p(day.sum(-1))/4.,
                         torch.log1p((day&visible[:,None]).sum(-1))/4.,
                         (day*centered[:,None]).sum(-1)/(day&visible[:,None]).sum(-1).clamp(min=1)),dim=-1)
        inp=torch.cat((qe,attended,static[:,None].expand(-1,n,-1),dyn),-1)
        return self.bias(ti)+self.head(inp)


def train(out,train,query,args):
    data=Data(train,query,args.matched_days,args.cf_factors)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    net=Net(data).to('mps')
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
    last=0
    if args.warm_parent:
        assert args.warm_parent=='own-temporal-attention-20261126' and not args.cf_factors
        root=ne.ROOT/args.warm_parent
        receipt=json.loads((root/'freeze.json').read_text())
        assert ne.sha(root/'selected.f32')==receipt['qualifying_hashes']['selected']
        path=root/(train+'.pt')
        seed_model=torch.load(path,map_location='mps',weights_only=False)
        assert seed_model['epoch']==18 and seed_model['config']['name']==args.warm_parent
        net.load_state_dict(seed_model['net'],strict=True)
        (out/(train+'-parent.json')).write_text(json.dumps(dict(path=str(path),sha256=ne.sha(path),
            original_freeze_sha256=ne.sha(root/'freeze.json')),indent=2)+'\n')
        del seed_model
    checkpoint=out/(train+'.pt')
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='mps',weights_only=False)
        assert (dict(matched_days=False,cf_factors=False,warm_parent=None)|saved['config'])==vars(args)
        net.load_state_dict(saved['net'])
        opt.load_state_dict(saved['opt'])
        last=saved['epoch']
    start=time.monotonic()
    for epoch in range(last+1,args.epochs+1):
        rng=np.random.default_rng(args.seed+epoch)
        idx=rng.permutation(data.ready)
        net.train()
        opt.param_groups[0]['lr']=.00018+.00072*(1+math.cos((epoch-1)*math.pi/args.epochs))/2
        if args.warm_parent:
            opt.param_groups[0]['lr']=.00005+.00025*(1+math.cos((epoch-1)*math.pi/args.epochs))/2
        loss_sum=0.
        for step in range(args.steps):
            us=idx[np.arange(step*args.batch,(step+1)*args.batch)%len(idx)]
            values,y=data.batch(us,rng,True)
            logits=net(*values)
            ce=F.cross_entropy(logits.reshape(-1,5),y.reshape(-1),ignore_index=-1)
            mean=logits.softmax(-1)@torch.arange(1,6,device='mps',dtype=torch.float32)
            seen=y>=0
            mse=((mean-(y+1))**2*seen).sum()/seen.sum().clamp(min=1)
            loss=.45*ce+.55*mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
            loss_sum+=float(loss.detach().cpu())
            if step%256==0:print(train,epoch,step,round(loss_sum/(step+1),4),round(time.monotonic()-start),flush=True)
        tmp=checkpoint.with_suffix('.tmp')
        torch.save(dict(config=vars(args),epoch=epoch,net=net.state_dict(),opt=opt.state_dict()),tmp)
        tmp.replace(checkpoint)
    preds=first.predict(net,data,args.batch,passes=3)
    assert np.isfinite(preds).all() and np.all((preds>=1)&(preds<=5))
    np.save(out/(query+'.npy'),preds)
    return preds


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='own-temporal-attention-20261126')
    ap.add_argument('--epochs',type=int,default=18)
    ap.add_argument('--matched-days',action='store_true')
    ap.add_argument('--warm-parent')
    ap.add_argument('--cf-factors',action='store_true')
    ap.add_argument('--steps',type=int,default=1280)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=20261127)
    args=ap.parse_args()
    out=ne.ROOT/args.name
    out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    y,folds,p,q=anchor.history_control()
    known=train(out,'trainx','probex',args)
    selected,stats=ne.select((known-p)[:,None],y,folds,p)
    report=dict(selected=selected,known=stats,standalone=ne.rmse(known,y),config=vars(args),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if selected!='control':
        qual=train(out,'fulltrain','qual',args)
        path=out/'selected.f32'
        ne.clipped(q+stats[selected]['weight']*(qual-q)).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
