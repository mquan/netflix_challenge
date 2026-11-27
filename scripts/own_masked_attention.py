"""Train paired masked-history attention models on Netflix trainx and fulltrain."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import own_neighbor_experiment as ne
from own_history_context import neighbor_control

OUT=ne.ROOT/'own-masked-attention-20261126'
HIST=160
TARGETS=20
WIDTH=48
HEADS=4
SEED=20261126


class Split:
    def __init__(self,split,query):
        self.i,self.r,self.d=[ne.history(split,k) for k in ('item_idxs','ratings','dates')]
        self.counts=ne.history(split,'user_cnts')
        self.off=np.r_[0,self.counts.cumsum()]
        self.qi,self.qu,self.qd=[ne.history(query,k) for k in ('item_idxs','user_idxs','dates')]
        self.order=np.argsort(self.qu,kind='stable')
        self.qoff=np.r_[0,np.bincount(self.qu,minlength=len(self.counts)).cumsum()]
        self.ready=np.flatnonzero(self.counts>=2)
        parent=ne.verified_rank(split,True)
        gram=parent.T@parent
        vec=np.linalg.eigh(gram)[1][:,-WIDTH:][:,::-1].copy()
        factors=parent@vec
        # Eigenvectors can reverse sign across independent fits.
        factors/=np.maximum(factors.std(axis=0),1e-6)
        self.factors=np.clip(factors,-6.,6.).astype('f')
        self.pop=np.log1p(ne.history(split,'item_cnts')).astype('f')/10.
        self.year=(np.clip(ne.history(split,'item_years'),1900,2010).astype('f')-1980)/25

    def batch(self,us,rng,training,queries=None):
        # Sample distinct positions in a user's profile. Every scored training
        # target is removed before constructing the visible rating sequence.
        b=len(us)
        hi=np.zeros((b,HIST),np.int32)
        hr=np.zeros_like(hi)
        hd=np.zeros_like(hi)
        if training:
            ti=np.zeros((b,TARGETS),np.int32)
            td=np.zeros_like(ti)
            ty=np.full_like(ti,-1)
        else:
            length=max(map(len,queries))
            ti,td=[np.zeros((b,length),np.int32) for _ in range(2)]
            ty=None
        for j,u in enumerate(us):
            n=self.counts[u]
            if training:
                t=min(TARGETS,max(1,int(n*.14)),n-1)
                rows=rng.choice(n,min(n,HIST+t),replace=False)+self.off[u]
                target,visible=rows[:t],rows[t:]
                ti[j,:t],td[j,:t],ty[j,:t]=self.i[target],self.d[target],self.r[target]-1
            else:
                visible=(np.arange(n) if n<=HIST else rng.choice(n,HIST,replace=False))+self.off[u]
                target=queries[j]
                ti[j,:len(target)],td[j,:len(target)]=self.qi[target],self.qd[target]
            m=len(visible)
            hi[j,:m],hd[j,:m],hr[j,:m]=self.i[visible],self.d[visible],self.r[visible]
        arrays=(hi,hr,hd,ti,td,self.counts[us].astype('f'))
        def tensor(x): return torch.as_tensor(x,device='mps')
        return tuple(map(tensor,arrays)),None if ty is None else tensor(ty).long()


class Predictor(nn.Module):
    def __init__(self,data):
        super().__init__()
        self.film=nn.Embedding.from_pretrained(torch.from_numpy(data.factors),freeze=False)
        self.register_buffer('pop',torch.from_numpy(data.pop))
        self.register_buffer('year',torch.from_numpy(data.year))
        self.stars=nn.Embedding(6,12,padding_idx=0)
        self.encoder=nn.Sequential(nn.Linear(WIDTH*2+15,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,WIDTH*2))
        self.query=nn.Linear(WIDTH,WIDTH)
        self.gate=nn.Parameter(torch.tensor([.1,1.,2.,.6]))
        self.head=nn.Sequential(nn.Linear(3*WIDTH+9,144),nn.LayerNorm(144),nn.GELU(),nn.Dropout(.12),
                                nn.Linear(144,64),nn.GELU(),nn.Linear(64,5))
        self.bias=nn.Embedding(len(data.pop),5)
        nn.init.zeros_(self.bias.weight)

    def forward(self,hi,hr,hd,ti,td,counts):
        b,h=hi.shape
        n=ti.shape[1]
        he,qe=self.film(hi),self.film(ti)
        rc=(hr.float()-3.5)/1.4
        xe=torch.cat([he,he*rc[:,:,None],self.stars(hr),torch.stack((self.pop[hi],self.year[hi],hd/2000.),-1)],-1)
        key,value=self.encoder(xe).chunk(2,dim=-1)
        def heads(x):return x.reshape(b,-1,HEADS,WIDTH//HEADS).transpose(1,2)
        logits=heads(self.query(qe))@heads(key).transpose(-1,-2)/math.sqrt(WIDTH//HEADS)
        distance=(td[:,:,None]-hd[:,None,:]).abs().float()
        logits=logits-self.gate.clamp(0,4)[None,:,None,None]*torch.log1p(distance[:,None]/7.)
        valid=hr!=0
        weights=logits.masked_fill(~valid[:,None,None,:],-10000).softmax(-1)
        attended=(weights@heads(value)).transpose(1,2).reshape(b,n,WIDTH)
        length=valid.sum(-1).clamp(min=1)
        pooled=(value*valid[:,:,None]).sum(1)/length[:,None]
        hist=torch.stack([(hr==k).sum(1)/length for k in range(1,6)],dim=-1)
        static=torch.cat([pooled,hist,torch.log1p(counts[:,None])/7],dim=-1)[:,None,:].expand(-1,n,-1)
        vector=torch.cat([qe,attended,static,torch.stack((td/2000.,self.pop[ti],self.year[ti]),-1)],-1)
        return self.bias(ti)+self.head(vector)


@torch.no_grad()
def predict(m,data,batch,passes=2):
    m.eval()
    accumulated=np.zeros(len(data.qi),dtype=np.float32)
    with_queries=np.flatnonzero(np.diff(data.qoff))
    for rep in range(passes):
        rng=np.random.default_rng(SEED+500+rep)
        for begin in range(0,len(with_queries),batch):
            us=with_queries[begin:begin+batch]
            indices=[data.order[data.qoff[u]:data.qoff[u+1]] for u in us]
            # Group extreme query counts into blocks to bound attention memory.
            for offset in range(0,max(map(len,indices)),32):
                rows=[a[offset:offset+32] for a in indices]
                args,_=data.batch(us,rng,False,rows)
                logits=m(*args)
                stars=torch.arange(1,6,device=logits.device,dtype=logits.dtype)
                pred=(logits.softmax(-1)@stars).cpu().numpy()
                for row,values in zip(rows,pred):
                    accumulated[row]+=values[:len(row)]
    return accumulated/passes


def train(split,query,epochs,batch,steps):
    data=Split(split,query)
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    m=Predictor(data).to('mps')
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
    start=time.monotonic()
    for e in range(1,epochs+1):
        rng=np.random.default_rng(SEED+e)
        us=rng.permutation(data.ready)
        m.train()
        opt.param_groups[0]['lr']=.0003+.0007*(1+math.cos((e-1)*math.pi/epochs))/2
        loss_sum=0.
        for step in range(steps):
            ix=np.arange(step*batch,(step+1)*batch)%len(us)
            args,y=data.batch(us[ix],rng,True)
            logits=m(*args)
            loss=F.cross_entropy(logits.reshape(-1,5),y.reshape(-1),ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),3.)
            opt.step()
            loss_sum+=float(loss.detach().cpu())
            if step%128==0:
                print(split,e,step,round(loss_sum/(step+1),4),round(time.monotonic()-start),flush=True)
        torch.save(m.state_dict(),OUT/(split+'-model.pt'))
    p=predict(m,data,batch)
    assert np.isfinite(p).all() and np.all((p>=1)&(p<=5))
    np.save(OUT/(query+'.npy'),p)
    return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--steps',type=int,default=1536)
    ap.add_argument('--batch',type=int,default=256)
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    y,folds,p,q=neighbor_control()
    known=train('trainx','probex',args.epochs,args.batch,args.steps)
    selected,stats=ne.select((known-p)[:,None],y,folds,p)
    report=dict(selected=selected,known=stats,standalone=ne.rmse(known,y),training=vars(args),
                known_comparison_is_nested=False,qualifying_hashes={},qualifying_labels_used_for_fitting=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if selected!='control':
        qual=train('fulltrain','qual',args.epochs,args.batch,args.steps)
        path=OUT/'selected.f32'
        ne.clipped(q+stats[selected]['weight']*(qual-q)).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control': ne.audit(OUT,ne.LABELS)


if __name__=='__main__':
    main()
