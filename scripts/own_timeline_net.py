"""Predict masked ratings from the ordered sessions before and after the target date."""
import ctypes
import json
import math
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_slate as slate
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend

OUT=ne.ROOT/'own-timeline-net-20261126'
G=12
WIDTH=13
FEATURES=6+2*G


class Data(slate.Data):
    def __init__(self,train,phase):
        super().__init__(train,phase)
        lib=ctypes.CDLL(str((OUT/'timeline.dylib').resolve()));it,pt=ctypes.c_int,ctypes.c_void_p
        for fn,ni,npointer in ((lib.timeline_counts,2,3),(lib.timeline_build,2,9),(lib.timeline_tokens,3,12)):
            fn.argtypes=[it]*ni+[pt]*npointer;fn.restype=None
        self.lib=lib
        self.small=np.ascontiguousarray(np.column_stack((self.geo[:,:6],self.geo[:,12:18])))
        self.average=np.ascontiguousarray((np.bincount(self.i,weights=self.r,minlength=len(self.small))+10*self.r.mean())/
                         (np.bincount(self.i,minlength=len(self.small))+10),'f')
        off=np.r_[0,self.uc.cumsum()].astype('i');nu=len(self.uc);nd=int(self.d.max())+1;assert self.d.min()>=0
        counts=np.zeros(nu,'i');lib.timeline_counts(nu,nd,*[x.ctypes.data for x in (off,self.d,counts)])
        self.soff=np.r_[0,counts.cumsum()].astype('i');self.sdval=np.empty(self.soff[-1],'i2')
        self.sums=np.zeros((self.soff[-1],FEATURES),'f')
        lib.timeline_build(nu,nd,*[x.ctypes.data for x in (off,self.i,self.d,self.r,self.small,self.average,self.soff,self.sdval,self.sums)])
        print(train,'sessions',len(self.sums),flush=True)
    def inputs(self,u,group=None,rng=None):
        inputs,labels,rows=self.batch(u,group,rng);total,n,m,d,valid,bias=inputs
        remove=np.zeros((len(u),0),'i') if group is None else self.masks[group][u]
        if group is not None:
            m=m[:,:slate.MASK];d=d[:,:slate.MASK];bias=bias[:,:slate.MASK];valid=valid[:,:slate.MASK]
            labels=labels[:,:slate.MASK]
        dates=np.ascontiguousarray(d.numpy(),'i2');us=np.ascontiguousarray(u,'i')
        ctx=np.zeros((*dates.shape,WIDTH,FEATURES+4),'f')
        args=(us,dates,remove,self.i,self.d,self.r,self.small,self.average,self.soff,self.sdval,self.sums,ctx)
        self.lib.timeline_tokens(len(u),m.shape[1],remove.shape[1],*[x.ctypes.data for x in args])
        return (total,n,m,bias,d,torch.from_numpy(ctx)),labels,rows


class Net(nn.Module):
    def __init__(self,data):
        super().__init__()
        for name in ('eb','op','geo'):self.register_buffer(name,slate.tensor(getattr(data,name)),persistent=False)
        self.movie=nn.Embedding(len(data.geo),12);nn.init.normal_(self.movie.weight,std=.08)
        self.session=nn.Sequential(nn.Linear(FEATURES+4,32),nn.LayerNorm(32),nn.SiLU())
        self.prev=nn.Linear(32,32,bias=False);self.next=nn.Linear(32,32,bias=False)
        self.key=nn.Linear(32,24);self.val=nn.Linear(32,36)
        self.query=nn.Linear(24+5+12,24)
        self.out=nn.Sequential(nn.Linear(36+32+24+5+12+3,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.16),
                        nn.Linear(80,40),nn.SiLU(),nn.Linear(40,5))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
        self.closeness=nn.Parameter(torch.tensor([.2,1.,2.]))
    def forward(self,total,n,m,bias,d,tokens):
        h=torch.tanh(total/n.clamp(min=1).sqrt()[:,None]+self.eb)
        logit=(self.op[m]*h[:,None,None]).sum(-1)+bias
        prob=logit.softmax(2)
        ident=torch.cat((self.geo[m],prob,self.movie(m)),2)
        usable=tokens[:,:,:,-1]>0
        state=self.session(tokens)*usable[:,:,:,None]
        left=F.pad(state[:,:,:-1],(0,0,1,0));right=F.pad(state[:,:,1:],(0,0,0,1))
        state=F.silu(state+self.prev(left)+self.next(right))
        shape=(*m.shape,WIDTH,3)
        keys=self.key(state).reshape(*shape,8);query=self.query(ident).reshape(*m.shape,3,8)
        scores=(keys*query[:,:,None]).sum(-1)/math.sqrt(8)+tokens[:,:,:,-2,None]*self.closeness
        weight=scores.masked_fill(~usable[:,:,:,None],-10000).softmax(2)*usable[:,:,:,None]
        value=self.val(state).reshape(*shape,12)
        attended=(value*weight[:,:,:,:,None]).sum(2).reshape(*m.shape,36)
        side=torch.stack((d/2000.,torch.log1p(n)[:,None].expand_as(d)/7,usable.sum(2)/WIDTH),2)
        z=self.out(torch.cat((attended,state[:,:,WIDTH//2],ident,side),2))
        return logit+1.2*z,prob


def run(train,phase):
    data=Data(train,phase);torch.manual_seed(97231);net=Net(data)
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.03);star=torch.arange(1,6,dtype=torch.float32)
    start=time.monotonic()
    for ep in range(14):
        rng=np.random.default_rng(597821+ep);net.train()
        opt.param_groups[0]['lr']=.00014+.00062*(1+math.cos(ep*math.pi/14))/2
        for step in range(380):
            users=rng.choice(data.ready,144,replace=False)
            args,labels,_=data.inputs(users,(ep+step)%4,rng);y=slate.tensor(labels);ok=y>=0
            logits,p=net(*args);mu=logits[ok].softmax(1)@star
            loss=.65*(mu-y[ok]-1).square().mean()+.33*F.cross_entropy(logits[ok],y[ok])+.06*F.kl_div(logits[ok].log_softmax(1),p[ok],reduction='batchmean')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        print(train,'epoch',ep+1,round(time.monotonic()-start),flush=True)
    torch.save(net.state_dict(),OUT/(train+'.pt'));net.eval();out=np.empty((data.n,7),'f')
    with torch.inference_mode():
        for j in range(0,len(data.users),256):
            args,_,rows=data.inputs(data.users[j:j+256]);logit,raw=net(*args);keep=(rows>=0)&(rows<data.n)
            assert np.max(np.abs((raw@star).numpy()[keep]-data.native[rows[keep]]),initial=0)<.0003
            z=logit.softmax(2);out[rows[keep]]=torch.cat((z,(z@star)[:,:,None],(raw@star)[:,:,None]),2).numpy()[keep]
    np.save(OUT/(phase+'.npy'),out);return out


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','scripts/own_timeline_net.cpp','-o',str(OUT/'timeline.dylib')],check=True)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    def features(train,phase,b):
        v=run(train,phase);u,i=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        return np.column_stack((v,b,v[:,5]-b,v[:,5]-v[:,6],ne.verified_rank(phase)-b,
              np.log1p(ne.history(train,'user_cnts')[u]),np.log1p(ne.history(train,'item_cnts')[i]))).astype('f')
    x=features('trainx','probex',base)
    def model():return CatBoostRegressor(iterations=700,depth=5,learning_rate=.04,l2_leaf_reg=60,thread_count=2,
               random_seed=27531,verbose=False,allow_writing_files=False)
    key,known,held=anchor.crossfit(x,y,f,base,model)
    rep=dict(parent=str(parent),selected=key,known=known,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb);delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
