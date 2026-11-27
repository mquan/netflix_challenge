"""Jointly train a chronological attention decoder and a frozen target-removed CF96 probability teacher."""
import ctypes
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_temporal_attention as temporal
from own_empirical_neighborhood import fit
OUT=ne.ROOT/'own-hybrid-attention-20261126';WARM=ne.ROOT/'own-cf-temporal-attention-20261126'
PARENT=blend.OUT/'schema-07'


class Data(temporal.Data):
    def __init__(self,train,phase,lib):
        super().__init__(train,phase,True,True);self.lib=lib
        u0,_,day0,_=anchor.old_probe();u=np.r_[u0,ne.history('qual','user_idxs')];day=np.r_[day0,ne.history('qual','dates')]
        self.cohort=np.bincount(u,minlength=len(self.counts)).astype('i');self.hidden=np.diff(self.ioff).astype('i')
        self.boundary=np.full(len(self.counts),30000,'i2');np.minimum.at(self.boundary,u,day);self.off=np.ascontiguousarray(self.off,'i8')
        self.cf=cf.checkpoint(train);self.enc=np.ascontiguousarray(self.cf['a'].cumsum(0,dtype=float),'f')
        d=ne.history(train,'dates');avg=d.mean(dtype=float)
        self.sd=(sum(np.sum((d[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(d),1000000))/len(d))**.5
        self.probs=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')[:,7:12]
        np.testing.assert_allclose(self.probs[::7]@np.arange(1,6,dtype='f'),full.cf96(phase)[::7],atol=.0003,rtol=0)
    def examples(self,us,seed):
        b=len(us);us=np.ascontiguousarray(us,'i');hi,hr,hd=[np.zeros((b,temporal.H),'i') for _ in range(3)]
        ti,td=[np.zeros((b,9),'i') for _ in range(2)];label=np.full((b,9),-1,'i');counts=np.zeros(b,'f')
        ctx=np.empty((b,96),'f');md=np.empty(b,'f')
        args=(us,self.off,self.i,self.d,self.r,self.boundary,self.cohort,self.hidden,hi,hr,hd,ti,td,label,counts,self.enc,ctx,md)
        self.lib.hybrid_masks(b,temporal.H,9,seed,len(self.pop),*[a.ctypes.data for a in args]);assert (label>=0).any(1).all()
        dev=(td-md[:,None])/self.sd;dev=np.sign(dev)*abs(dev)**.4
        cv=lambda a:torch.as_tensor(a,device='mps')
        return tuple(map(cv,(hi,hr,hd,ti,td,counts,us))),cv(label).long(),cv(ctx),cv(dev.astype('f'))


class Net(nn.Module):
    def __init__(self,data,train):
        super().__init__();self.attn=temporal.Net(data)
        source=torch.load(WARM/(train+'.pt'),map_location='cpu',weights_only=False)
        assert source['config']['cf_factors'] and source['epoch']==16
        self.attn.load_state_dict(source['net']);s=data.cf
        for k,v in dict(proj=s['enc_proj'],eb=s['enc_bias'],op=(s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2),
                         b=s['b'].cumsum(0).T,ub=s['user_bias'].cumsum(0).T,drift=s['user_drift'].cumsum(0).T).items():
            self.register_buffer(k,torch.from_numpy(np.ascontiguousarray(v)),persistent=False)
        self.fuse=nn.Sequential(nn.Linear(5*3+6+16,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.14),nn.Linear(64,24),nn.SiLU(),nn.Linear(24,6))
        self.side=nn.Linear(self.attn.film.embedding_dim,16)
        nn.init.zeros_(self.fuse[-1].weight);nn.init.zeros_(self.fuse[-1].bias)
    def teacher(self,ctx,dev,m,u):
        h=torch.tanh(ctx@self.proj.T+self.eb)
        return (self.op[m]*h[:,None,None,:]).sum(-1)+self.b[m]+self.ub[u,None]+dev[:,:,None]*self.drift[u,None]
    def forward(self,args,c):
        raw=self.attn(*args);hi,hr,hd,m,d,count,u=args
        aa=raw-raw.mean(-1,keepdim=True);cc=c-c.mean(-1,keepdim=True)
        p=aa.softmax(-1);q=cc.softmax(-1)
        near=(d[:,:,None]==hd[:,None])&(hr>0)[:,None]&(hi[:,None]!=m[:,:,None])
        observed=(hr>0)&(hr<6);available=(near&observed[:,None]).sum(2)
        info=torch.stack((d/2000.,torch.log1p(count)[:,None].expand_as(d)/7,self.attn.pop[m],self.attn.year[m],
                   available.float().log1p()/4,(near&(hr==6)[:,None]).sum(2).float()/8),2)
        z=self.fuse(torch.cat((p,q,p-q,info,self.side(self.attn.film(m))),2))
        w=.18+.67*torch.sigmoid(.55+z[:,:,:1])
        return w*cc+(1-w)*aa+.52*z[:,:,1:].tanh(),raw


def run(train,phase,lib):
    data=Data(train,phase,lib);torch.manual_seed(73703);torch.mps.manual_seed(73703);torch.set_num_threads(1)
    model=Net(data,train).to('mps');opt=torch.optim.AdamW(model.parameters(),lr=.00026,weight_decay=.012)
    stars=torch.arange(1,6,device='mps',dtype=torch.float32);started=time.monotonic()
    for ep in range(11):
        rng=np.random.default_rng(72197+ep);order=rng.permutation(data.ready);model.train()
        opt.param_groups[0]['lr']=.000055+.00026*(1+math.cos(ep*math.pi/11))/2
        for j in range(640):
            us=order[np.arange(j*208,(j+1)*208)%len(order)]
            args,y,ctx,dev=data.examples(us,int(rng.integers(1,2**30)))
            law=model.teacher(ctx,dev,args[3],args[-1]);fused,raw=model(args,law);valid=y>=0
            logits=fused[valid];p=logits.softmax(-1);actual=y[valid]
            loss=.63*(p@stars-actual-1).square().mean()+.34*F.cross_entropy(logits,actual)+.06*F.cross_entropy(raw[valid],actual)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.4);opt.step()
        torch.save(dict(epoch=ep+1,net=model.state_dict()),OUT/(train+'.pt'))
        print(train,'epoch',ep+1,round(time.monotonic()-started),flush=True)
    model.eval();out=np.zeros(len(data.qi),'f');users=np.flatnonzero(np.diff(data.qoff))
    with torch.inference_mode():
        for rep in range(3):
            rng=np.random.default_rng(1369+rep)
            for start in range(0,len(users),256):
                us=users[start:start+256];rr=[data.order[data.qoff[u]:data.qoff[u+1]] for u in us]
                args,_=data.batch(us,rng,False,rr);n=args[3].shape[1]
                probs=np.full((len(us),n,5),.2,'f')
                for k,rows in enumerate(rr):probs[k,:len(rows)]=data.probs[rows]
                law=torch.from_numpy(np.log(probs.clip(1e-9))).to('mps');logits,_=model(args,law)
                v=(logits.softmax(-1)@stars).cpu().numpy()
                for rows,val in zip(rr,v):out[rows]+=val[:len(rows)]/3
    assert np.isfinite(out).all();np.save(OUT/(phase+'.npy'),out);return out


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_hybrid_attention.cpp','-o',str(OUT/'hybrid.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'hybrid.dylib').resolve()));lib.hybrid_masks.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*18;lib.hybrid_masks.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def design(v,b,phase):
        w=np.load(WARM/(phase+'.npy'));native=full.cf96(phase)
        return np.column_stack((v-b,v-(.62*native+.38*w),v-w))
    x=design(run('trainx','probex',lib),base,'probex');held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),
        warm_checkpoint_sha256={k:ne.sha(WARM/(k+'.pt')) for k in ('trainx','fulltrain')},qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        delta=design(run('fulltrain','qual',lib),qb,'qual')@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
