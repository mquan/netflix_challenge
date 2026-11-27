"""Train a differentiable CFNADE imputation model on held-out groups of plausible competition ratings."""
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
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
from own_query_sets import projection

OUT=ne.ROOT/'own-cf-slate-20261126'
MASK=5
EPOCHS=13
STEPS=400
BATCH=128


def tensor(a):
    dtype='float32' if a.dtype.kind=='f' else a.dtype
    return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype))


class Data:
    def __init__(self,train,phase):
        i,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
        self.r=r;self.i=i;self.d=d
        self.uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
        off=np.r_[0,self.uc.cumsum()].astype('i');nu=len(self.uc)
        us,mi,dd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.n=len(mi);self.native=full.cf96(phase)
        if phase=='probex':
            us,mi,dd=[np.concatenate((x,ne.history('qual',k))) for x,k in zip((us,mi,dd),('user_idxs','item_idxs','dates'))]
        qo=np.argsort(us,kind='stable').astype('i')
        qn=np.bincount(us,minlength=nu);qoff=np.r_[0,qn.cumsum()].astype('i')
        self.ready=np.flatnonzero((self.uc>=2)&(qn>0));self.users=np.flatnonzero(qn)
        self.qrows=np.full((nu,int(qn.max())),-1,'i')
        self.qrows[us[qo],np.arange(len(qo))-qoff[us[qo]]]=qo
        assert self.qrows.shape[1]<=12
        self.qi=mi;self.qd=dd
        lib=ctypes.CDLL(str((OUT/'slate.dylib').resolve()));it=ctypes.c_int;pt=ctypes.c_void_p
        lib.cf_totals.argtypes=[it,it]+[pt]*7;lib.cf_totals.restype=None
        lib.slate_masks.argtypes=[it]*3+[pt]*7;lib.slate_masks.restype=None
        s=cf.checkpoint(train);enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
        totals=np.empty((nu,96),'f');md=np.empty(nu,'f')
        lib.cf_totals(nu,len(ic),*[a.ctypes.data for a in (off,i,d,r,enc,totals,md)])
        self.total=totals@s['enc_proj'].T;self.md=md
        mean=d.mean(dtype=float);md[self.uc==0]=mean
        self.sd=(sum(np.sum((d[j:j+1000000].astype(float)-mean)**2) for j in range(0,len(d),1000000))/len(d))**.5
        self.ep=np.ascontiguousarray((enc@s['enc_proj'].T).transpose(1,0,2))
        self.op=np.ascontiguousarray((s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2))
        self.eb=s['enc_bias'];self.b,self.ub,self.drift=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
        self.geo=np.column_stack((projection(s['a'].mean(0),12),projection(s['p'].mean(0),12))).astype('f')
        old=anchor.old_probe();_,visible=anchor.training_indices(train,*old[:3])
        flags=np.zeros(len(r),'i1');flags[visible]=1
        self.masks=[]
        for k in range(4):
            mask=np.full((nu,MASK),-1,'i')
            lib.slate_masks(nu,MASK,96612+k,*[a.ctypes.data for a in (off,d,flags,qoff,qo,dd,mask)])
            self.masks.append(mask)
        print(train,'prepared',flush=True)
    def batch(self,u,group=None,rng=None):
        rows=self.qrows[u];v=rows>=0;idx=rows.clip(min=0)
        m=self.qi[idx];d=self.qd[idx]
        total=self.total[u].copy();n=self.uc[u].copy();labels=None
        if group is not None:
            rr=self.masks[group][u];active=rr>=0;rr=rr.clip(min=0)
            mi,day,star=self.i[rr],self.d[rr],self.r[rr]
            total-=(self.ep[mi,star-1]*active[:,:,None]).sum(1);n-=active.sum(1).astype('i')
            # Vary the number of true unobserved peers while fitting the additional pseudo-targets.
            keep=rng.random(v.shape)<rng.uniform(.4,1.,(len(u),1))
            v=np.column_stack((active,v&keep));m=np.column_stack((mi,m));d=np.column_stack((day,d))
            labels=np.full(m.shape,-1,'int64');labels[:,:MASK]=np.where(active,star-1,-1)
        diff=d-self.md[u,None];dev=np.sign(diff)*np.abs(diff/self.sd)**.4
        bias=self.b[m]+self.ub[u,None]+self.drift[u,None]*dev[:,:,None]
        return tuple(tensor(z) for z in (total,n.astype('f'),m.astype('int64'),d.astype('f'),v,bias)),labels,rows


class Net(nn.Module):
    def __init__(self,data):
        super().__init__()
        for name in ('ep','op','eb','geo'):self.register_buffer(name,tensor(getattr(data,name)),persistent=False)
        self.token=nn.Sequential(nn.Linear(24+5+2,40),nn.LayerNorm(40),nn.SiLU())
        self.q=nn.Linear(40,16);self.k=nn.Linear(40,16)
        self.gate=nn.Sequential(nn.Linear(5,16),nn.SiLU(),nn.Linear(16,1))
        self.tilt=nn.Sequential(nn.Linear(40+5,32),nn.SiLU(),nn.Linear(32,5))
        self.final=nn.Sequential(nn.Linear(40+5*3+3,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.15),nn.Linear(48,5))
        for layer in (self.gate[-1],self.tilt[-1],self.final[-1]):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,total,n,m,d,valid,bias):
        ep=self.ep[m];op=self.op[m]
        def decode(z):return (op*z.unsqueeze(2)).sum(-1)+bias
        original=decode(torch.tanh(total[:,None,:]/n.clamp(min=1).sqrt()[:,None,None]+self.eb))
        p0=original.softmax(2)
        features=torch.cat((self.geo[m],p0,d[:,:,None]/2000.,torch.log1p(n)[:,None,None].expand(-1,m.shape[1],1)/6),2)
        token=self.token(features)
        dot=self.q(token)@self.k(token).transpose(1,2)/4
        gap=(d[:,:,None]-d[:,None,:]).abs()
        sim=F.normalize(self.geo[m],dim=2)@F.normalize(self.geo[m],dim=2).transpose(1,2)
        available=valid[:,None,:]&(m[:,:,None]!=m[:,None,:])
        pair=torch.stack((dot,1/(1+gap/21),sim,(gap==0).float(),torch.log1p(n)[:,None,None].expand_as(sim)/6),3)
        weight=1.8*torch.sigmoid(self.gate(pair).squeeze(3))*available
        norm=(n[:,None]+weight.square().sum(2)).clamp(min=1).sqrt()[:,:,None]
        forecasts=[];prior=p0
        for _ in range(2):
            prob=(torch.log(prior.clamp(min=1e-9))+.7*self.tilt(torch.cat((token,prior),2))).softmax(2)
            expected=(ep*prob[:,:,:,None]).sum(2)
            updated=torch.tanh((total[:,None,:]+weight@expected)/norm+self.eb)
            logp=decode(updated)
            prior=logp.softmax(2);forecasts.append(prior)
        excess=torch.stack((weight.sum(2)/8,(weight*sim).sum(2)/8,(weight/(1+gap/20)).sum(2)/8),2)
        correction=.7*self.final(torch.cat((token,p0,*forecasts,excess),2))
        return logp+correction,p0,forecasts,excess


def run(train,phase):
    data=Data(train,phase);torch.manual_seed(67219);net=Net(data)
    opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.012)
    star=torch.arange(1,6,dtype=torch.float32);started=time.monotonic()
    for ep in range(EPOCHS):
        net.train();rng=np.random.default_rng(19311+ep)
        opt.param_groups[0]['lr']=.00011+.00058*(1+math.cos(ep*math.pi/EPOCHS))/2
        for step in range(STEPS):
            u=rng.choice(data.ready,BATCH,replace=False)
            inp,truth,_=data.batch(u,(step+ep)%4,rng)
            y=tensor(truth);active=y>=0
            logits,p0,_,_=net(*inp);logits=logits[active];y=y[active]
            mu=logits.softmax(1)@star
            loss=.55*F.mse_loss(mu,y.float()+1)+.40*F.cross_entropy(logits,y)+.05*F.kl_div(logits.log_softmax(1),p0[active],reduction='batchmean')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
        torch.save(net.state_dict(),OUT/(train+'.pt'))
        print(train,'epoch',ep+1,'seconds',round(time.monotonic()-started),flush=True)
    net.eval();out=np.empty((data.n,23),'f')
    with torch.inference_mode():
        for start in range(0,len(data.users),384):
            inp,_,rows=data.batch(data.users[start:start+384]);logits,raw,pr,coverage=net(*inp)
            keep=(rows>=0)&(rows<data.n)
            assert np.max(np.abs((raw@star).numpy()[keep]-data.native[rows[keep]]),initial=0)<.0003
            out[rows[keep]]=torch.cat((raw,*pr,logits.softmax(2),coverage),2).numpy()[keep]
    assert np.isfinite(out).all();np.save(OUT/(phase+'.npy'),out);return out


def features(raw,phase,b):
    mu=raw[:,15:20]@np.arange(1,6,dtype='f')
    return np.column_stack((b,mu-b,full.cf96(phase)-b,ne.verified_rank(phase)-b,raw)).astype('f')


def model():
    return CatBoostRegressor(iterations=800,depth=5,learning_rate=.045,l2_leaf_reg=60,thread_count=2,
        random_seed=738,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cf_slate.cpp','scripts/own_cf_adapter.cpp',
                    '-o',str(OUT/'slate.dylib')],check=True)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features(run('trainx','probex'),'probex',base)
    key,stats,held=anchor.crossfit(x,y,f,base,model);np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features(run('fulltrain','qual'),'qual',qb);delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
