"""Fine-tune a local-history and ordinal adaptation of our trained CF96 on matched masks."""
import ctypes
import json
import math
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_cf_context as cf
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_retained_history as retained
import own_full_context as full
import own_overnight_blend as blend
from own_kernel_family import gram
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-cf-adapter-20261126'
EPOCHS=12
STEPS=480
BATCH=1024


class Data:
    def __init__(self,train,phase):
        self.u,self.i,self.d,self.r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
        self.qu,self.qi,self.qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        uc=ne.history(train,'user_cnts');self.nm=len(ne.history(train,'item_cnts'))
        self.off=np.r_[0,uc.cumsum()].astype('i');n=len(uc)
        self.s=cf.checkpoint(train);self.enc=np.ascontiguousarray(self.s['a'].cumsum(0,dtype=float),'f')
        self.lib=ctypes.CDLL(str((OUT/'adapter.dylib').resolve()))
        it,pt=ctypes.c_int,ctypes.c_void_p
        self.lib.cf_totals.argtypes=[it,it]+[pt]*7
        self.lib.cf_batch.argtypes=[it]*3+[pt]*16
        self.lib.chronological.argtypes=[it]+[pt]*3
        self.lib.choose_masks.argtypes=[it,it]+[pt]*7;self.lib.choose_masks.restype=it
        for fn in (self.lib.chronological,self.lib.cf_totals,self.lib.cf_batch):fn.restype=None
        self.order=np.empty(len(self.i),'i')
        self.total=np.empty((n,96),'f');self.md=np.empty(n,'f')
        self.lib.chronological(n,*[v.ctypes.data for v in (self.off,self.d,self.order)])
        self.lib.cf_totals(n,self.nm,*[v.ctypes.data for v in (self.off,self.i,self.d,self.r,self.enc,self.total,self.md)])
        self.sim=gram(self.s['p'][2:].mean(0))
        avg=self.d.mean(dtype=float)
        self.sd=(sum(np.dot(self.d[j:j+1_000_000].astype(float),self.d[j:j+1_000_000].astype(float))
                     for j in range(0,len(self.d),1_000_000))/len(self.d)-avg**2)**.5
        self.md[uc==0]=avg
        self.rank=ne.verified_rank(train);self.cf=full.cf96(train)
        self.qcf=full.cf96(phase)
        qo=np.argsort(self.qu,kind='stable').astype('i')
        qs=np.r_[0,np.bincount(self.qu,minlength=n).cumsum()].astype('i')
        self.groups=[]
        for k in range(4):
            starts=np.empty(n+1,'i');rows=np.empty(32*n,'i')
            count=self.lib.choose_masks(n,1778+k,*[v.ctypes.data for v in (self.off,self.d,qs,qo,self.qd,starts,rows)])
            self.groups.append((starts,rows[:count].copy()))
        print(train,'prepared',flush=True)

    def batch(self,rows,training,group):
        rows=np.ascontiguousarray(rows,'i')
        us,it,day=(self.u,self.i,self.d) if training else (self.qu,self.qi,self.qd)
        off,mask=self.groups[group]
        ctx=np.empty((len(rows),3,96),'f');sup=np.empty((len(rows),3),'f')
        self.lib.cf_batch(len(rows),self.nm,int(training),*[v.ctypes.data for v in (rows,us,it,day,self.off,self.order,
            self.i,self.d,self.r,self.enc,self.total,off,mask,self.sim,ctx,sup)])
        u,m,d=us[rows],it[rows],day[rows]
        diff=d-self.md[u]
        dev=np.sign(diff)*np.power(np.abs(diff)/self.sd,.4)
        bias=(self.s['b'][:,m]+self.s['user_bias'][:,u]+self.s['user_drift'][:,u]*dev).T
        dec=np.ascontiguousarray(self.s['p'][:,m,:].transpose(1,0,2))
        return tuple(torch.from_numpy(np.ascontiguousarray(v,'f')) for v in (ctx,sup,dec,bias))


class Net(nn.Module):
    def __init__(self,s):
        super().__init__()
        for key in ('enc_proj','dec_proj','enc_bias'):
            self.register_buffer(key,torch.from_numpy(s[key]))
        self.compress=nn.Sequential(nn.Linear(96,48),nn.LayerNorm(48),nn.SiLU())
        self.adapter=nn.Sequential(nn.Linear(3*48+3*5+3,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.12),
                                   nn.Linear(128,96),nn.SiLU(),nn.Linear(96,2+96+5+2))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)

    def forward(self,ctx,sup,p,bias):
        h=(ctx@self.enc_proj.T+self.enc_bias).tanh()
        z=h@self.dec_proj.T
        raw=torch.bmm(z,p.transpose(1,2))+bias[:,None,:]
        probs=raw.cumsum(2).softmax(2)
        params=self.adapter(torch.cat((self.compress(z).flatten(1),probs.flatten(1),sup),1))
        gates=.5*params[:,:2].tanh()
        add=.14*params[:,2:98].tanh()
        adj=(gates[:,:,None]*(raw[:,1:]-raw[:,:1])).sum(1)
        corrected=raw[:,0]+adj+torch.bmm(p,add[:,:,None]).squeeze(2)+.35*params[:,98:103]
        return corrected.cumsum(1),probs[:,0],params[:,103:]


def run(train,phase):
    data=Data(train,phase);torch.manual_seed(20261128)
    net=Net(data.s)
    opt=torch.optim.AdamW(net.parameters(),lr=.00055,weight_decay=.004)
    stars=torch.arange(1,6,dtype=torch.float32)
    stamp=time.monotonic()
    for ep in range(EPOCHS):
        rng=np.random.default_rng(99168+ep);net.train()
        opt.param_groups[0]['lr']=.00008+.00047*(1+math.cos(ep*math.pi/EPOCHS))*.5
        for step in range(STEPS):
            k=(ep+step)%4;subset=data.groups[k][1]
            rows=subset[rng.integers(0,len(subset),BATCH)]
            logits,original,resid=net(*data.batch(rows,True,k))
            mu=logits.softmax(1)@stars
            y=torch.from_numpy(data.r[rows].astype('int64')-1)
            aux=torch.from_numpy(np.column_stack((data.r[rows]-data.rank[rows],data.r[rows]-data.cf[rows])))
            loss=.57*(mu-y.float()-1).square().mean()+.37*F.cross_entropy(logits,y)+.12*F.mse_loss(resid,aux)
            loss+=.06*F.kl_div(logits.log_softmax(1),original,reduction='batchmean')
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
        torch.save(net.state_dict(),OUT/(train+'.pt'))
        print(train,'epoch',ep+1,'elapsed',round(time.monotonic()-stamp),flush=True)
    net.eval();out=np.empty((len(data.qi),3),'f')
    with torch.inference_mode():
        for start in range(0,len(out),2048):
            rows=np.arange(start,min(start+2048,len(out)),dtype='i')
            logits,orig,aux=net(*data.batch(rows,False,0))
            assert np.max(np.abs((orig@stars).numpy()-data.qcf[rows]))<.0003
            out[rows]=torch.column_stack((logits.softmax(1)@stars,aux)).numpy()
    assert np.isfinite(out).all()
    np.save(OUT/(phase+'.npy'),out)
    return out.astype(float)


def directions(a,cf,base):
    return np.column_stack((a[:,0]-base,a[:,0]-cf,a[:,1:]))


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cf_adapter.cpp','scripts/own_matched_masks.cpp',
                    'scripts/own_session_spectrum.cpp','-o',str(OUT/'adapter.dylib')],check=True)
    torch.set_num_threads(2)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    a=run('trainx','probex')
    x=directions(a,full.cf96('probex'),base)
    held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);orig=blend.metrics(base,y,f);stats=blend.metrics(ne.clipped(base+held),y,f)
    ok=retained.qualifies(stats,orig)
    np.save(OUT/'oof.npy',held)
    report=dict(selected='candidate' if ok else 'control',known=stats,control=orig,parent=str(parent),standalone=ne.rmse(a[:,0],y),
                weights=w.tolist(),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        qx=directions(run('fulltrain','qual'),full.cf96('qual'),qb)
        delta=qx@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
