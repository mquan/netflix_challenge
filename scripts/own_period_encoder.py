"""Train a masked ordinal predictor on exact day, week, quarter and whole-history summaries."""
import ctypes
import json
import math
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_neighbor_experiment as ne
import own_other_raters as other
import own_full_context as full
import own_retained_history as retained
import own_overnight_blend as blend
import own_anchor_context as anchor
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-period-encoder-20261126'
BATCH=1536
EPOCHS=14
STEPS=650


class Data:
    def __init__(self,train,query):
        self.u,self.i,self.d,self.r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
        self.qu,self.qi,self.qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc,self.ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
        self.off=np.r_[0,self.uc.cumsum()].astype('i')
        self.eligible=np.flatnonzero(self.uc>=2)
        self.re=self.r-ne.verified_rank(train)
        self.ce=self.r-full.cf96(train)
        c=other.geometry(train,self.ic)
        self.c=np.ascontiguousarray(c[:,[5,6,7,8,9,10,11,19,20,21,22,23]])
        self.film=np.column_stack((c,np.log1p(self.ic)*.1,(ne.history(train,'item_years')-1980)/30)).astype('f')
        self.lib=ctypes.CDLL(str((OUT/'period.dylib').resolve()))
        it,pt=ctypes.c_int,ctypes.c_void_p
        self.lib.periods.argtypes=[it]+[pt]*3
        self.lib.prefixes.argtypes=[it]+[pt]*10
        self.lib.assemble.argtypes=[it,it]+[pt]*12
        for func in (self.lib.periods,self.lib.prefixes,self.lib.assemble):func.restype=None
        ptr=lambda *a:[v.ctypes.data for v in a]
        self.starts=np.empty(len(self.uc)+1,'i')
        assert self.d.min()>=0 and self.d.max()<4096
        self.lib.periods(len(self.uc),*ptr(self.off,self.d,self.starts))
        self.days=np.empty(self.starts[-1],'i2')
        self.sums=np.empty((self.starts[-1],44),'f')
        self.lib.prefixes(len(self.uc),*ptr(self.off,self.d,self.i,self.r,self.re,self.ce,self.c,self.starts,self.days,self.sums))
        print(train,'sessions',len(self.days),flush=True)

    def batch(self,rows,training):
        rows=np.ascontiguousarray(rows,dtype='i')
        u,i,d=(self.u,self.i,self.d) if training else (self.qu,self.qi,self.qd)
        x=np.empty((len(rows),176),'f')
        self.lib.assemble(len(rows),int(training),*[v.ctypes.data for v in
            (rows,u,i,d,self.r,self.re,self.ce,self.c,self.starts,self.days,self.sums,x)])
        mov=i[rows].astype('int64')
        n=(self.uc[u[rows]]-int(training)).clip(min=0)
        extra=np.column_stack((d[rows]/2000.,np.log1p(n)*.16)).astype('f')
        return torch.from_numpy(x),torch.from_numpy(mov),torch.from_numpy(extra)

    def sample(self,rng):
        size=BATCH//3
        u=rng.choice(self.eligible,size)
        row=self.off[u]+(rng.random(size)*self.uc[u]).astype('i')
        return np.r_[row,rng.integers(0,len(self.i),BATCH-size)].astype('i')


class Predictor(nn.Module):
    def __init__(self,film):
        super().__init__()
        self.register_buffer('geometry',torch.from_numpy(film))
        self.movie=nn.Embedding(len(film),32)
        nn.init.normal_(self.movie.weight,std=.04)
        self.time=nn.Sequential(nn.Linear(44,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32))
        self.interact=nn.Linear(film.shape[1]+32,32)
        n=4*32*2+film.shape[1]+32+2
        self.net=nn.Sequential(nn.Linear(n,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.12),
                               nn.Linear(192,96),nn.SiLU(),nn.Dropout(.08),nn.Linear(96,7))

    def forward(self,x,m,e):
        v=torch.cat((self.geometry[m],self.movie(m)),1)
        z=self.time(x.reshape(-1,4,44))
        interact=z*self.interact(v)[:,None,:]
        return self.net(torch.cat((v,e,z.flatten(1),interact.flatten(1)),1))


def predict(train,query):
    data=Data(train,query)
    device='cpu'
    torch.manual_seed(20261126)
    net=Predictor(data.film).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
    stars=torch.arange(1,6,dtype=torch.float32)
    start=time.monotonic()
    for ep in range(EPOCHS):
        rng=np.random.default_rng(176011+ep)
        opt.param_groups[0]['lr']=.00015+.00085*(1+math.cos(ep*math.pi/EPOCHS))*.5
        net.train()
        for step in range(STEPS):
            rows=data.sample(rng)
            pred=net(*data.batch(rows,True))
            logits=pred[:,:5]
            y=torch.from_numpy(data.r[rows].astype('int64')-1)
            aux=torch.from_numpy(np.column_stack((data.re[rows],data.ce[rows])))
            mu=logits.softmax(1)@stars
            loss=.6*F.mse_loss(mu,y.float()+1)+.4*F.cross_entropy(logits,y)+.24*F.mse_loss(pred[:,5:],aux)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(net.state_dict(),OUT/(train+'.pt'))
        print(train,'epoch',ep+1,'elapsed',round(time.monotonic()-start),flush=True)
    net.eval()
    out=np.empty((len(data.qi),3),'f')
    with torch.inference_mode():
        for a in range(0,len(out),4096):
            rows=np.arange(a,min(a+4096,len(out)),dtype='i')
            vals=net(*data.batch(rows,False))
            out[rows]=torch.column_stack((vals[:,:5].softmax(1)@stars,vals[:,5:])).numpy()
    assert np.isfinite(out).all()
    np.save(OUT/(query+'.npy'),out)
    return out.astype(float)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_period_encoder.cpp','-o',str(OUT/'period.dylib')],check=True)
    torch.set_num_threads(3)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    x=predict('trainx','probex')
    standalone=ne.rmse(x[:,0],y)
    x[:,0]-=base
    held=np.empty(len(y))
    for j in range(5):
        tr=f!=j
        held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    stats=blend.metrics(ne.clipped(base+held),y,f)
    orig=blend.metrics(base,y,f)
    ok=retained.qualifies(stats,orig)
    w=fit(x,y-base)
    np.save(OUT/'oof.npy',held)
    report=dict(selected='candidate' if ok else 'control',known=stats,control=orig,parent=str(parent),standalone=standalone,
                weights=w.tolist(),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        qx=predict('fulltrain','qual');qx[:,0]-=qb
        delta=qx@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
