"""Train on complete masked groups sampled to match each user's unlabeled query dates."""
import ctypes
import json
import math
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_period_encoder as period
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_retained_history as retained
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-matched-masks-20261126'
STEPS=550
EPOCHS=14
BATCH=1536


class Data(period.Data):
    def setup_masks(self):
        self.libmask=ctypes.CDLL(str((OUT/'mask.dylib').resolve()))
        it,pt=ctypes.c_int,ctypes.c_void_p
        self.libmask.choose_masks.argtypes=[it,it]+[pt]*7
        self.libmask.choose_masks.restype=it
        self.libmask.exclude_peers.argtypes=[it]+[pt]*11
        self.libmask.exclude_peers.restype=None
        n=len(self.uc)
        qs=np.r_[0,np.bincount(self.qu,minlength=n).cumsum()].astype('i')
        qo=np.argsort(self.qu,kind='stable').astype('i')
        self.groups=[]
        for k in range(4):
            starts=np.empty(n+1,'i')
            ids=np.empty(min(len(self.r),32*n),'i')
            used=self.libmask.choose_masks(n,20261129+k,*[a.ctypes.data for a in
                (self.off,self.d,qs,qo,self.qd,starts,ids)])
            assert used==starts[-1] and used>1_000_000
            self.groups.append((starts,ids[:used].copy()))
        # Fit the output prior from genuine training ratings only.
        count=np.empty((len(self.ic),5),'f')
        for star in range(1,6):
            count[:,star-1]=(np.bincount(self.i,weights=self.r==star,minlength=len(self.ic))+4)/(self.ic+20)
        self.film=np.column_stack((self.film,count)).astype('f')
        print('masked group sizes',[len(a) for _,a in self.groups],flush=True)

    def training_batch(self,rng,k):
        starts,ids=self.groups[k]
        rows=ids[rng.integers(0,len(ids),BATCH)]
        x,m,e=self.batch(rows,True)
        self.libmask.exclude_peers(len(rows),*[a.ctypes.data for a in
            (rows,self.u,self.i,self.d,self.r,self.re,self.ce,self.c,starts,ids,x.numpy())])
        # Match the global support count to the actually visible profile.
        e[:,1]=x[:,3*44]*.8
        return rows,(x,m,e)


class Net(nn.Module):
    def __init__(self,film):
        super().__init__()
        self.register_buffer('geometry',torch.from_numpy(film))
        self.movie=nn.Embedding(len(film),32)
        nn.init.normal_(self.movie.weight,std=.05)
        self.project=nn.Linear(film.shape[1]+32,48)
        self.context=nn.Sequential(nn.Linear(44,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,48))
        self.context_bias=nn.Parameter(torch.zeros(4,48))
        self.output=nn.Sequential(nn.Linear(48*6+2,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.17),
                                  nn.Linear(192,96),nn.SiLU(),nn.Linear(96,7))
    def forward(self,x,m,e):
        v=self.project(torch.cat((self.geometry[m],self.movie(m)),1))
        z=self.context(x.reshape(-1,4,44))+self.context_bias
        w=(z*v[:,None,:]).sum(2)/math.sqrt(48)
        pooled=(z*w.softmax(1)[:,:,None]).sum(1)
        return self.output(torch.cat((v,pooled,(z*v[:,None,:]).flatten(1),e),1))


def run(train,phase):
    data=Data(train,phase)
    data.setup_masks()
    torch.manual_seed(20261129)
    net=Net(data.film)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.004)
    stars=torch.arange(1,6,dtype=torch.float32)
    start=time.monotonic()
    for ep in range(EPOCHS):
        net.train()
        rng=np.random.default_rng(8723+ep)
        opt.param_groups[0]['lr']=.0001+.0009*(1+math.cos(ep*math.pi/EPOCHS))*.5
        for step in range(STEPS):
            rows,inputs=data.training_batch(rng,(ep+step)%4)
            pred=net(*inputs)
            y=torch.from_numpy(data.r[rows].astype('int64')-1)
            aux=torch.from_numpy(np.column_stack((data.re[rows],data.ce[rows])))
            mean=pred[:,:5].softmax(1)@stars
            loss=.55*(mean-y.float()-1).square().mean()+.45*F.cross_entropy(pred[:,:5],y)+.20*F.mse_loss(pred[:,5:],aux)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(net.state_dict(),OUT/(train+'.pt'))
        print(train,'epoch',ep+1,'elapsed',round(time.monotonic()-start),flush=True)
    net.eval();v=np.empty((len(data.qi),3),'f')
    with torch.inference_mode():
        for a in range(0,len(v),4096):
            ids=np.arange(a,min(len(v),a+4096),dtype='i')
            pred=net(*data.batch(ids,False))
            v[ids]=torch.column_stack((pred[:,:5].softmax(1)@stars,pred[:,5:])).numpy()
    assert np.isfinite(v).all()
    np.save(OUT/(phase+'.npy'),v)
    return v.astype(float)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_matched_masks.cpp','-o',str(OUT/'mask.dylib')],check=True)
    torch.set_num_threads(3)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    x=run('trainx','probex')
    single=ne.rmse(x[:,0],y)
    x[:,0]-=base
    held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base)
    orig=blend.metrics(base,y,f);stats=blend.metrics(ne.clipped(base+held),y,f)
    ok=retained.qualifies(stats,orig)
    np.save(OUT/'oof.npy',held)
    report=dict(selected='candidate' if ok else 'control',known=stats,control=orig,parent=str(parent),standalone=single,
                weights=w.tolist(),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        qx=run('fulltrain','qual');qx[:,0]-=qb
        delta=qx@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
