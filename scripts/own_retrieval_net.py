"""Predict masked stars by attending to each user's most relevant observed movies."""
import argparse
import ctypes
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_neighbor_experiment as ne
import own_full_context as full
import own_anchor_context as anchor
import own_session_latent as latent
from own_kernel_family import gram

K=32
DIM=32


class Dataset:
    def __init__(self,train,query,out):
        self.i,self.d,self.r,self.u=[ne.history(train,k) for k in ('item_idxs','dates','ratings','user_idxs')]
        self.qi,self.qd,self.qu=[ne.history(query,k) for k in ('item_idxs','dates','user_idxs')]
        self.uc=ne.history(train,'user_cnts')
        self.off=np.r_[0,self.uc.cumsum()].astype('i')
        self.eligible=np.flatnonzero(self.uc>=2)
        self.p=ne.verified_rank(train)
        self.cf=full.cf96(train)
        mat=ne.verified_rank(train,True)
        _,vec=np.linalg.eigh(mat.T@mat)
        rank=mat@vec[:,-DIM:]
        rank/=np.maximum(rank.std(axis=0),1e-6)
        graph_path=latent.OUT/'graph-coordinates.npy'
        assert ne.sha(graph_path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
        scene=np.load(graph_path)
        self.films=np.ascontiguousarray(np.clip(np.column_stack((rank,scene[:,-DIM:])), -8,8),dtype='f')
        self.gram,self.scene=gram(mat),gram(scene)
        self.ic=ne.history(train,'item_cnts')
        self.year=ne.history(train,'item_years')
        self.globalvec=np.ascontiguousarray(self.films[:,:DIM])
        self.pr=np.zeros((len(self.uc),2*DIM+5),'f')
        self.lib=ctypes.CDLL(str((out/'retrieval.dylib').resolve()))
        it,pt=ctypes.c_int,ctypes.c_void_p
        self.lib.retrieve.argtypes=[it]*3+[pt]*9
        self.lib.global_profile.argtypes=[it]*2+[pt]*5
        self.lib.retrieve.restype=self.lib.global_profile.restype=None
        self.lib.global_profile(len(self.uc),DIM,*[x.ctypes.data for x in (self.off,self.i,self.r,self.globalvec,self.pr)])
        self.args=[x.ctypes.data for x in (self.off,self.i,self.d,self.gram,self.scene)]
        self.mu=float(self.r.mean())

    def sample(self,rng,size):
        # Equal-user and equal-rating sampling cover sparse and dense profiles.
        n=size//2
        users=rng.choice(self.eligible,n)
        uniform=self.off[users]+(rng.random(n)*self.uc[users]).astype('i')
        rows=np.r_[uniform,rng.integers(0,len(self.i),size-n)]
        return rows.astype('i')

    def neighbors(self,u,i,d):
        order=np.argsort(i,kind='stable')
        us,items,days=[np.ascontiguousarray(v[order]) for v in (u,i,d)]
        tmp=np.empty((len(i),K),'i')
        def process(lo):
            sl=slice(lo,min(len(i),lo+4096))
            self.lib.retrieve(len(us[sl]),K,len(self.ic),*[v[sl].ctypes.data for v in (us,items,days)],
                              *self.args,tmp[sl].ctypes.data)
        with ThreadPoolExecutor(2) as ex:list(ex.map(process,range(0,len(i),4096)))
        result=np.empty_like(tmp)
        result[order]=tmp
        return result

    def batch(self,rows,neighbors,training):
        u,i,d=(self.u[rows],self.i[rows],self.d[rows]) if training else (self.qu[rows],self.qi[rows],self.qd[rows])
        idx=np.maximum(neighbors,0)
        mask=neighbors>=0
        other=self.i[idx]
        assert not np.any((other==i[:,None])&mask)
        r=self.r[idx]
        dt=(self.d[idx]-d[:,None]).astype('f')
        re=r-self.p[idx];ce=r-self.cf[idx]
        input_values=np.stack((np.sign(dt)*np.log1p(np.abs(dt))/6,dt==0,re,ce,(r-3.6),
            self.gram[i[:,None],other]/16000.,self.scene[i[:,None],other]/16000.),axis=-1).astype('f')
        counts=self.uc[u].astype('f')-int(training)
        prof=self.pr[u].copy()
        y=self.r[rows].astype('i') if training else None
        if training:
            vec=self.globalvec[i]
            prof[:,:DIM]-=vec
            prof[:,DIM:2*DIM]-=(y[:,None]-3.6)*vec
            prof[np.arange(len(i)),2*DIM+y-1]-=1
        prof[:,:2*DIM]/=np.sqrt(counts[:,None]+20)
        prof[:,2*DIM:]/=np.maximum(counts[:,None],1)
        prof=np.column_stack((np.clip(prof,-20,20),np.log1p(counts)/7,d/2000.,np.log1p(self.ic[i])/10.,
                              (self.year[i]-1980)/30.)).astype('f')
        args=(i.astype('i'),other.astype('i'),r.astype('i'),mask,input_values,prof)
        return tuple(torch.as_tensor(x,device='mps') for x in args),None if y is None else torch.as_tensor(y-1,device='mps').long()


class Net(nn.Module):
    def __init__(self,data):
        super().__init__()
        width=data.films.shape[1]
        self.emb=nn.Embedding.from_pretrained(torch.from_numpy(data.films),freeze=False)
        self.star=nn.Embedding(6,12)
        self.values=nn.Sequential(nn.Linear(2*width+19,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,56))
        self.profile=nn.Sequential(nn.Linear(2*DIM+9,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,48))
        self.query=nn.Linear(width+48,48)
        self.out=nn.Sequential(nn.Linear(width+48+48+8,160),nn.LayerNorm(160),nn.GELU(),nn.Dropout(.12),
                               nn.Linear(160,96),nn.GELU(),nn.Linear(96,7))
    def forward(self,items,others,ratings,valid,ctx,profile):
        qe=self.emb(items);other=self.emb(others)
        user=self.profile(profile)
        h=self.values(torch.cat((other,other*qe[:,None,:],self.star(ratings),ctx),-1))
        v,logit=h[:,:,:48],h[:,:,48:]
        query=self.query(torch.cat((qe,user),-1)).reshape(-1,1,8,6)
        vec=v.reshape(*v.shape[:2],8,6)
        a=(logit+(vec*query).sum(-1)/math.sqrt(6.)).masked_fill(~valid[:,:,None],-10000).softmax(1)*valid[:,:,None]
        pool=(vec*a[:,:,:,None]).sum(1).reshape(len(items),48)
        local_r=(a*ctx[:,:,4:5]).sum(1)
        return self.out(torch.cat((qe,user,pool,local_r),-1))


def run(out,train,query,args):
    data=Dataset(train,query,out)
    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    model=Net(data).to('mps')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.002)
    stamp=time.monotonic()
    for ep in range(args.epochs):
        rng=np.random.default_rng(args.seed+ep)
        model.train()
        opt.param_groups[0]['lr']=.00015+.00085*(1+math.cos(ep*math.pi/args.epochs))/2
        for section in range(args.sections):
            rows=data.sample(rng,args.batch*64)
            neighbors=data.neighbors(data.u[rows],data.i[rows],data.d[rows])
            for a in range(0,len(rows),args.batch):
                sl=slice(a,a+args.batch)
                xx,y=data.batch(rows[sl],neighbors[sl],True)
                outv=model(*xx)
                logits=outv[:,:5]
                mean=logits.softmax(-1)@torch.arange(1,6,device='mps',dtype=torch.float32)
                # Auxiliary heads predict parent residuals using the other
                # movies' history only. The query's fitted parent is a label.
                aux=np.column_stack((data.r[rows[sl]]-data.p[rows[sl]],data.r[rows[sl]]-data.cf[rows[sl]]))
                loss=.4*F.cross_entropy(logits,y)+.6*F.mse_loss(mean,y.float()+1)+.15*F.mse_loss(outv[:,5:],torch.as_tensor(aux,device='mps'))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
                opt.step()
            if section%4==0:print(train,ep,section,round(float(loss.detach().cpu()),4),round(time.monotonic()-stamp),flush=True)
        torch.save(model.state_dict(),out/(train+'.pt'))
    model.eval()
    values=np.empty((len(data.qi),3),'f')
    with torch.no_grad():
        for lo in range(0,len(data.qi),65536):
            rows=np.arange(lo,min(lo+65536,len(data.qi)))
            nb=data.neighbors(data.qu[rows],data.qi[rows],data.qd[rows])
            for a in range(0,len(rows),args.batch):
                sl=slice(a,a+args.batch)
                xx,_=data.batch(rows[sl],nb[sl],False)
                outv=model(*xx)
                mean=outv[:,:5].softmax(-1)@torch.arange(1,6,device='mps',dtype=torch.float32)
                values[rows[sl]]=torch.column_stack((mean,outv[:,5:])).cpu().numpy()
    assert np.isfinite(values).all()
    np.save(out/(query+'.npy'),values)
    return values.astype(float)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--sections',type=int,default=14)
    ap.add_argument('--batch',type=int,default=1024)
    ap.add_argument('--seed',type=int,default=20261128)
    args=ap.parse_args()
    out=ne.ROOT/'own-retrieval-net-20261126'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_retrieval.cpp','-o',str(out/'retrieval.dylib')],check=True)
    from own_kernel_family import control
    from own_empirical_neighborhood import fit
    y,f,p,q=control()
    x=run(out,'trainx','probex',args)
    standalone=ne.rmse(x[:,0],y)
    x[:,0]-=p
    held=p.copy()
    for fold in range(5):
        tr=f!=fold
        held[~tr]=ne.clipped(p[~tr]+x[~tr]@fit(x[tr],(y-p)[tr]))
    w=fit(x,y-p)
    gains=[ne.rmse(p[f==j],y[f==j])-ne.rmse(held[f==j],y[f==j]) for j in range(5)]
    ok=sorted(gains)[1]>0 and ne.rmse(held,y)<ne.rmse(p,y)
    report=dict(selected='candidate' if ok else 'control',known_rmse=ne.rmse(held,y),fold_gains=gains,standalone=standalone,
                weights=w.tolist(),config=vars(args),qualifying_hashes={},known_comparison_is_nested=False,qualifying_labels_used_for_fitting=False)
    np.save(out/'oof.npy',held-p)
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if ok:
        qx=run(out,'fulltrain','qual',args)
        qx[:,0]-=q
        delta=qx@w
        np.save(out/'qual-correction.npy',delta)
        path=out/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
