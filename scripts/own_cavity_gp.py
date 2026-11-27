"""Learn a time-dependent residual covariance on jointly held-out real ratings around CFNADE."""
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

OUT=ne.ROOT/'own-cavity-gp-20261126'
K=9
STARS=np.arange(1,6,dtype='f')


class Data:
    def __init__(self,train,phase):
        self.i,self.d,self.r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
        self.qu,self.qi,self.qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
        self.off=np.r_[0,self.uc.cumsum()].astype('i')
        self.ready=np.flatnonzero(self.uc>=3)
        qu,qd=self.qu,self.qd
        if phase=='probex':
            qu=np.r_[qu,ne.history('qual','user_idxs')];qd=np.r_[qd,ne.history('qual','dates')]
        self.qoff=np.r_[0,np.bincount(qu,minlength=len(self.uc)).cumsum()].astype('i')
        self.qdays=qd[np.argsort(qu,kind='stable')]
        u,m,d,_=anchor.old_probe();seen,idx=anchor.training_indices(train,u,m,d)
        self.aoff=np.r_[0,np.bincount(u[seen],minlength=len(self.uc)).cumsum()].astype('i')
        self.arows=idx[np.argsort(u[seen],kind='stable')].astype('i')
        self.lib=ctypes.CDLL(str((OUT/'cavity.dylib').resolve()));it,pt=ctypes.c_int,ctypes.c_void_p
        self.lib.cavity_group.argtypes=[it]*5+[pt]*13;self.lib.cavity_group.restype=None
        self.lib.cf_totals.argtypes=[it]*2+[pt]*7;self.lib.cf_totals.restype=None
        s=cf.checkpoint(train);a=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
        tot=np.empty((len(self.uc),96),'f');self.md=np.empty(len(self.uc),'f')
        self.lib.cf_totals(len(self.uc),len(ic),*[x.ctypes.data for x in (self.off,self.i,self.d,self.r,a,tot,self.md)])
        avg=self.d.mean(dtype=float);self.md[self.uc==0]=avg
        self.sd=(sum(np.sum((self.d[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(self.d),1000000))/len(self.d))**.5
        self.total=tot@s['enc_proj'].T
        self.ep=np.ascontiguousarray((a@s['enc_proj'].T).transpose(1,0,2))
        self.op=np.ascontiguousarray((s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2))
        self.b,self.ub,self.ud=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
        self.eb=s['enc_bias'];self.pop=np.log1p(ic).astype('f')/9
        self.geo=np.ascontiguousarray(np.column_stack((projection(s['a'].mean(0),12),projection(s['p'].mean(0),12))))
        self.geo/=np.maximum(np.linalg.norm(self.geo,axis=1,keepdims=True),1e-8)
        self.native=full.cf96(phase)
        print(train,'data ready',flush=True)

    def batch(self,users,mode,seed,targets=None):
        u=users.astype('i');tr=targets is None
        t=np.zeros(len(u),'i') if tr else targets.astype('i')
        rr=np.full((len(u),K),-1,'i')
        args=(u,t,self.qi,self.qd,self.off,self.i,self.d,self.geo,self.qoff,self.qdays,self.aoff,self.arows,rr)
        self.lib.cavity_group(len(u),K,seed,mode,int(tr),*[z.ctypes.data for z in args])
        observed=rr>=0;idx=rr.clip(min=0)
        movies=self.i[idx].copy();days=self.d[idx].copy();stars=self.r[idx].copy()
        valid=observed.copy()
        if not tr:
            movies[:,0]=self.qi[t];days[:,0]=self.qd[t];stars[:,0]=0;valid[:,0]=True
        enc=self.ep[movies,np.maximum(0,stars.astype('i')-1)]
        context=self.total[u]-(enc*observed[:,:,None]).sum(1)
        n=(self.uc[u]-observed.sum(1)).clip(min=1).astype('f')
        hidden=np.tanh(context/np.sqrt(n)[:,None]+self.eb)
        diff=(days-self.md[u,None])/self.sd
        dev=np.sign(diff)*abs(diff)**.4
        bias=self.b[movies]+self.ub[u,None]+self.ud[u,None]*dev.astype('f')[:,:,None]
        op=self.op[movies]
        z=np.einsum('bksh,bh->bks',op,hidden)+bias;z-=z.max(2,keepdims=True)
        p=np.exp(z);p/=p.sum(2,keepdims=True)
        mu=p@STARS
        if not tr:
            whole=np.tanh(self.total[u]/np.sqrt(self.uc[u].clip(min=1)).astype('f')[:,None]+self.eb)
            raw=np.einsum('bsh,bh->bs',op[:,0],whole)+bias[:,0]
            raw=np.exp(raw-raw.max(1,keepdims=True));raw/=raw.sum(1,keepdims=True)
            err=float(np.max(abs(raw@STARS-self.native[t])))
            assert err<.0003,err
        feats=np.concatenate((self.geo[movies],p,self.pop[movies,None],days[:,:,None]/2000.,
            np.broadcast_to(np.log1p(self.uc[u])[:,None,None]/7,(*movies.shape,1)),dev[:,:,None]),2)
        cv=lambda a:torch.from_numpy(np.ascontiguousarray(a))
        return cv(feats.astype('f')),cv(movies.astype('int64')),cv(days.astype('f')),cv(valid),cv(stars.astype('f')),cv(mu)


class Net(nn.Module):
    def __init__(self,nm):
        super().__init__()
        self.movie=nn.Embedding(nm,12);nn.init.normal_(self.movie.weight,std=.13)
        self.net=nn.Sequential(nn.Linear(33+12,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,39))
        self.register_buffer('decay',torch.tensor([.001,1/110,1/5]))
    def forward(self,features,movies,days,valid,stars,mu):
        a=self.net(torch.cat((features,self.movie(movies)),2))
        prior=mu+.5*torch.tanh(a[:,:,0])
        noise=.14+F.softplus(a[:,:,1])
        factors=.44*torch.tanh(a[:,:,2:38].reshape(*mu.shape,3,12)).permute(0,2,1,3)
        cov=factors@factors.transpose(-1,-2)
        gap=(days[:,:,None]-days[:,None,:]).abs()
        cov=(cov*torch.exp(-gap[:,None]*self.decay[None,:,None,None])).sum(1)
        cov=cov+torch.diag_embed(noise+25*(~valid))
        e=(stars-prior)*valid
        sub=cov[:,1:,1:]
        factor=torch.linalg.cholesky(sub)
        cross=cov[:,1:,0]*valid[:,1:]
        gain=torch.cholesky_solve(cross[:,:,None],factor).squeeze(2)
        adjustment=(gain*e[:,1:]).sum(1)
        variance=(cov[:,0,0]-(gain*cross).sum(1)).clamp(min=.1)
        posterior=prior[:,0]+adjustment
        nll=(posterior-stars[:,0]).square()/variance+variance.log()
        joint=torch.cholesky_solve(e[:,1:,None],factor).squeeze(2)
        det=2*factor.diagonal(dim1=1,dim2=2).log().sum(1)
        peer_nll=((e[:,1:]*joint).sum(1)+det)/valid[:,1:].sum(1).clamp(min=1)
        return posterior,prior[:,0],variance,adjustment,nll,peer_nll


def run(train,phase):
    data=Data(train,phase);torch.manual_seed(51673);torch.set_num_threads(2)
    net=Net(len(data.pop));opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.025)
    started=time.monotonic()
    for ep in range(12):
        rng=np.random.default_rng(72413+ep);net.train()
        opt.param_groups[0]['lr']=.00014+.00086*(1+math.cos(ep*math.pi/12))/2
        for j in range(500):
            u=rng.choice(data.ready,192,replace=False)
            args=data.batch(u,j%2,int(rng.integers(1,2**30)))
            pred,prior,_,_,nll,peer_nll=net(*args)
            loss=.52*(pred-args[4][:,0]).square().mean()+.29*nll.mean()+.13*peer_nll.mean()+.06*(prior-args[4][:,0]).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
        print(train,'epoch',ep+1,round(time.monotonic()-started),flush=True)
    torch.save(net.state_dict(),OUT/(train+'.pt'));net.eval()
    result=np.empty((len(data.qi),12),'f')
    with torch.no_grad():
        for j in range(0,len(result),512):
            rows=np.arange(j,min(len(result),j+512))
            for mode in range(2):
                args=data.batch(data.qu[rows],mode,567,rows)
                post,prior,v,delta,_,_=net(*args)
                result[rows,6*mode:6*mode+6]=np.column_stack((post,prior,v,delta,args[5][:,0],args[3].sum(1)))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cavity_gp.cpp','scripts/own_cf_adapter.cpp',
                    '-o',str(OUT/'cavity.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    def design(train,phase,b):
        z=run(train,phase);np.save(OUT/(phase+'.npy'),z)
        c=full.cf96(phase);rank=ne.verified_rank(phase);u,i=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        return np.column_stack((b,c-b,rank-b,z[:,0]-c,z[:,6]-c,z[:,1]-c,z[:,7]-c,
            np.log1p(ne.history(train,'user_cnts')[u]),np.log1p(ne.history(train,'item_cnts')[i]),z)).astype('f')
    x=design('trainx','probex',base)
    def model():return CatBoostRegressor(iterations=850,depth=5,learning_rate=.045,l2_leaf_reg=60,thread_count=2,
            random_seed=3711,verbose=False,allow_writing_files=False)
    key,known,held=anchor.crossfit(x,y,f,base,model)
    rep=dict(parent=str(parent),selected=key,known=known,qualifying_hashes={},qualifying_labels_used_for_fitting=False,
             known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        np.save(OUT/'oof.npy',held)
        qx=design('fulltrain','qual',qb)
        delta=model().fit(x,y-base).predict(qx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
