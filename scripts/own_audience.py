"""Train a movie-centered ordinal model on the ratings of similar audience members."""
import ctypes
import json
import math
import subprocess
import time
import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
from own_empirical_neighborhood import fit
from own_query_sets import projection

OUT=ne.ROOT/'own-audience-20261126'
K=20
BATCH=512
EPOCHS=14
STEPS=480


def ptrs(*a):return [v.ctypes.data for v in a]


def lib():
    dll=ctypes.CDLL(str((OUT/'audience.dylib').resolve()));it=ctypes.c_int;pt=ctypes.c_void_p
    for name,nums,pointers in [('audience_add',1,4),('audience_group',2,5),('audience_neighbors',2,11),('cf_totals',2,7)]:
        fn=getattr(dll,name);fn.argtypes=[it]*nums+[pt]*pointers;fn.restype=None
    return dll


def geometry():
    # All watched movie identities are available in either rating split. Anchor their coordinate system to trainx.
    mat=np.column_stack((projection(ne.verified_rank('trainx',True),12),projection(cf.checkpoint('trainx')['p'].mean(0),8)))
    count=ne.history('trainx','item_cnts');mat=np.ascontiguousarray(mat/(1+np.log1p(count[:,None])*.13),'f')
    nu=len(ne.history('trainx','user_cnts'));x=np.zeros((nu,20),'f');n=np.zeros(nu,'i')
    dll=lib()
    for phase in ('trainx','probex','qual'):
        u,i=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        dll.audience_add(len(i),*ptrs(u,i,mat,x));n+=np.bincount(u,minlength=nu).astype('i')
    x/=np.maximum(n[:,None],1)
    x-=x.mean(0);x/=np.maximum(x.std(0),1e-5)
    x=np.ascontiguousarray(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-5))
    sample=np.random.default_rng(1411).choice(nu,40000,replace=False)
    cent,_=kmeans2(x[sample],96,iter=20,minit='++',seed=481)
    cent/=np.maximum(np.linalg.norm(cent,axis=1,keepdims=True),1e-5)
    cl=np.empty(nu,'i')
    for j in range(0,nu,8192):cl[j:j+8192]=np.argmax(x[j:j+8192]@cent.T,axis=1)
    nearest=np.ascontiguousarray(np.argsort(-(cent@cent.T),axis=1)[:,:4],'i')
    return x,cl,nearest


class Data:
    def __init__(self,train,phase,geom):
        self.u,self.i,self.d,self.r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
        self.qu,self.qi,self.qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc,self.ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
        self.rank=ne.verified_rank(train);self.nade=full.cf96(train);self.native=full.cf96(phase)
        self.geo,self.cl,self.nearest=geom;self.lib=lib()
        self.off=np.r_[0,self.uc.cumsum()].astype('i');self.ready=np.flatnonzero(self.uc>=2)
        self.film_offset=np.zeros(len(self.ic)*96+1,'i');self.order=np.empty(len(self.i),'i')
        self.lib.audience_group(len(self.i),len(self.ic),*ptrs(self.u,self.i,self.cl,self.film_offset,self.order))
        s=cf.checkpoint(train);self.s=s;self.a=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
        self.total=np.empty((len(self.uc),96),'f');self.md=np.empty(len(self.uc),'f')
        self.lib.cf_totals(len(self.uc),len(self.ic),*ptrs(self.off,self.i,self.d,self.r,self.a,self.total,self.md))
        mean=self.d.mean(dtype=float);self.md[self.uc==0]=mean
        self.sd=(sum(np.sum((self.d[j:j+1000000].astype(float)-mean)**2) for j in range(0,len(self.d),1000000))/len(self.d))**.5
        self.movie=projection(s['p'].mean(0),24)
        print(train,'prepared',flush=True)
    def rows(self,rng,size):
        us=rng.choice(self.ready,size//2)
        return np.r_[self.off[us]+(rng.random(len(us))*self.uc[us]).astype('i'),rng.integers(0,len(self.i),size-len(us))].astype('i')
    def neighbors(self,rows,training,seed):
        us,it,day=(self.u,self.i,self.d) if training else (self.qu,self.qi,self.qd)
        q=[np.ascontiguousarray(a[rows]) for a in (us,it,day)]
        result=np.empty((len(rows),K),'i')
        self.lib.audience_neighbors(len(rows),seed,*ptrs(*q,self.u,self.d,self.cl,self.nearest,self.geo,self.film_offset,self.order,result))
        return result
    def batch(self,rows,nb,training):
        us,it,day=(self.u,self.i,self.d) if training else (self.qu,self.qi,self.qd)
        u,m,d=us[rows],it[rows],day[rows];valid=nb>=0;ix=nb.clip(min=0)
        ou=self.u[ix];stars=self.r[ix]
        assert not np.any((ou==u[:,None])&valid)
        count=self.uc[u]-int(training);ctx=self.total[u].copy()
        if training:ctx-=self.a[self.r[rows]-1,m]
        ctx/=np.sqrt(np.maximum(count,1)).astype('f')[:,None]
        other=(self.total[ou]-self.a[stars-1,self.i[ix]])/np.sqrt(np.maximum(self.uc[ou]-1,1)).astype('f')[:,:,None]
        dev=d-self.md[u];dev=np.sign(dev)*np.abs(dev/self.sd)**.4
        bias=(self.s['b'][:,m]+self.s['user_bias'][:,u]+self.s['user_drift'][:,u]*dev).T
        decoder=np.ascontiguousarray(self.s['p'][:,m].transpose(1,0,2))
        dt=self.d[ix].astype('i')-d[:,None]
        aux=np.stack((stars-3.5,stars-self.rank[ix],stars-self.nade[ix],np.sign(dt)*np.log1p(np.abs(dt))/5,
            (dt==0).astype('f'),np.log1p(self.uc[ou])/7,np.einsum('nk,nlk->nl',self.geo[u],self.geo[ou])),axis=2)
        side=np.column_stack((self.geo[u],self.movie[m],d/2000.,dev,np.log1p(self.ic[m])/10,np.log1p(count)/7))
        return tuple(torch.from_numpy(np.ascontiguousarray(v,'f')) for v in (ctx,other,aux,side,bias,decoder,valid))


class Net(nn.Module):
    def __init__(self,s):
        super().__init__()
        for k in ('enc_proj','dec_proj','enc_bias'):self.register_buffer(k,torch.from_numpy(s[k]))
        self.profile=nn.Sequential(nn.Linear(96,24),nn.LayerNorm(24),nn.SiLU())
        self.query=nn.Sequential(nn.Linear(24+48+5,48),nn.LayerNorm(48),nn.SiLU())
        self.peer=nn.Sequential(nn.Linear(24*2+7,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,4*9))
        self.gate=nn.Linear(48,32)
        self.head=nn.Sequential(nn.Linear(48+32+4,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.12),nn.Linear(80,7))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,ctx,other,aux,side,bias,p,valid):
        z=torch.tanh(ctx@self.enc_proj.T+self.enc_bias)@self.dec_proj.T
        logits=(torch.bmm(p,z[:,:,None]).squeeze(2)+bias).cumsum(1)
        raw=logits.softmax(1);q=self.profile(ctx);o=self.profile(other)
        query=self.query(torch.cat((q,side,raw),1))
        v=self.peer(torch.cat((o,o*q[:,None],aux),2)).reshape(len(q),-1,4,9)
        scores=v[:,:,:,0]+(v[:,:,:,1:]*self.gate(query).reshape(-1,1,4,8)).sum(3)/math.sqrt(8)
        w=scores.masked_fill(~valid.bool()[:,:,None],-10000).softmax(1)*valid[:,:,None]
        pool=(w[:,:,:,None]*v[:,:,:,1:]).sum(1).flatten(1);local=(w*aux[:,:,:1]).sum(1)
        pred=self.head(torch.cat((query,pool,local),1))
        return logits+.8*pred[:,:5],raw,pred[:,5:]


def run(train,phase,geom):
    data=Data(train,phase,geom);torch.manual_seed(94116)
    model=Net(data.s);opt=torch.optim.AdamW(model.parameters(),lr=.00075,weight_decay=.005)
    stars=torch.arange(1,6,dtype=torch.float32)
    started=time.monotonic()
    for ep in range(EPOCHS):
        rng=np.random.default_rng(51913+ep);model.train()
        opt.param_groups[0]['lr']=.00012+.00065*(1+math.cos(ep*math.pi/EPOCHS))/2
        for section in range(STEPS//32):
            rows=data.rows(rng,BATCH*32);neighbors=data.neighbors(rows,True,8941+ep*29+section)
            for lo in range(0,len(rows),BATCH):
                ix=rows[lo:lo+BATCH];inputs=data.batch(ix,neighbors[lo:lo+BATCH],True)
                logits,raw,aux=model(*inputs);y=torch.from_numpy(data.r[ix].astype('int64')-1)
                mu=logits.softmax(1)@stars
                targets=torch.from_numpy(np.column_stack((data.r[ix]-data.rank[ix],data.r[ix]-data.nade[ix])))
                loss=.5*F.mse_loss(mu,y.float()+1)+.42*F.cross_entropy(logits,y)+.12*F.mse_loss(aux,targets)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        torch.save(model.state_dict(),OUT/(train+'.pt'))
        print(train,'epoch',ep+1,'seconds',round(time.monotonic()-started),flush=True)
    model.eval();outputs=np.zeros((len(data.qi),3),'f')
    with torch.inference_mode():
        for start in range(0,len(outputs),32768):
            rows=np.arange(start,min(start+32768,len(outputs)),dtype='i')
            for seed in (803,917,1049):
                nb=data.neighbors(rows,False,seed)
                for lo in range(0,len(rows),1024):
                    ix=rows[lo:lo+1024];logits,raw,aux=model(*data.batch(ix,nb[lo:lo+1024],False))
                    assert np.max(np.abs((raw@stars).numpy()-data.native[ix]))<.0003
                    outputs[ix]+=torch.column_stack((logits.softmax(1)@stars,aux)).numpy()/3
    np.save(OUT/(phase+'.npy'),outputs);return outputs


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_audience.cpp','scripts/own_cf_adapter.cpp',
                    '-o',str(OUT/'audience.dylib')],check=True)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    geom=geometry();v=run('trainx','probex',geom)
    def dirs(z,b,phase):return np.column_stack((z[:,0]-b,z[:,0]-full.cf96(phase),z[:,1:])).astype(float)
    x=dirs(v,base,'probex');held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);orig=blend.metrics(base,y,f);stats=blend.metrics(ne.clipped(base+held),y,f)
    ok=retained.qualifies(stats,orig);np.save(OUT/'oof.npy',held)
    rep=dict(selected='candidate' if ok else 'control',known=stats,control=orig,parent=str(parent),weights=w.tolist(),
             standalone=ne.rmse(v[:,0],y),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        delta=dirs(run('fulltrain','qual',geom),qb,'qual')@w
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
