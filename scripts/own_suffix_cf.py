"""Learn how a user's ordinal preference distribution changes over a held-out terminal cohort."""
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
import own_retained_history as ret
from own_query_sets import projection
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-suffix-cf-20261126'
Q=9


class Data:
    def __init__(self,train,phase,lib):
        self.lib=lib;self.native=full.cf96(phase)
        u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
        self.nu=len(ne.history(train,'user_cnts'));ic=ne.history(train,'item_cnts')
        order=np.lexsort((i,d,u));self.i,self.d,self.r=[v[order] for v in (i,d,r)];del order
        self.off=np.r_[0,ne.history(train,'user_cnts').cumsum()].astype('i8')
        self.ready=np.flatnonzero(np.diff(self.off)>=2);self.avg=d.mean(dtype=float)
        self.sd=(sum(np.sum((d[j:j+1000000].astype(float)-self.avg)**2) for j in range(0,len(d),1000000))/len(d))**.5
        ou,oi,od,_=anchor.old_probe();seen,idx=anchor.training_indices(train,ou,oi,od)
        users,self.fi,self.fd=[np.r_[a,ne.history('qual',k)].astype(t) for a,k,t in zip((ou,oi,od),('user_idxs','item_idxs','dates'),('i','i2','i2'))]
        self.obs=np.zeros(len(users),'i1');self.obs[np.flatnonzero(seen)]=r[idx]
        rows=np.argsort(users,kind='stable');sz=np.bincount(users,minlength=self.nu);off=np.r_[0,sz.cumsum()]
        assert sz.max()==Q
        self.grid=np.full((self.nu,Q),-1,'i');self.grid[users[rows],np.arange(len(rows))-off[users[rows]]]=rows
        self.boundary=np.full(self.nu,30000,'i2');np.minimum.at(self.boundary,users,self.fd)
        target=ret.map_rows() if phase=='probex' else np.arange(len(ou),len(users))
        self.lookup=np.full(len(users),-1,'i');self.lookup[target]=np.arange(len(target))
        self.eval_users=np.unique(users[target])
        s=cf.checkpoint(train);self.enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f');self.eproj=s['enc_proj'];self.eb=s['enc_bias']
        self.op=np.ascontiguousarray((s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2))
        self.b,self.ub,self.ud=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
        self.geo=np.column_stack((projection(s['a'].mean(0),12),projection(s['p'].mean(0),12))).astype('f')
        self.pop=np.log1p(ic).astype('f')/9
        print(train,'prepared',flush=True)
    def batch(self,us,seed,train,return_state=False):
        users=np.ascontiguousarray(us,'i');n=len(us)
        movies=np.full((n,Q),-1,'i');dates=np.zeros((n,Q),'f');anc=np.zeros((n,Q),'i');target=np.full((n,Q),-1,'i')
        ctx=np.zeros((n,4,96),'f');norms=np.zeros((n,4),'f');stat=np.zeros((n,6),'f')
        args=(users,self.off,self.i,self.d,self.r,self.boundary,self.grid,self.fi,self.fd,self.obs,self.enc,movies,dates,anc,target,ctx,norms,stat)
        self.lib.suffix_cf(n,seed,int(train),len(self.pop),*[z.ctypes.data for z in args])
        valid=movies>=0;movies=movies.clip(min=0)
        hidden=np.tanh(ctx@self.eproj.T+self.eb)
        z=np.einsum('bqsh,blh->bqls',self.op[movies],hidden)
        md=np.where(stat[:,5]>0,stat[:,0],self.avg);diff=(dates-md[:,None])/self.sd
        dev=np.sign(diff)*abs(diff)**.4
        bias=self.b[movies]+self.ub[users,None]+self.ud[users,None]*dev[:,:,None]
        logits=z+bias[:,:,None,:]
        gap=(dates-stat[:,1,None]);last=np.max(np.where(valid,dates,-1),axis=1)
        rel=lambda x:np.sign(x)*np.log1p(abs(x))/5
        feats=np.stack((self.pop[movies],dates/2000,dev,rel(gap),rel(dates-last[:,None]),rel(dates-stat[:,3,None]),
                 np.broadcast_to(stat[:,4,None]/4,dates.shape),np.broadcast_to(np.log1p(stat[:,5,None])/7,dates.shape)),axis=2)
        cv=lambda x:torch.from_numpy(np.ascontiguousarray(x))
        values=tuple(cv(v) for v in (movies.astype('int64'),dates,valid,anc.astype('f'),feats.astype('f'),logits.astype('f')))
        if return_state:values+=tuple(cv(v) for v in (ctx,norms,bias.astype('f')))
        return values,cv(target.astype('int64'))


class Net(nn.Module):
    def __init__(self,geo):
        super().__init__();self.register_buffer('geo',torch.from_numpy(geo),persistent=False)
        self.movie=nn.Embedding(len(geo),12);nn.init.normal_(self.movie.weight,std=.13)
        self.token=nn.Sequential(nn.Linear(24+12+8+20+2,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,48),nn.SiLU())
        self.key=nn.Linear(48,24);self.query=nn.Linear(48,24);self.value=nn.Linear(48,36)
        self.head=nn.Sequential(nn.Linear(48+36,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.16),nn.Linear(64,8))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.scale=nn.Parameter(torch.tensor([.1,.7,1.9]));self.register_buffer('stars',torch.arange(1,6,dtype=torch.float32))
    def forward(self,m,d,valid,anc,features,logits):
        p=logits.softmax(3);raw=logits[:,:,0];mu=p[:,:,1]@self.stars
        z=self.token(torch.cat((self.geo[m],self.movie(m),features,p.flatten(2),(anc>0).float()[:,:,None],((anc-mu)*(anc>0))[:,:,None]),2))
        qq=self.query(z).reshape(-1,Q,3,8).transpose(1,2);kk=self.key(z).reshape(-1,Q,3,8).transpose(1,2)
        v=self.value(z).reshape(-1,Q,3,12).transpose(1,2)
        dt=(d[:,:,None]-d[:,None,:]).abs()
        a=qq@kk.transpose(-1,-2)/math.sqrt(8)-self.scale[None,:,None,None]*torch.log1p(dt[:,None]/16)
        keep=valid[:,None,:]&(m[:,:,None]!=m[:,None,:])
        w=a.masked_fill(~keep[:,None],-10000).softmax(-1)*keep[:,None]
        context=(w@v).transpose(1,2).reshape(-1,Q,36)
        o=self.head(torch.cat((z,context),2)).tanh()
        contrasts=logits[:,:,1:]-raw[:,:,None,:];contrasts-=contrasts.mean(-1,keepdim=True)
        out=raw+.45*torch.einsum('bql,bqls->bqs',o[:,:,5:],contrasts.clamp(-5,5))+.90*o[:,:,:5]
        return out,p[:,:,0],p@self.stars


def run(train,phase,lib):
    d=Data(train,phase,lib);torch.manual_seed(72031);net=Net(d.geo)
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.025);start=time.monotonic()
    for ep in range(18):
        rng=np.random.default_rng(89317+ep);net.train()
        opt.param_groups[0]['lr']=.00014+.00076*(1+math.cos(ep*math.pi/18))/2
        for j in range(520):
            u=rng.choice(d.ready,160,replace=False);args,y=d.batch(u,int(rng.integers(1,2**30)),True)
            out,raw,_=net(*args);active=y>=0;prob=out[active].softmax(1);mu=prob@net.stars
            loss=.62*(mu-y[active]-1).square().mean()+.38*F.cross_entropy(out[active],y[active])+.08*(mu-raw[active]@net.stars).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.5);opt.step()
        print(train,'epoch',ep+1,round(time.monotonic()-start),flush=True)
    torch.save(net.state_dict(),OUT/(train+'.pt'));net.eval();result=np.empty((len(d.native),10),'f');maxerr=0.
    with torch.inference_mode():
        for j in range(0,len(d.eval_users),384):
            u=d.eval_users[j:j+384];args,_=d.batch(u,170,False);out,raw,means=net(*args);p=out.softmax(2)
            grid=d.grid[u];rows=d.lookup[grid.clip(min=0)];keep=(grid>=0)&(rows>=0)
            err=np.max(abs(means[:,:,0].numpy()[keep]-d.native[rows[keep]]),initial=0)
            maxerr=max(maxerr,float(err));assert maxerr<.0004,maxerr
            result[rows[keep]]=torch.cat((p,(p@net.stars)[:,:,None],means),2).numpy()[keep]
    np.save(OUT/(phase+'.npy'),result)
    print(phase,'max native replay error',maxerr,flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','scripts/own_suffix_cf.cpp','-o',str(OUT/'suffix.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'suffix.dylib').resolve()));lib.suffix_cf.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*18;lib.suffix_cf.restype=None
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    v=run('trainx','probex',lib)
    def design(raw,b):return np.column_stack((raw[:,5]-b,raw[:,5]-raw[:,6],raw[:,7:10]-raw[:,6,None]))
    def tree(raw,b):return np.column_stack((raw,b,design(raw,b))).astype('f')
    model=lambda:CatBoostRegressor(iterations=600,depth=5,learning_rate=.045,l2_leaf_reg=60,thread_count=2,
         random_seed=574,allow_writing_files=False,verbose=False)
    x=tree(v,base);_,_,td=anchor.crossfit(x,y,f,base,model);z=np.column_stack((design(v,base),td))
    held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=z[~tr]@fit(z[tr],(y-base)[tr])
    w=fit(z,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);take=ret.qualifies(known,ctrl)
    rep=dict(selected='candidate' if take else 'control',parent=str(parent),known=known,control=ctrl,weights=w.tolist(),qualifying_hashes={},
        qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if take:
        qv=run('fulltrain','qual',lib);t=model().fit(x,y-base).predict(tree(qv,qb),thread_count=2)
        delta=np.column_stack((design(qv,qb),t))@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
