"""Learn how a user's other unlabeled movie queries change a held-out rating forecast."""
import ctypes
import json
import subprocess
import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F
import own_cf_context as cf
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained

OUT=ne.ROOT/'own-query-sets-20261126'
WIDTH=14


def projection(arr,k):
    a=arr-arr.mean(0)
    _,vec=np.linalg.eigh(a.T@a)
    x=a@vec[:,-k:]
    return np.clip(x/np.maximum(x.std(0),1e-5),-7,7).astype('f')


def geometry():
    qu,qi,qd=[np.concatenate([ne.history(k,field) for k in ('probex','qual')]) for field in ('user_idxs','item_idxs','dates')]
    nm=len(ne.history('trainx','item_cnts'))
    tables=[projection(cov.movie_geometry(),18)]
    ck=cf.checkpoint('trainx')
    for mat in (ck['a'],ck['p']):tables.append(projection(mat.reshape(5,nm,-1).transpose(1,0,2).reshape(nm,-1),16))
    tables.append(projection(ne.verified_rank('trainx',True),12))
    # Only query addresses enter these two implicit decompositions.
    for row in (qu,np.unique(qu.astype('int64')*4096+qd,return_inverse=True)[1]):
        rd=np.bincount(row);cd=np.bincount(qi,minlength=nm)
        w=(1/(np.maximum(rd[row],1)**.5*np.maximum(cd[qi],1)**.35)).astype('f')
        x=sparse.csr_matrix((w,(row,qi)),shape=(len(rd),nm));xt=x.T.tocsr()
        v=np.random.default_rng(921).standard_normal((nm,32),dtype='f')
        for _ in range(6):v=np.linalg.qr(xt@(x@v))[0]
        tables.append(projection(v,24))
    return np.column_stack(tables)


class Data:
    def __init__(self,phase,base):
        u,self.i,self.d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        n=len(u)
        if phase=='probex':
            u,self.i,self.d=[np.concatenate((a,ne.history('qual',k))) for a,k in zip((u,self.i,self.d),('user_idxs','item_idxs','dates'))]
        nu=len(ne.history('trainx','user_cnts'));cnt=np.bincount(u,minlength=nu)
        off=np.r_[0,cnt.cumsum()].astype('i');ix=np.argsort(u,kind='stable').astype('i')
        self.peers=np.full((n,WIDTH),-1,'i')
        lib=ctypes.CDLL(str((OUT/'sets.dylib').resolve()))
        lib.query_sets.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*6
        lib.query_sets.restype=None
        lib.query_sets(n,WIDTH,*[v.ctypes.data for v in (u,self.i,self.d,off,ix,self.peers)])
        tr='trainx' if phase=='probex' else 'fulltrain'
        ctx=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
        self.x=np.column_stack((base,full.cf96(phase)-base,ne.verified_rank(phase)-base,
            np.log1p(cnt[u[:n]]),np.log1p(ne.history(tr,'user_cnts')[u[:n]]),
            np.log1p(ne.history(tr,'item_cnts')[self.i[:n]]),ctx[:,::9])).astype('f')
        self.base=base
        self.i=torch.from_numpy(np.array(self.i,'int32'));self.d=torch.from_numpy(np.array(self.d,'int32'))
        self.peers=torch.from_numpy(self.peers);self.x=torch.from_numpy(self.x)

    def batch(self,ids):
        ix=self.peers[ids].long();mask=ix>=0;ix=ix.clamp(min=0)
        dt=(self.d[ix]-self.d[ids,None]).float()/28
        tm=torch.stack((dt.sign()*torch.log1p(dt.abs())/4,1/(1+dt.abs()),(dt==0).float()),-1)
        return self.x[ids],self.i[ids].long(),self.i[ix].long(),tm,mask


class Net(nn.Module):
    def __init__(self,geometry,center,scale):
        super().__init__()
        self.register_buffer('geo',torch.from_numpy(geometry))
        self.register_buffer('center',center);self.register_buffer('scale',scale)
        self.film=nn.Embedding(len(geometry),12)
        nn.init.normal_(self.film.weight,std=.025)
        self.proj=nn.Sequential(nn.Linear(geometry.shape[1]+12,28),nn.LayerNorm(28),nn.SiLU())
        self.peer=nn.Sequential(nn.Linear(3*28+3,40),nn.SiLU(),nn.Linear(40,24))
        self.gate=nn.Linear(24,3)
        self.head=nn.Sequential(nn.Linear(len(center)+28+24*3,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.2),
                                nn.Linear(80,40),nn.SiLU(),nn.Linear(40,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def encode(self,i):
        return self.proj(torch.cat((self.geo[i],self.film(i)),dim=-1))
    def forward(self,x,i,peers,tm,mask):
        z=self.encode(i);v=self.encode(peers)
        if self.training:mask=mask & (torch.rand_like(tm[:,:,0])>.16)
        q=z[:,None,:].expand_as(v)
        value=self.peer(torch.cat((q,v,q*v,tm),-1))
        weight=self.gate(value).clamp(-5,5).exp()*mask[:,:,None]
        pooled=torch.bmm(weight.transpose(1,2),value)/(1+weight.sum(1)[:,:,None])
        val=self.head(torch.cat((((x-self.center)/self.scale).clamp(-5,5),z,pooled.flatten(1)),1))
        return .3*val[:,0],val[:,1:]


def fit(data,geometry,tr,err,truth,seed,path):
    torch.manual_seed(seed)
    sample=data.x[tr[::12]]
    center=sample.mean(0);scale=sample.std(0).clamp(min=.01)
    net=Net(geometry,center,scale)
    opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.04)
    states=[]
    for ep in range(9):
        net.train()
        if ep==6:opt.param_groups[0]['lr']=.00028
        for rows in tr[torch.randperm(len(tr))].split(1536):
            out,logits=net(*data.batch(rows))
            b=data.x[rows,0]
            star=torch.arange(1,6,dtype=b.dtype)
            prior=-.5*((b[:,None]-star)/.85)**2
            loss=(out-err[rows]).square().mean()+.12*F.cross_entropy(prior+logits,truth[rows])+.06*out.square().mean()
            opt.zero_grad();loss.backward();opt.step()
        if ep in (2,5,8):states.append({k:v.detach().clone() for k,v in net.state_dict().items() if k!='geo'})
    torch.save(states,path)
    return states


def predict(data,geometry,states,ix):
    center=states[0]['center'];net=Net(geometry,center,states[0]['scale'])
    result=np.zeros(len(ix));net.eval()
    with torch.inference_mode():
        for state in states:
            net.load_state_dict(state,strict=False)
            for pos in range(0,len(ix),4096):
                rows=ix[pos:pos+4096]
                a,_=net(*data.batch(rows));result[pos:pos+len(rows)]+=a.numpy()/len(states)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_query_sets.cpp','-o',str(OUT/'sets.dylib')],check=True)
    torch.set_num_threads(2)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    geo=geometry();dataset=Data('probex',base);print('prepared',geo.shape,flush=True)
    held=np.empty(len(y));err=torch.from_numpy((y-base).astype('f'));truth=torch.from_numpy(y.astype('int64')-1)
    for fold in range(5):
        tr,va=[torch.from_numpy(np.flatnonzero(mask)) for mask in (f!=fold,f==fold)]
        states=fit(dataset,geo,tr,err,truth,5301+fold,OUT/f'fold-{fold}.pt')
        held[va]=predict(dataset,geo,states,va)
        print('fold',fold,ne.rmse(base[va]+held[va],y[va]),flush=True)
    np.save(OUT/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del dataset
        dataset=Data('qual',qb);rows=torch.arange(len(qb));delta=np.zeros(len(qb))
        for fold in range(5):delta+=predict(dataset,geo,torch.load(OUT/f'fold-{fold}.pt',weights_only=True),rows)/5
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
