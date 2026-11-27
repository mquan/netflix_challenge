"""Learn a shared nonlinear series-rating update on genuine historical target/context pairs, with both ratings removed from the context prediction."""
import ctypes
import json
import subprocess
import numpy as np
import torch
from torch import nn
import own_anchor_context as anchor
import own_cf_context as cf
import own_cf_slate as slate
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as trans
from own_query_sets import projection
from own_empirical_neighborhood import fit
OUT=ne.ROOT/'own-series-transfer-20261126'
PARENT=blend.OUT/'paired-03'


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Sequential(nn.Linear(12,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.att=nn.Linear(32,3)
        self.head=nn.Sequential(nn.Linear(3*32+32+1,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,1))
    def forward(self,x):
        valid=x[:,:,0]>0;s=self.token(x);w=self.att(s).transpose(1,2).masked_fill(~valid[:,None,:],-10000).softmax(2)*valid[:,None]
        n=valid.sum(1).float();avg=(s*valid[:,:,None]).sum(1)/n.clamp(min=1)[:,None]
        return self.head(torch.cat(((w@s).flatten(1),avg,n.log1p()[:,None]),1)).squeeze(1)*(n>0)


def run(train,phase,lib):
    e,_=trans.loo(train);d=slate.Data(train,phase)
    root=ne.ROOT/'own-paired-title-neighbors-20261126';path=root/'catalog.npz';assert ne.sha(path)==json.loads((root/'catalog.json').read_text())['derived_sha256']
    cat=np.load(path);src=ne.ROOT/'own-learned-series-20261126'/(phase+'-neighbors.npz');learn=np.load(src)
    s,w,g=cat['neighbors'],cat['affinity'],cat['bands'];est,ev=learn['neighbors'],learn['weight']
    nb=np.full((len(s),20),-1,'i');strength=np.zeros(nb.shape,'f')
    for m in range(len(s)):
        a=np.flatnonzero(s[m]>=0);a=a[np.argsort(g[m,a]-w[m,a]*.5,kind='stable')[:12]]
        nb[m,:len(a)]=s[m,a];strength[m,:len(a)]=np.sqrt(w[m,a])
        k=np.flatnonzero(est[m]>=0)[:8];nb[m,12:12+len(k)]=est[m,k];strength[m,12:12+len(k)]=ev[m,k]
    geo=projection(cf.checkpoint(train)['p'].mean(0),16);geo=np.ascontiguousarray(geo/np.linalg.norm(geo,axis=1,keepdims=True).clip(.001),'f')
    year=np.ascontiguousarray(ne.history(train,'item_years'),'i2');off=np.r_[0,d.uc.cumsum()].astype('i')
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')];mu=np.ascontiguousarray(d.native,'f')
    shared=(qi,qd,mu,nb,strength,year,geo,d.md,d.total,d.ep,d.op,d.eb,d.b,d.ub,d.drift)
    def gather(indices,users,mode):
        result=np.zeros((len(indices),6,12),'f');aa=(off,d.i,d.d,d.r,e,users,indices,*shared,result)
        lib.series_rows(len(d.uc),len(indices),mode,d.sd,*[v.ctypes.data for v in aa]);return result
    rng=np.random.default_rng(31987);idx=np.sort(rng.choice(len(e),6000000,replace=False)).astype('i');owner=ne.history(train,'user_idxs')
    xx=[];yy=[]
    for j in range(0,len(idx),80000):
        rows=idx[j:j+80000];v=gather(rows,np.ascontiguousarray(owner[rows],'i'),1);mask=v[:,0,0]>0
        xx.append(v[mask]);yy.append(e[rows[mask]])
    x=torch.from_numpy(np.concatenate(xx)).to('mps');y=torch.from_numpy(np.concatenate(yy)).to('mps');del xx,yy
    print(train,'supported real training rows',len(x),flush=True)
    torch.manual_seed(8017);torch.mps.manual_seed(8017);net=Net().to('mps');optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.035)
    states=[]
    for ep in range(9):
        optimizer.param_groups[0]['lr']=.00014+.00070*(1+np.cos(ep*np.pi/9))/2;net.train()
        for ix in rng.permutation(len(x))[:550000].reshape(-1)[:(min(550000,len(x))//1536)*1536].reshape(-1,1536):
            k=torch.from_numpy(ix).to('mps');pred=net(x[k]);loss=(y[k]-pred).square().mean()+.055*pred.square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if ep in (2,5,8):states.append({k:a.detach().clone() for k,a in net.state_dict().items()})
    del x,y
    nets=[]
    for state in states:
        model=Net().to('mps');model.load_state_dict(state);nets.append(model.eval())
    qrows=np.argsort(qu,kind='stable').astype('i');result=np.empty((len(qrows),3),'f')
    with torch.inference_mode():
        for j in range(0,len(qrows),16384):
            rows=qrows[j:j+16384];v=torch.from_numpy(gather(rows,np.ascontiguousarray(qu[rows],'i'),0)).to('mps')
            result[rows]=torch.stack([m(v) for m in nets],1).cpu().numpy()
    assert np.isfinite(result).all();np.save(OUT/(phase+'.npy'),result)
    return result,dict(catalog=ne.sha(path),inferred_graph=ne.sha(src))


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations','scripts/own_series_transfer.cpp','-framework','Accelerate','-o',str(OUT/'series.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'series.dylib').resolve()));lib.series_rows.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*23;lib.series_rows.restype=None
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');v,src=run('trainx','probex',lib);held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=v[~tr]@fit(v[tr],(y-b)[tr])
    w=fit(v,y-b);ctrl=blend.metrics(b,y,f);known=blend.metrics(ne.clipped(b+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),inputs={'probex':src},
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv,src=run('fulltrain','qual',lib);rep['inputs']['qual']=src;delta=qv@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
