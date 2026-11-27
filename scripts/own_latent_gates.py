"""Condition residual expert weights on movie geometry, user tastes and rating uncertainty."""
import argparse
import ctypes
import json
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import own_anchor_context as anchor
import own_conditional_gates as gates
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_session_latent as latent
import own_retained_history as retained
import own_full_context as full


def side(train,query):
    lib=ctypes.CDLL(str((ne.ROOT/'own-other-raters-20261126/raters.dylib').resolve()))
    lib.profiles.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*6
    lib.profiles.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    # Both splits share the same label-free movie coordinate system.
    g=np.load(path)
    _,v=np.linalg.eigh(g.T@g)
    c=g@v[:,-24:]
    c=(c-c.mean(0))/np.maximum(c.std(0),1e-6)
    c=np.ascontiguousarray(np.clip(c,-8,8),dtype='f')
    mm=np.ascontiguousarray((np.bincount(i,weights=r,minlength=len(ic))+20*r.mean())/(ic+20),dtype='f')
    prof=np.zeros((len(uc),48),'f')
    lib.profiles(len(uc),*[a.ctypes.data for a in (off,i,r,mm,c,prof)])
    prof/=np.maximum(uc,1)[:,None]
    # Remove systematic axis-level shifts before interacting film and user coordinates.
    prof-=prof.mean(0)
    prof/=np.maximum(prof.std(0),.01)
    uf=np.clip(prof[qu],-6,6)
    mov=c[qi]
    return np.column_stack((mov,uf,mov*uf[:,:24],mov*uf[:,24:])).astype('f')


class Net(nn.Module):
    def __init__(self,nctx,nexp):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(nctx+nexp,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.18),
                                  nn.Linear(96,64),nn.SiLU(),nn.Dropout(.15))
        self.delta=nn.Linear(64,nexp+1)
        self.stars=nn.Linear(64,5)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self,x,d):
        z=self.shared(torch.cat((x,(8*d).clamp(-5,5)),1))
        w=self.delta(z).tanh()
        return w[:,0]*.10+(w[:,1:]*d).sum(1)*.40,self.stars(z)


def inputs(keys,p,q,known):
    cx,d=gates.inputs(keys,p,q,known)
    tr,phase=('trainx','probex') if known else ('fulltrain','qual')
    latent_features=side(tr,phase)
    # A frozen set of first-level neighborhood summaries supplies distribution and support.
    sources=[ne.ROOT/f'own-{name}-20261126'/(phase+'.npy') for name in ('user-gp','session-pairs')]
    chunks=[cx.numpy(),latent_features]
    for path in sources:
        raw=np.load(path,mmap_mode='r')
        chunks.append(raw[:,::max(1,raw.shape[1]//16)])
    return torch.from_numpy(np.column_stack(chunks).astype('f')),d


def fit_predict(x,xx,tr,va,truth,errors,seed,epochs):
    cx,d=x
    qx,qd=xx
    torch.manual_seed(seed)
    lo,hi=torch.quantile(cx[tr[::16]],torch.tensor([.02,.98]),dim=0)
    center=(lo+hi)*.5
    scale=(hi-lo).clamp(min=.005)*.5
    net=Net(cx.shape[1],d.shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.04)
    def norm(t):return ((t-center)/scale).clamp(-4,4)
    result=np.zeros(len(va))
    checkpoints={epochs//3-1,2*epochs//3-1,epochs-1}
    for ep in range(epochs):
        net.train()
        if ep==epochs*2//3:
            opt.param_groups[0]['lr']*=.5
        for ix in tr[torch.randperm(len(tr))].split(4096):
            pred,logits=net(norm(cx[ix]),d[ix])
            loss=(pred-errors[ix]).square().mean()+.025*pred.square().mean()+.08*F.cross_entropy(logits,truth[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in checkpoints:
            net.eval()
            with torch.inference_mode():
                for start in range(0,len(va),8192):
                    ix=va[start:start+8192]
                    pred,_=net(norm(qx[ix]),qd[ix])
                    result[start:start+len(ix)]+=pred.numpy()/len(checkpoints)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--parent',default='stage-05')
    parser.add_argument('--epochs',type=int,default=18)
    parser.add_argument('--seed',type=int,default=20261128)
    args=parser.parse_args()
    out=ne.ROOT/args.name
    out.mkdir(exist_ok=True)
    parent=blend.OUT/args.parent
    report=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==report['qualifying_hashes']['selected']
    y,f,p,q=anchor.history_control()
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    keys=report['columns']
    torch.set_num_threads(3)
    x=inputs(keys,p,q,True)
    e=torch.from_numpy((y-base).astype('f'))
    truth=torch.from_numpy((y-1).astype('int64'))
    held=np.empty(len(y))
    for j in range(5):
        tr,va=[torch.from_numpy(np.flatnonzero(z)) for z in (f!=j,f==j)]
        held[va.numpy()]=fit_predict(x,x,tr,va,truth,e,args.seed,args.epochs)
        print('fold',j,ne.rmse(base[va]+held[va],y[va]),flush=True)
    np.save(out/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    report=dict(selected=key,known=stats,config=vars(args),baseline_sha256=report['qualifying_hashes']['selected'],
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        xx=inputs(keys,p,q,False)
        delta=fit_predict(x,xx,torch.arange(len(y)),torch.arange(len(q)),truth,e,args.seed,args.epochs)
        np.save(out/'qual-correction.npy',delta)
        path=out/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
