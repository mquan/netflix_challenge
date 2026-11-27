"""Bag conditional expert weights and ordinal distributions centered on the selected prediction."""
import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_full_context as full
import own_cf_context as cf
import own_trajectory_bank as trajectory
import own_retained_history as retained


def tilted(probs,base):
    star=np.arange(1,6,dtype='f')
    logits=np.log(np.maximum(probs,1e-12));lo=np.full(len(base),-12.,'f');hi=-lo
    for _ in range(16):
        mid=(lo+hi)*.5
        val=logits+mid[:,None]*star
        weights=np.exp(val-val.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
        below=weights@star<base
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    adjusted=logits+((lo+hi)*.5)[:,None]*star
    return adjusted-adjusted.max(1,keepdims=True)


def data(phase,keys,p,q,base,blocks):
    known=phase=='probex'
    expert=np.column_stack([blend.direction(k,p,q,known)[0].astype('f') for k in keys])
    ctx=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
    path=cf.OUT/(phase+'.npy')
    if not path.exists():
        assert not known
        cf.features('fulltrain',phase,base)
    raw=np.load(path,mmap_mode='r')
    assert raw.shape==(len(base),42)
    supp=json.loads((trajectory.OUT/'sources.json').read_text())['support']
    # Take every third coordinate from the verified 90-column factor ordering.
    support=list(supp.values())[::3]
    aux=[np.load(v['paths'][int(not known)],mmap_mode='r') for v in support]
    tables=[np.load(path/(phase+'.npy'),mmap_mode='r') for path in blocks]
    nctx=len(range(0,ctx.shape[1],4))+5+30+5+len(aux)+sum(v.shape[1] for v in tables)
    x=np.empty((len(base),nctx),'f');prior=np.empty((len(base),5),'f')
    for start in range(0,len(base),16384):
        sl=slice(start,start+16384)
        # The 30 probabilities comprise all six unmodified and local CF forecasts.
        pi=raw[sl,7:37]
        np.testing.assert_allclose(pi[:,:5].sum(1),1,atol=1e-5)
        prior[sl]=tilted(pi[:,:5],base[sl])
        x[sl]=np.column_stack((ctx[sl,::4],raw[sl,2:7]-raw[sl,1:2],pi,raw[sl,37:42],
                              *[a[sl] for a in aux],*[a[sl] for a in tables]))
    return tuple(torch.from_numpy(v) for v in (x,expert,prior))


class Net(nn.Module):
    def __init__(self,n,d):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(n+d,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.20),
                                 nn.Linear(80,48),nn.SiLU(),nn.Dropout(.15))
        self.head=nn.Linear(48,d+6)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,d,prior):
        z=self.head(self.trunk(torch.cat((x,(8*d).clamp(-6,6)),1))).tanh()
        logits=prior+.55*z[:,:5]
        stars=torch.arange(1,6,dtype=d.dtype,device=d.device)
        ord_delta=(logits.softmax(1)-prior.softmax(1))@stars
        delta=.55*ord_delta+.06*z[:,5]+.40*(z[:,6:]*d).sum(1)
        return delta,logits,ord_delta


def fit(x,tr,truth,err,seed,epochs,file):
    torch.manual_seed(seed)
    cx,d,prior=x
    low,high=torch.quantile(cx[tr[::16]],torch.tensor([.02,.98]),dim=0)
    center=(low+high)*.5;scale=(high-low).clamp(min=.01)*.5
    net=Net(cx.shape[1],d.shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.05)
    checkpoints=[]
    for ep in range(epochs):
        if ep==epochs*2//3:opt.param_groups[0]['lr']*=.5
        net.train()
        for ix in tr[torch.randperm(len(tr))].split(4096):
            xx=((cx[ix]-center)/scale).clamp(-4,4)
            pred,logits,od=net(xx,d[ix],prior[ix])
            loss=(pred-err[ix]).square().mean()+.15*F.cross_entropy(logits,truth[ix])+.20*(od-err[ix]).square().mean()+.05*pred.square().mean()
            opt.zero_grad();loss.backward();opt.step()
        if ep in (epochs//3-1,2*epochs//3-1,epochs-1):
            checkpoints.append({k:v.detach().clone() for k,v in net.state_dict().items()})
    obj=dict(center=center,scale=scale,checkpoints=checkpoints)
    torch.save(obj,file)
    return obj


def infer(x,rows,model):
    cx,d,prior=x
    net=Net(cx.shape[1],d.shape[1]);net.eval()
    pred=np.zeros(len(rows))
    with torch.inference_mode():
        for state in model['checkpoints']:
            net.load_state_dict(state)
            for j in range(0,len(rows),8192):
                ix=rows[j:j+8192]
                v,_,_=net(((cx[ix]-model['center'])/model['scale']).clamp(-4,4),d[ix],prior[ix])
                pred[j:j+len(ix)]+=v.numpy()/len(model['checkpoints'])
    return pred


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='own-joint-experts-20261126')
    ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--seed',type=int,default=20261130)
    args=ap.parse_args();out=ne.ROOT/args.name;out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    y,f,p,q=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    blocks=[v for v in (ne.ROOT/'own-cf-user-fit-20261126',ne.ROOT/'own-user-gp-20261126')
            if (v/'probex.npy').is_file() and (v/'qual.npy').is_file()]
    x=data('probex',rec['columns'],p,q,base,blocks)
    held=np.empty(len(y))
    truth=torch.from_numpy(y.astype('int64')-1);err=torch.from_numpy((y-base).astype('f'))
    for j in range(5):
        tr,va=[torch.from_numpy(np.flatnonzero(v)) for v in (f!=j,f==j)]
        model=fit(x,tr,truth,err,args.seed+j,args.epochs,out/('fold'+str(j)+'.pt'))
        held[va.numpy()]=infer(x,va,model)
        print('fold',j,ne.rmse(base[va]+held[va],y[va]),flush=True)
    np.save(out/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    eligible=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(eligible,key=lambda k:stats[k]['pooled']) if eligible else 'control'
    report=dict(selected=key,known=stats,parent=str(parent),keys=rec['columns'],config=vars(args),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:
        del x
        xx=data('qual',rec['columns'],p,q,qb,blocks)
        delta=np.zeros(len(qb));rows=torch.arange(len(qb))
        for j in range(5):delta+=infer(xx,rows,torch.load(out/('fold'+str(j)+'.pt'),weights_only=True))/5
        np.save(out/'qual-correction.npy',delta)
        path=out/'selected.f32';ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
