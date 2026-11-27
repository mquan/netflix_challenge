"""Integrate the temporal model's attention over the complete observed rating history."""
import argparse
import json
import math
import time
import numpy as np
import torch
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_temporal_attention as temporal
from own_empirical_neighborhood import fit

MODES=5


def tensors(data,us):
    rows=[data.order[data.qoff[u]:data.qoff[u+1]] for u in us]
    n=max(map(len,rows));width=int(np.max(data.counts[us]+np.diff(data.ioff)[us]))
    hi,hr,hd=[np.zeros((len(us),width),'i') for _ in range(3)]
    ti,td=[np.zeros((len(us),n),'i') for _ in range(2)]
    med=np.zeros(len(us),'i')
    for j,(u,target) in enumerate(zip(us,rows)):
        lo,hirow=data.off[u:u+2];a,b=data.ioff[u:u+2];k=hirow-lo;h=k+b-a
        hi[j,:k]=data.i[lo:hirow];hr[j,:k]=data.r[lo:hirow];hd[j,:k]=data.d[lo:hirow]
        hi[j,k:h]=data.ii[a:b];hr[j,k:h]=6;hd[j,k:h]=data.iday[a:b]
        ti[j,:len(target)]=data.qi[target];td[j,:len(target)]=data.qd[target];med[j]=int(np.median(data.qd[target]))
    return [torch.as_tensor(a,device='mps') for a in (hi,hr,hd,ti,td,data.counts[us].astype('f'),us.astype('i'),med)],rows


def integrate(net,hi,hr,hd,ti,td,counts,u,median):
    b,h=hi.shape;n=ti.shape[1];nh=temporal.HEADS;dim=temporal.WIDTH
    he,qe=net.film(hi),net.film(ti)
    valid=hr!=0;visible=valid&(hr<6);unknown=hr==6
    centered=(hr.float()-3.5)*visible/1.4
    extra=torch.stack((net.pop[hi],net.year[hi],hd/2000.,unknown.float()),-1)
    key,value=net.encoder(torch.cat((he,he*centered[:,:,None],net.star(hr),extra),2)).chunk(2,-1)
    def heads(x):return x.reshape(b,-1,nh,dim//nh).transpose(1,2)
    raw=heads(net.query(qe))@heads(key).transpose(-1,-2)/math.sqrt(dim//nh)
    distance=(td[:,:,None]-hd[:,None,:]).abs().float()
    raw=raw-net.time_gate.clamp(0,5)[None,:,None,None]*torch.log1p(distance[:,None]/5.)
    raw=raw+net.hidden_gate[None,:,None,None]*unknown[:,None,None,:]
    usable=valid[:,None,:]&(ti[:,:,None]!=hi[:,None,:])
    count=counts[:,None,None]
    cap=(temporal.H-unknown.sum(1))[:,None,None].float()
    def inclusion(dist,days,share):
        near=(dist<=days)&visible[:,None,:]
        inside=near.sum(-1,keepdim=True);take=torch.minimum(inside,torch.floor(cap*share))
        p0=((cap-take)/(count-take).clamp(min=1)).clamp(0,1)
        pn=take/inside.clamp(min=1);selected=p0+(1-p0)*pn*near
        return torch.where(unknown[:,None,:],1.,selected).expand(b,n,h)
    uniform=torch.ones((b,n,h),device=hi.device)
    fractions=[uniform,inclusion((median[:,None]-hd).abs()[:,None,:],14,.5),
        inclusion(distance,14,.5),inclusion(distance,45,.7),inclusion(distance,3,.75)]
    onehot=torch.stack([(hr==k).float() for k in range(1,6)],-1)
    day=usable&(distance==0);star=torch.arange(1,6,dtype=qe.dtype,device=qe.device)
    result=[]
    for frac in fractions:
        w=(raw+frac.clamp(min=1e-12).log()[:,None]).masked_fill(~usable[:,None],-10000).softmax(-1)*usable[:,None]
        attended=(w@heads(value)).transpose(1,2).reshape(b,n,dim)
        obs=frac*visible[:,None,:];obs=obs/obs.sum(-1,keepdim=True).clamp(min=1)
        static=torch.cat((obs@value,net.user(u)[:,None,:].expand(-1,n,-1),obs@onehot,
            torch.log1p(counts)[:,None,None].expand(-1,n,1)/7),2)
        same=day*frac;known=same*visible[:,None,:]
        dyn=torch.stack((td/2000.,net.pop[ti],net.year[ti],torch.log1p(same.sum(-1))/4.,
            torch.log1p(known.sum(-1))/4.,(same*centered[:,None,:]).sum(-1)/known.sum(-1).clamp(min=1)),2)
        logits=net.bias(ti)+net.head(torch.cat((qe,attended,static,dyn),2))
        result.append(logits.softmax(-1)@star)
    return torch.stack(result,-1)


def run(out,src,train,phase):
    ck=src/(train+'.pt');state=torch.load(ck,map_location='cpu',weights_only=False);cfg=state['config']
    assert state['epoch']==cfg['epochs'] and cfg['name']==src.name
    data=temporal.Data(train,phase,cfg.get('matched_days',False),cfg.get('cf_factors',False))
    net=temporal.Net(data).to('mps');net.load_state_dict(state['net']);del state;net.eval()
    size=data.counts+np.diff(data.ioff);users=np.flatnonzero(np.diff(data.qoff));users=users[np.argsort(size[users])]
    result=np.empty((len(data.qi),MODES),'f');front=0;start=time.monotonic();validated=False
    with torch.inference_mode():
        while front<len(users):
            end=min(front+192,len(users))
            while (end-front)*size[users[end-1]]>23500 and end-front>1:end=front+(end-front+1)//2
            args,rows=tensors(data,users[front:end]);ans=integrate(net,*args)
            if not validated and size[users[end-1]]>=125:
                check=net(*args[:-1]).softmax(-1)@torch.arange(1,6,device='mps',dtype=torch.float32)
                diff=float((check-ans[:,:,0]).abs().max().cpu());assert diff<.0001,diff
                print(train,'full-history forward checked',diff,flush=True);validated=True
            ans=ans.cpu().numpy()
            for j,ix in enumerate(rows):result[ix]=ans[j,:len(ix)]
            if front//64000!=end//64000:print(train,end,round(time.monotonic()-start),flush=True)
            front=end
    assert validated and np.isfinite(result).all()
    np.save(out/(phase+'.npy'),result)
    return result


def directions(raw,base,original):
    return np.column_stack((raw-base[:,None],raw-original[:,None]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent-model',default='own-temporal-attention-20261126')
    ap.add_argument('--name',default='own-exact-attention-20261126');ap.add_argument('--blend-parent');args=ap.parse_args()
    src=ne.ROOT/args.parent_model;out=ne.ROOT/args.name;out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    parent=blend.OUT/(args.blend_parent or json.loads((blend.OUT/'queue.json').read_text())['parent'])
    y,f,_,_=anchor.history_control();base=np.load(parent/'known-selected.npy')
    src_rec=json.loads((src/'freeze.json').read_text())
    assert ne.sha(src/'selected.f32')==src_rec['qualifying_hashes']['selected']
    v=run(out,src,'trainx','probex');x=directions(v,base,np.load(src/'probex.npy'))
    held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),trained_model=str(src),training_checkpoint_sha256=ne.sha(src/'trainx.pt'),
          known=known,control=ctrl,selected='candidate' if ok else 'control',weights=w.tolist(),qualifying_hashes={},
          qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        assert ne.sha(parent/'selected.f32')==json.loads((parent/'freeze.json').read_text())['qualifying_hashes']['selected']
        qb=np.fromfile(parent/'selected.f32','<f4').astype(float)
        delta=directions(run(out,src,'fulltrain','qual'),qb,np.load(src/'qual.npy'))@w
        np.save(out/'oof.npy',held);np.save(out/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(out/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(out/'selected.f32')
    (out/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
