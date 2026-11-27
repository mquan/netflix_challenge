"""Train an item-centered attention model on other raters' target-removed CF errors and dated preferences."""
import ctypes
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_cf_context as cf
import own_cf_slate as slate
import own_full_context as full
import own_joint_jacobian as jac
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_transition as transition
from own_empirical_neighborhood import fit
from own_query_sets import projection
OUT=ne.ROOT/'own-item-attention-20261126'
PARENT=blend.OUT/'schema-04'
DEVICE='mps'


def tensor(a):
    return torch.as_tensor(np.ascontiguousarray(a),device=DEVICE)


class Data:
    def __init__(self,train,phase):
        err,_=transition.loo(train)
        u,m,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
        self.uc,self.ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
        self.qu,self.qm,self.qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.prior=full.cf96(phase)
        s=cf.checkpoint(train);enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
        off=np.r_[0,self.uc.cumsum()].astype('i');total=np.empty((len(self.uc),96),'f');md=np.empty(len(self.uc),'f')
        lib=ctypes.CDLL(str((slate.OUT/'slate.dylib').resolve()));lib.cf_totals.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*7;lib.cf_totals.restype=None
        lib.cf_totals(len(self.uc),len(self.ic),*[a.ctypes.data for a in (off,m,d,r,enc,total,md)])
        flat=enc.reshape(-1,96);_,vec=np.linalg.eigh(flat.T@flat);proj=vec[:,-24:]
        self.enc=np.ascontiguousarray((enc@proj).transpose(1,0,2));self.total=np.ascontiguousarray(total@proj)
        self.norm=(self.total/np.sqrt(self.uc.clip(1))[:,None]).std(0).astype('f').clip(.02);del total
        self.geo=np.column_stack((projection(s['p'].mean(0),16),projection(s['a'].mean(0),16)))
        path=ret.QUAL/'dataset/dense-to-raw-user.npy';sign=json.loads((path.parent/'manifest.json').read_text())
        assert ne.sha(path)==sign['outputs'][path.name]
        raw=np.load(path);order=np.argsort(raw);ids=order[np.searchsorted(raw[order],ne.history(train,'uidx_to_uid'))]
        self.user=np.column_stack([np.fromfile(jac.OUT/(name+'.users.f32'),'<f4').reshape(len(raw),12)[ids] for name in jac.MODELS])
        self.user=((self.user-self.user.mean(0))/self.user.std(0).clip(.0001)).clip(-6,6).astype('f')
        # Sort by movie and day once so the support includes both local dates and a representative historical sample.
        ix=np.lexsort((d,m));self.u,self.d,self.r,self.e=[a[ix] for a in (u,d,r,err)];del ix,err
        self.off=np.r_[0,self.ic.cumsum()].astype('i8')
        self.ready=np.flatnonzero(self.ic>0);self.p=self.ic[self.ready].astype(float);self.p/=self.p.sum()
        order=np.lexsort((self.qd,self.qm));counts=np.bincount(self.qm,minlength=len(self.ic));qoff=np.r_[0,counts.cumsum()]
        groups=[]
        for movie in np.flatnonzero(counts):
            for j in range(qoff[movie],qoff[movie+1],24):
                rows=order[j:min(qoff[movie+1],j+24)];g=np.full(24,-1,'i');g[:len(rows)]=rows;groups.append(g)
        self.query_groups=np.stack(groups)
        print(train,'prepared',len(self.ready),'movies',flush=True)
    def batch(self,rng,rows=None):
        if rows is None:
            movie=rng.choice(self.ready,72,p=self.p);off=self.off[movie];cnt=self.ic[movie]
            # Draw a narrow or broad chronology of targets per movie, as at grouped inference.
            middle=rng.random(len(movie));span=rng.choice([.10,.35,1.],len(movie),p=[.40,.35,.25])
            pos=(middle[:,None]+(rng.random((len(movie),12))-.5)*span[:,None]).clip(0,.9999999)
            target=off[:,None]+(pos*cnt[:,None]).astype('i')
            tu,td,ts,te=[v[target] for v in (self.u,self.d,self.r,self.e)]
            prior=ts-te;truth=ts.copy();trained=ts.astype('i')
            loc=(pos.mean(1)*cnt).astype('i')
        else:
            valid=rows>=0;ids=rows.clip(min=0);tu,td,prior=self.qu[ids],self.qd[ids],self.prior[ids]
            movie=self.qm[ids[:,0]];off=self.off[movie];cnt=self.ic[movie]
            # Search each movie's chronological support for the center of this query batch.
            avg=(td*valid).sum(1)/valid.sum(1)
            loc=np.array([np.searchsorted(self.d[a:a+n],t) for a,n,t in zip(off,cnt,avg)])
            truth=None;trained=np.zeros_like(tu)
        b=len(movie);n=96;uniform=rng.random((b,n//2))*cnt[:,None]
        radius=np.maximum(30,(cnt*.18).astype('i'));near=loc[:,None]+(rng.random((b,n//2))-.5)*2*radius[:,None]
        pos=np.concatenate((uniform,near),1).astype('i').clip(0,np.maximum(0,cnt-1)[:,None])
        support=np.minimum(len(self.u)-1,off[:,None]+pos);su,sd,sr,se=[v[support] for v in (self.u,self.d,self.r,self.e)]
        # Per-sample weights ensure rare movies contribute each observed rating at most once.
        first=np.argsort(pos,axis=1,kind='stable');sorted_pos=np.take_along_axis(pos,first,axis=1)
        keep=np.ones(pos.shape,bool);keep[:,1:]=np.diff(sorted_pos,axis=1)!=0
        keep=np.take_along_axis(keep,np.argsort(first,axis=1),axis=1)&(cnt[:,None]>0)
        inp=tuple(tensor(v) for v in (movie.astype('i'),tu.astype('i'),td.astype('f'),trained.astype('i'),prior.astype('f'),
                                      su.astype('i'),sd.astype('f'),sr.astype('i'),se.astype('f'),keep))
        return inp,None if truth is None else tensor(truth.astype('i')).long()


class Net(nn.Module):
    def __init__(self,d):
        super().__init__()
        for key,value in [('enc',d.enc),('total',d.total),('norm',d.norm),('original',d.user),('count',d.uc.astype('f')),('geo',d.geo),
                          ('pop',np.log1p(d.ic).astype('f')/9)]:
            self.register_buffer(key,torch.from_numpy(np.ascontiguousarray(value)),persistent=False)
        self.user_embed=nn.Embedding(len(d.uc),12);nn.init.normal_(self.user_embed.weight,std=.025)
        self.film=nn.Embedding(len(d.ic),12);nn.init.normal_(self.film.weight,std=.08)
        self.person=nn.Sequential(nn.Linear(61,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,48))
        self.values=nn.Sequential(nn.Linear(48+9,56),nn.SiLU(),nn.Linear(56,40))
        self.keys=nn.Linear(48,48);self.queries=nn.Linear(48,48)
        self.time=nn.Parameter(torch.tensor([0.,.6,1.5,2.8]))
        self.decoder=nn.Sequential(nn.Linear(48+40+32+12+8+4,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.14),
                                   nn.Linear(96,48),nn.SiLU(),nn.Linear(48,7))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def features(self,m,u,r):
        included=r>0
        enc=self.total[u]-self.enc[m[:,None],(r-1).clamp(min=0)]*included[:,:,None]
        cn=(self.count[u]-included.float()).clamp(min=1)
        x=(enc/(cn.sqrt()[:,:,None]*self.norm)).clamp(-7,7)
        return self.person(torch.cat((x,self.original[u],self.user_embed(u),cn.log1p()[:,:,None]/6),2))
    def forward(self,m,u,d,r,prior,su,sd,sr,se,valid):
        z=self.features(m,u,r);s=self.features(m,su,sr);b=len(m)
        show=(sr.float()-3.5)/1.4
        side=torch.cat((se[:,:,None],show[:,:,None],(sr.float()-se-3.5)[:,:,None],(sd/2000)[:,:,None],
                         F.one_hot(sr.long()-1,5).float()),2)
        def heads(v):return v.reshape(b,-1,4,v.shape[-1]//4).transpose(1,2)
        logits=heads(self.queries(z))@heads(self.keys(s)).transpose(-1,-2)/math.sqrt(12)
        dist=(d[:,:,None]-sd[:,None,:]).abs()
        logits-=self.time.clamp(0,5)[None,:,None,None]*torch.log1p(dist[:,None]/45)
        ok=valid[:,None,:]&(u[:,:,None]!=su[:,None,:])
        weights=logits.masked_fill(~ok[:,None],-10000.).softmax(-1)*ok[:,None]
        attended=(weights@heads(self.values(torch.cat((s,side),2)))).transpose(1,2).reshape(b,-1,40)
        innovation=(weights*se[:,None,None,:]).sum(3).transpose(1,2)
        rating=(weights*show[:,None,None,:]).sum(3).transpose(1,2)
        meta=torch.stack((prior-3.5,d/2000,self.pop[m,None].expand_as(d),ok.sum(2).float().log1p()/4.5),2)
        fixed=torch.cat((self.geo[m],self.film(m)),1)[:,None].expand(-1,u.shape[1],-1)
        a=self.decoder(torch.cat((z,attended,fixed,innovation,rating,meta),2))
        delta=.50*a[:,:,0].tanh()+.42*(a[:,:,1:5].tanh()*innovation).sum(2)
        return delta,a[:,:,5:]


def run(train,phase):
    d=Data(train,phase);torch.manual_seed(81521);torch.mps.manual_seed(81521);torch.set_num_threads(1)
    net=Net(d).to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.026)
    epochs=16;selected=[];start=time.monotonic()
    for ep in range(epochs):
        net.train();rng=np.random.default_rng(91841+ep)
        opt.param_groups[0]['lr']=.00013+.00067*(1+math.cos(ep*math.pi/epochs))/2
        for _ in range(480):
            inp,y=d.batch(rng);delta,aux=net(*inp);err=y.float()-inp[4]
            loss=(delta-err).square().mean()+.14*F.binary_cross_entropy_with_logits(aux,torch.stack((y<=2,y>=5),2).float())+.055*delta.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.4);opt.step()
        if ep in (4,9,15):selected.append({k:v.cpu().clone() for k,v in net.state_dict().items()})
        print(train,'epoch',ep+1,round(time.monotonic()-start),flush=True)
    torch.save(selected,OUT/(train+'.pt'));net.eval();result=np.zeros((len(d.qu),3),'f')
    with torch.inference_mode():
        for k,state in enumerate(selected):
            net.load_state_dict(state)
            for pass_no in range(2):
                rng=np.random.default_rng(937+pass_no)
                for j in range(0,len(d.query_groups),64):
                    rows=d.query_groups[j:j+64];inp,_=d.batch(rng,rows);delta,_=net(*inp);mask=rows>=0
                    result[rows[mask],k]+=.5*delta.cpu().numpy()[mask]
    assert np.isfinite(result).all();np.save(OUT/(phase+'.npy'),result)
    return result


def design(v,b,phase):
    native=full.cf96(phase)
    return np.column_stack((v,native[:,None]+v-b[:,None])).astype(float)


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    v=run('trainx','probex');x=design(v,base,'probex');held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        v=run('fulltrain','qual');delta=design(v,qb,'qual')@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
