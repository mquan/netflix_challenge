"""Fit an ordinal probability model after matching its prior expected rating to a frozen local ensemble."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as peer
import own_retained_history as ret
import own_schema_receipt as receipt
import own_suffix_impute as impute

OUT=ne.ROOT/'own-tilted-stars-20261126'
PARENT=blend.OUT/'paired-03'
S=np.arange(1,6,dtype='f')


class Role:
    def __init__(self,phase,base):
        self.base=base;train='trainx' if phase=='probex' else 'fulltrain'
        self.index=ret.map_rows() if phase=='probex' else np.arange(1408395,1408395+len(base))
        self.mode=1 if phase=='probex' else 2;self.u,self.i,self.d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc,self.ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')];self.sources={}
        def load(path):
            self.sources[str(path)]=receipt.streamed_sha(path);return np.load(path,mmap_mode='r')
        self.nade=load(cf.OUT/(phase+'.npy'));self.impute=load(impute.OUT/(phase+'.npy'));self.p=load(peer.OUT/'probabilities.npy')
        assert self.sources[str(peer.OUT/'probabilities.npy')]==json.loads((peer.OUT/'manifest.json').read_text())['probabilities_sha256']
        assert not json.loads((impute.OUT/'freeze.json').read_text())['qualifying_labels_used_for_fitting']
        assert (impute.OUT/(train+'.pt')).exists()
        self.sources[str(impute.OUT/(train+'.pt'))]=receipt.streamed_sha(impute.OUT/(train+'.pt'))
        raw=full.cf96(phase);np.testing.assert_allclose(self.impute[:,6],raw,rtol=0,atol=.0004)
        np.testing.assert_allclose(self.impute[:,:5]@S,self.impute[:,5],rtol=0,atol=.0004)
        np.testing.assert_allclose(self.nade[:,7:12]@S,raw,rtol=0,atol=.0004)
        self.rank=ne.verified_rank(phase)
    def block(self,sl):
        base=self.base[sl].astype('f');native=self.nade[sl,7:12];imp=self.impute[sl];peer=self.p[self.index[sl],self.mode].astype('f')
        np.testing.assert_allclose(peer.sum(1),1,rtol=0,atol=.001)
        prior=(.45*native+.35*imp[:,:5]+.2*peer).clip(.000001);prior/=prior.sum(1,keepdims=True)
        lo=np.full(len(base),-18,'f');hi=-lo
        for _ in range(19):
            tilt=(lo+hi)*.5;log=np.log(prior)+tilt[:,None]*(S-3)
            q=np.exp(log-log.max(1,keepdims=True));q/=q.sum(1,keepdims=True);mask=(q@S)<base
            lo=np.where(mask,tilt,lo);hi=np.where(mask,hi,tilt)
        assert np.max(np.abs(q@S-base))<.00035
        x=np.column_stack((base-3.5,prior,native,imp[:,:5],peer,q,imp[:,5:8]-base[:,None],self.rank[sl]-base,peer@S-base,
            self.d[sl]/2000,np.log1p(self.uc[self.u[sl]])/6,np.log1p(self.ic[self.i[sl]])/9))
        return x.astype('f'),np.log(q.clip(1e-10))


class Model(nn.Module):
    def __init__(self,n):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.16),nn.Linear(48,5))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x,prior):
        return prior+.48*self.layers(x).tanh()


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);y,f,_,_=anchor.history_control();fr=receipt.read(PARENT/'freeze.json')
    assert receipt.streamed_sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');state=Role('probex',b);sources={'probex':state.sources}
    values=[state.block(slice(j,j+32768)) for j in range(0,len(b),32768)]
    x,p=[torch.from_numpy(np.concatenate(v)).to('mps') for v in zip(*values)];del values
    yt=torch.from_numpy(y.astype('int64')-1).to('mps');stars=torch.tensor(S,device='mps');models=[];held=np.zeros(len(y),'f')
    for k in range(5):
        rng=np.random.default_rng(6141+k);torch.manual_seed(6141+k);torch.mps.manual_seed(6141+k);m=Model(x.shape[1]).to('mps')
        opt=torch.optim.AdamW(m.parameters(),lr=.0013,weight_decay=.04);tr=np.flatnonzero(f!=k);va=np.flatnonzero(f==k)
        for ep in range(12):
            order=rng.permutation(tr);m.train();opt.param_groups[0]['lr']=.00014+.00090*(1+np.cos(ep*np.pi/12))/2
            for j in range(0,len(tr),4096):
                ix=torch.from_numpy(order[j:j+4096]).to('mps');logits=m(x[ix],p[ix]);prob=logits.softmax(1);mu=prob@stars
                loss=(mu-yt[ix]-1).square().mean()+.32*F.cross_entropy(logits,yt[ix])+.10*(mu-p[ix].softmax(1)@stars).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        m.eval()
        with torch.inference_mode():
            for j in range(0,len(va),8192):
                rows=va[j:j+8192];ix=torch.from_numpy(rows).to('mps');held[rows]=(m(x[ix],p[ix]).softmax(1)@stars).cpu().numpy()-b[rows]
        models.append(m);print('ordinal fold',k,flush=True)
    stats={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];choice=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=stats,raw_model_sources=sources,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x,p,state
        state=Role('qual',qb);sources['qual']=state.sources;delta=np.zeros(len(qb),'f')
        with torch.inference_mode():
            for j in range(0,len(qb),16384):
                sl=slice(j,j+16384);xx,pp=[torch.from_numpy(a).to('mps') for a in state.block(sl)]
                delta[sl]=sum(m(xx,pp).softmax(1)@stars for m in models).cpu().numpy()/5-qb[sl]
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta);ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
