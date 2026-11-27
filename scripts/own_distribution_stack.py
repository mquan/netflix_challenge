"""Nonlinear ordinal stacking on all distinct fresh checkpoints and their pairwise disagreements."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_full_ridge as bank
import own_joint_experts as joint
import own_joint_jacobian as jac
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
import own_trajectory_bank as trajectory

OUT=ne.ROOT/'own-distribution-stack-20261126'
PC=72


def rotation(z,n):
    values,axis=np.linalg.eigh(z.T@z/len(z))
    return (axis[:,-n:]/np.sqrt(values[-n:].clip(.05))).astype('f')


class Tables:
    def __init__(self,phase,p,q,base,keys,cols,transform=None):
        known=phase=='probex';self.base=base
        self.raw=bank.Inputs(cols,phase,base,None if known else transform[:2])
        if known:
            z=self.raw.raw(slice(None,None,24));mean,sd=self.raw.transform
            z=((z-mean)/sd).clip(-6,6);proj=rotation(z,PC)
            sq=z*z;smean=sq.mean(0);sp=rotation(sq-smean,30)
            transform=(mean,sd,proj,smean,sp)
        self.transform=transform
        self.cf=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')
        self.ctx=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')[:,::5]
        support=json.loads((trajectory.OUT/'sources.json').read_text())['support']
        self.support=[]
        for meta in list(support.values())[::3]:
            file=Path(meta['paths'][int(not known)]);assert ne.sha(file)==meta['sha256'][int(not known)]
            self.support.append(np.load(file,mmap_mode='r'))
        self.parts=[np.load(ne.ROOT/('own-'+r+'-20261126')/(phase+'.npy'),mmap_mode='r')[:,::s]
                    for r,s in [('user-gp',2),('kernel-family',4),('session-pairs',5)]]
        self.ex=np.column_stack([blend.direction(k,p,q,known)[0] for k in keys]).astype('f')
        self.u=np.asarray(ne.history(phase,'user_idxs'));self.movie=np.asarray(ne.history(phase,'item_idxs'))
        self.date=ne.history(phase,'dates')
        self.train='trainx' if known else 'fulltrain'
        self.usercoords=[]
        mapping=retained.QUAL/'dataset/dense-to-raw-user.npy';sig=json.loads((mapping.parent/'manifest.json').read_text())
        assert ne.sha(mapping)==sig['outputs'][mapping.name]
        raw=np.load(mapping);rorder=np.argsort(raw)
        canonical=ne.history(self.train,'uidx_to_uid');ordered=rorder[np.searchsorted(raw[rorder],canonical)]
        np.testing.assert_array_equal(raw[ordered],canonical)
        for name in jac.MODELS:
            file=jac.OUT/(name+'.users.f32')
            v=np.fromfile(file,'<f4').reshape(len(raw),12)
            self.usercoords.append(v[ordered])
    def batch(self,sl):
        z=self.raw.raw(sl);mean,sd,proj,smean,sp=self.transform;z=((z-mean)/sd).clip(-8,8)
        linear=(z@proj).clip(-8,8)
        quadratic=((z*z-smean)@sp).clip(-12,12)
        prior=joint.tilted(self.cf[sl,7:12],self.base[sl])
        ex=self.ex[sl]
        x=np.column_stack((linear,quadratic,self.base[sl],np.std(z,axis=1),self.date[sl]/2000.,
            self.ctx[sl],self.cf[sl,7:37:2],ex,*[a[sl] for a in self.support],*[a[sl] for a in self.parts],
            *[a[self.u[sl]] for a in self.usercoords])).astype('f')
        return x,prior
    def known(self):
        x=np.empty((len(self.base),self.batch(slice(0,4))[0].shape[1]),'f');prior=np.empty((len(x),5),'f')
        for j in range(0,len(x),8192):
            sl=slice(j,j+8192);x[sl],prior[sl]=self.batch(sl)
        return torch.from_numpy(x),torch.from_numpy(prior)


class Net(nn.Module):
    def __init__(self,width,nm):
        super().__init__()
        self.emb=nn.Embedding(nm,12);nn.init.normal_(self.emb.weight,std=.12)
        self.pair=nn.Parameter(torch.randn(PC,24)*.04)
        self.trunk=nn.Sequential(nn.Linear(width+24+12,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.23),
                                 nn.Linear(96,48),nn.SiLU(),nn.Dropout(.15))
        self.head=nn.Linear(48,22)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('stars',torch.arange(1,6,dtype=torch.float32))
    def forward(self,x,prior,m):
        z=x[:,:PC]
        v=self.pair
        pair=(z@v).square()-z.square()@v.square()
        h=self.trunk(torch.cat((x,pair.clamp(-6,6),self.emb(m)),1));o=self.head(h).tanh()
        # A row-dependent combination of signed disagreement coordinates tilts the rating law.
        slope=.065*(o[:,:16]*z[:,:16]).sum(1)+.20*o[:,16]
        logits=prior+slope[:,None]*(self.stars-3)+.60*o[:,17:22]
        delta=(logits.softmax(1)-prior.softmax(1))@self.stars
        return delta,logits


def train(x,prior,m,y,base,f,k):
    idx=torch.from_numpy(np.flatnonzero(f!=k))
    torch.manual_seed(8191+k);lower,upper=torch.quantile(x[idx[::18]],torch.tensor([.02,.98]),dim=0)
    center=.5*(upper+lower);scale=.5*(upper-lower).clamp(min=.03)
    net=Net(x.shape[1],17770);opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.035)
    states=[]
    for ep in range(22):
        if ep==13:opt.param_groups[0]['lr']=.00035
        if ep==18:opt.param_groups[0]['lr']=.00016
        net.train()
        for rows in idx[torch.randperm(len(idx))].split(4096):
            delta,logits=net(((x[rows]-center)/scale).clamp(-5,5),prior[rows],m[rows])
            truth=y[rows]
            cumulative=logits.softmax(1).cumsum(1)[:,:4]
            targets=(truth[:,None]<=torch.arange(4)[None]).float()
            err=(truth.float()+1-base[rows])
            loss=(delta-err).square().mean()+.29*F.cross_entropy(logits,truth)+.23*(cumulative-targets).square().mean()+.12*delta.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.5);opt.step()
        if ep in (6,13,21):states.append({k:v.detach().clone() for k,v in net.state_dict().items()})
    torch.save(dict(center=center,scale=scale,states=states),OUT/f'fold-{k}.pt')
    return center,scale,states


def models(width,saved):
    center,scale,states=saved;out=[]
    for state in states:
        net=Net(width,17770);net.load_state_dict(state);net.eval();out.append(net)
    return center,scale,out


def infer(x,p,m,ensemble):
    with torch.inference_mode():
        c,s,nets=ensemble;z=((x-c)/s).clamp(-5,5)
        return sum(net(z,p,m)[0] for net in nets).numpy()/len(nets)


def main(verified_forecasts=False,single_execution=False):
    global OUT
    if verified_forecasts:OUT=ne.ROOT/('own-single-execution-distribution-20261126' if single_execution else 'own-verified-distribution-20261126')
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    y,f,p,q=anchor.history_control()
    parent=blend.OUT/('stage-24' if verified_forecasts else json.loads((blend.OUT/'queue.json').read_text())['parent'])
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    cols=bank.sources()
    if verified_forecasts:
        import own_aligned_forecasts as aligned
        file=aligned.OUT/'schema.json'
        schema=json.loads(file.read_text());vetted=schema['forecasts']
        assert all(cols[k]==v for k,v in vetted.items()) and (set(cols)-set(vetted))==set(schema['excluded'])
        assert len(vetted)==426
        cols=vetted
        if single_execution:
            cols={k:v for k,v in vetted.items() if 'producer' in v}
            for arm in range(2):assert len({v['sha256'][arm] for v in cols.values()})==len(cols)
            assert len(cols)==202
            (OUT/'inputs.json').write_text(json.dumps(cols,indent=2)+'\n')
    known=Tables('probex',p,q,base,fr['columns'],cols);transform=known.transform
    x,prior=known.known();m=torch.as_tensor(known.movie.astype('int64'));truth=torch.from_numpy(y.astype('int64')-1)
    baseline=torch.from_numpy(base.astype('f'));del known
    ensembles=[];held=np.empty(len(y),'f')
    for k in range(5):
        e=models(x.shape[1],train(x,prior,m,truth,baseline,f,k));ensembles.append(e)
        rows=np.flatnonzero(f==k)
        for j in range(0,len(rows),8192):
            ix=rows[j:j+8192];held[ix]=infer(x[ix],prior[ix],m[ix],e)
        print('fold',k,ne.rmse(base[rows]+held[rows],y[rows]),flush=True)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    rep=dict(parent=str(parent),selected=key,known=stats,distinct_fresh_pairs=len(cols),qualifying_hashes={},
             qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    if verified_forecasts:rep['forecast_schema_sha256']=ne.sha(file)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del x,prior
        qt=Tables('qual',p,q,qb,fr['columns'],cols,transform);delta=np.empty(len(qb),'f')
        for j in range(0,len(qb),8192):
            sl=slice(j,j+8192);xx,pp=map(torch.from_numpy,qt.batch(sl));mm=torch.from_numpy(qt.movie[sl].astype('int64'))
            delta[sl]=sum(infer(xx,pp,mm,e) for e in ensembles)/5
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--verified-forecasts',action='store_true')
    ap.add_argument('--single-execution',action='store_true');args=ap.parse_args()
    main(args.verified_forecasts or args.single_execution,args.single_execution)
