"""Test a bounded, one-layer ensemble of independently trained raw rating predictions under fresh user folds."""
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
import own_anchor_context as anchor
import own_cf_context as cf
import own_day_states as ds
import own_full_context as full
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
from own_standalone_ensemble import mixed_folds
OUT=ne.ROOT/'own-direct-canonical-20261126'
PARENT=blend.OUT/'ablation-02'
MARK=ne.ROOT/'own-single-execution-distribution-20261126/inputs.json'


class Role:
    def __init__(self,phase):
        idx=int(phase=='qual');train='fulltrain' if idx else 'trainx';manifest=json.loads(MARK.read_text())
        assert len(manifest)==202 and all(len({v['sha256'][k] for v in manifest.values()})==202 for k in (0,1))
        families={}
        for name,v in manifest.items():
            assert 'producer' in v;path=Path(v['paths'][idx]);assert ds.sha(path)==v['sha256'][idx]
            ep=re.search(r'_ep([0-9]+)',name);families.setdefault(v['producer'],[]).append((int(ep.group(1)) if ep else 10000,np.load(path,mmap_mode='r')))
        self.experts=[];self.contrasts=[]
        for family in sorted(families):
            values=sorted(families[family],key=lambda x:x[0]);self.experts.append(values[-1][1])
            if len(values)>2:self.contrasts.append((values[len(values)//2][1],values[0][1],values[-1][1]))
        self.native=full.cf96(phase);self.experts.extend((self.native,ne.verified_rank(phase)))
        for family in ('temporal-attention','suffix-attention','cf-temporal-attention','temporal-continuation','hybrid-attention'):
            root=ne.ROOT/('own-'+family+'-20261126');r=json.loads((root/'freeze.json').read_text());assert not r['qualifying_labels_used_for_fitting']
            self.experts.append(np.load(root/(phase+'.npy'),mmap_mode='r'))
        ix=slice(1408395,None) if idx else ret.map_rows()
        original=json.loads((joint.OUT/'manifest.json').read_text());assert not original['query_outcomes_read']
        for info in original['models'].values():
            assert ds.sha(info['vector'])==info['vector_sha256'];self.experts.append(np.memmap(info['vector'],'<f4','r')[ix])
        self.prior=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')[:,7:12]
        self.u,self.i,self.day=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        self.uc,self.ic=[np.log1p(ne.history(train,k)) for k in ('user_cnts','item_cnts')]
    def block(self,rows):
        n=self.native[rows]
        scores=np.column_stack([v[rows] for v in self.experts]).clip(1,5).astype('f')
        delta=(scores-n[:,None]).astype('f');extra=[]
        for middle,early,final in self.contrasts:extra.extend((middle[rows]-early[rows],final[rows]-middle[rows]))
        ctx=np.column_stack((delta.clip(-1.2,1.2)*2,*extra,self.prior[rows],n-3.5,self.uc[self.u[rows]]/6-1,
                            self.ic[self.i[rows]]/8-1,self.day[rows]/2000-.7,delta.std(1))).astype('f').clip(-3,3)
        return delta,ctx
    def table(self):
        a,b=self.block(slice(0,1));delta=np.empty((len(self.u),a.shape[1]),'f');ctx=np.empty((len(self.u),b.shape[1]),'f')
        for j in range(0,len(self.u),16384):
            sl=slice(j,j+16384);delta[sl],ctx[sl]=self.block(sl)
        return delta,ctx


class Net(nn.Module):
    def __init__(self,n,k):
        super().__init__()
        self.prior=nn.Parameter(torch.zeros(k));self.body=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,k+1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,x,delta):
        out=self.body(x);weights=(self.prior+1.8*out[:,:-1].tanh()).softmax(1)
        return (weights*delta).sum(1)+.075*out[:,-1].tanh()


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);y,f,_,_=anchor.history_control();r=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==r['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    d=Role('probex');raw,ctx=d.table();hold=mixed_folds(d.u);raw=torch.from_numpy(raw).to('mps');ctx=torch.from_numpy(ctx).to('mps')
    err=torch.from_numpy((y-d.native).astype('f')).to('mps');held=np.empty(len(y),'f');models=[]
    for k in range(5):
        torch.manual_seed(7361+k);torch.mps.manual_seed(7361+k);rng=np.random.default_rng(7361+k);net=Net(ctx.shape[1],raw.shape[1]).to('mps')
        opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.03);tr=np.flatnonzero(hold!=k);va=np.flatnonzero(hold==k)
        for ep in range(14):
            order=rng.permutation(tr);net.train();opt.param_groups[0]['lr']=.00028+.0012*(1+np.cos(ep*np.pi/14))/2
            for j in range(0,len(order),4096):
                ix=torch.from_numpy(order[j:j+4096]).to('mps');delta=net(ctx[ix],raw[ix]);loss=(delta-err[ix]).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        net.eval()
        with torch.inference_mode():
            for j in range(0,len(va),8192):
                ix=va[j:j+8192];t=torch.from_numpy(ix).to('mps');held[ix]=d.native[ix]+net(ctx[t],raw[t]).cpu().numpy()
        models.append(net);print('bounded raw ensemble fold',k,flush=True)
    delta=held-b;weight=lambda mask:float(np.clip(delta[mask]@(y-b)[mask]/(40+delta[mask]@delta[mask]),0,1))
    oof=np.zeros(len(y),'f')
    for k in range(5):oof[f==k]=weight(f!=k)*delta[f==k]
    w=weight(np.ones(len(y),bool));stats=blend.metrics(ne.clipped(b+oof),y,f);ctrl=blend.metrics(b,y,f);ok=ret.qualifies(stats,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=stats,control=ctrl,standalone_probe=blend.metrics(held,y,f),
             weight=w,single_execution_inputs_sha256=ds.sha(MARK),experimental_role='Separate diagnostic, outside the two preregistered comparison branches.',
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del raw,ctx,d;d=Role('qual');pred=np.zeros(len(qb),'f')
        with torch.inference_mode():
            for j in range(0,len(qb),8192):
                sl=slice(j,j+8192);v,x=[torch.from_numpy(z).to('mps') for z in d.block(sl)]
                pred[sl]=d.native[sl]+sum(m(x,v) for m in models).cpu().numpy()/5
        qd=pred-qb;np.save(OUT/'oof.npy',oof);np.save(OUT/'qual-correction.npy',w*qd)
        ne.clipped(qb+w*qd).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
