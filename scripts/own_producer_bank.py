"""Model the predictions and training trajectories of individually verified fresh producers."""
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_retained_history as retained
import own_full_context as full

OUT=ne.ROOT/'own-producer-bank-20261126'
UP=Path('../netflix_upstream_repro')
FRESH=UP/'artifacts/upstream-repro/fresh'
RECEIPTS=FRESH.parent/'fresh-receipts'
MAN=UP/'experiments/upstream-full-fwls-inputs/manifest.json'
PIN='49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c'
SKIP=('rbmx2-40','bknbrx-50','bknbrx-100')
PARENT=blend.OUT/'stage-06'


def audit_sources():
    assert ne.sha(MAN)=='67f8e3a8f5c4f18bb597d7866dd34cf50ab6883092559f206270b72623d8939e'
    manifest=json.loads(MAN.read_text())['source_sha256']
    verified={}
    for path in sorted(RECEIPTS.glob('*.json')):
        if manifest.get(str(path.relative_to(UP)))!=ne.sha(path):continue
        receipt=json.loads(path.read_text())
        name=receipt['job']
        if receipt.get('provenance')!='Fresh execution of pinned upstream source; no released prediction inputs':continue
        if receipt['source_commit']!=PIN or any(k in name for k in SKIP):continue
        verified[name]=(receipt,path)
    cols={}
    for name,(rec,src) in verified.items():
        parents=name.split('__')
        if not all('__'.join(parents[:h]) in verified for h in range(1,len(parents))):continue
        outputs=rec['outputs']
        for key,digest in outputs.items():
            if not key.startswith('preds_new/') or not key.endswith('.probex.npy'):continue
            name_q=key.replace('.probex.npy','.qual.npy')
            if name_q not in outputs:continue
            paths=[FRESH/key,FRESH/name_q]
            if not all(path.is_file() for path in paths):continue
            digests=[digest,outputs[name_q]]
            if any(manifest.get(str(path.relative_to(UP)))!=s for path,s in zip(paths,digests)):continue
            if not all(ne.sha(path)==s for path,s in zip(paths,digests)):continue
            stem=Path(key).name[:-len('.probex.npy')]
            cols[stem]=dict(producer=name,receipt=str(src),sha256=digests,paths=list(map(str,paths)))
    assert len(cols)>=80
    return cols


def project(cols,p,q,outdir=OUT):
    vectors=[np.load(v['paths'][0],mmap_mode='r') for v in cols.values()]
    sample=np.arange(0,len(p),17)
    z=np.clip(np.column_stack([m[sample] for m in vectors]),1,5)-p[sample,None]
    mean,std=z.mean(0),np.maximum(z.std(0),.02)
    z=(z-mean)/std
    eig,vec=np.linalg.eigh(z.T@z/len(z))
    rotation=vec[:,-64:]/np.sqrt(np.maximum(eig[-64:],.025))
    del z
    np.savez(outdir/'transform.npz',mean=mean,std=std,rotation=rotation)
    for phase,base,idx in [('probex',p,0),('qual',q,1)]:
        vectors=[np.load(v['paths'][idx],mmap_mode='r') for v in cols.values()]
        assert all(m.shape==(len(base),) for m in vectors)
        out=np.lib.format.open_memmap(outdir/(phase+'.npy'),mode='w+',dtype='f',shape=(len(base),64))
        for start in range(0,len(base),16384):
            sl=slice(start,start+16384)
            a=np.clip(np.column_stack([m[sl] for m in vectors]),1,5)
            out[sl]=((a-base[sl,None]-mean)/std)@rotation
            assert np.isfinite(out[sl]).all()
        out.flush()
        print(phase,'compressed',len(cols),'paired columns',flush=True)


def inputs(phase):
    x=np.load(OUT/(phase+'.npy'),mmap_mode='r')
    ctx=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
    r=json.loads((full.OUT/(phase+'-sources.json')).read_text())
    assert ne.sha(full.OUT/(phase+'.npy'))==r['feature_sha256']
    return torch.from_numpy(np.column_stack((x,ctx[:,::4])).astype('f'))


class Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.weights=nn.Sequential(nn.Linear(n,80),nn.SiLU(),nn.Dropout(.20),nn.Linear(80,65))
        nn.init.zeros_(self.weights[-1].weight)
        nn.init.zeros_(self.weights[-1].bias)
    def forward(self,x):
        w=self.weights(x).tanh()
        return .012*(w[:,:64]*x[:,:64]).sum(1)+.08*w[:,-1]


def fit(x,qx,tr,va,errors):
    torch.manual_seed(20261126)
    lo,hi=torch.quantile(x[tr[::16]],torch.tensor([.02,.98]),dim=0)
    center=(lo+hi)*.5
    scale=(hi-lo).clamp(min=.01)*.5
    def norm(t):return ((t-center)/scale).clamp(-5,5)
    model=Net(x.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.05)
    result=np.zeros(len(va))
    for ep in range(18):
        model.train()
        if ep==12:opt.param_groups[0]['lr']*=.5
        for ix in tr[torch.randperm(len(tr))].split(4096):
            pred=model(norm(x[ix]))
            loss=(pred-errors[ix]).square().mean()+.08*pred.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in (5,11,17):
            model.eval()
            with torch.inference_mode():
                for j in range(0,len(va),8192):
                    z=va[j:j+8192]
                    result[j:j+len(z)]+=model(norm(qx[z])).numpy()/3
    return result


def main():
    OUT.mkdir(exist_ok=True)
    y,f,p,q=anchor.history_control()
    fr=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy')
    qb=np.fromfile(PARENT/'selected.f32','<f4')
    cols=audit_sources()
    print('admitted',len(cols),flush=True)
    (OUT/'sources.json').write_text(json.dumps(cols,indent=2)+'\n')
    project(cols,p,q)
    torch.set_num_threads(2)
    x=inputs('probex')
    e=torch.from_numpy((y-base).astype('f'))
    held=np.empty(len(y))
    for j in range(5):
        tr,va=[torch.from_numpy(np.flatnonzero(v)) for v in (f!=j,f==j)]
        held[va.numpy()]=fit(x,x,tr,va,e)
        print('fold',j,ne.rmse(base[va]+held[va],y[va]),flush=True)
    np.save(OUT/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    eligible=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(eligible,key=lambda k:stats[k]['pooled']) if eligible else 'control'
    report=dict(selected=key,known=stats,source_receipt_sha256=ne.sha(OUT/'sources.json'),paired_columns=len(cols),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:
        delta=fit(x,inputs('qual'),torch.arange(len(y)),torch.arange(len(q)),e)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
