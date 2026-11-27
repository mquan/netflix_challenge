"""Train on the complete native split-3 holdout with user-disjoint validation on ProbeX."""
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_retained_history as retained
import own_session_latent as latent
import own_producer_bank as bank
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-native-holdout-20261126'
UP=bank.UP
DATA=UP/'artifacts/upstream-split3-pilot/source/data'
MAN=UP/'experiments/upstream-remote15h-all-snapshots/manifest.json'
DIGEST='dbe055690f1f200437b1ba28937b4d003da9ba70e1bd9f34ffbc1fe058d74e01'


def identities():
    assert ne.sha(MAN)==DIGEST
    sig=json.loads(MAN.read_text())['source_sha256']
    def array(p):
        assert ne.sha(p)==sig[str(p.relative_to(UP))]
        return np.load(p,mmap_mode='r')
    u,i,d,y=[array(DATA/'probe_3'/(k+'.npy')) for k in ('user_idxs','item_idxs','dates','ratings')]
    ix=array(UP/'artifacts/upstream-remote15h-native-stream/probex-row-map.npy')
    for k,native in (('user_idxs',u),('item_idxs',i),('dates',d),('ratings',y)):
        np.testing.assert_array_equal(native[ix],ne.history('probex',k))
    assert len(np.unique(ix))==len(ix)
    return u,i,d,y,ix


def family_pairs():
    families={}
    for study,jobs in [('upstream-split3-pilot',['s3-attn16']),('upstream-split3-next3',['s3-attn64','s3-rbm200','s3-tx100r']),
                       ('upstream-split3-diversity',['s3-nade12s','s3-dnn32nm','s3-dnn16t'])]:
        for job in jobs:
            exp=UP/'experiments'/study
            pilot=study=='upstream-split3-pilot'
            rp=exp/('paired-receipt.json' if pilot else job+'-receipt.json')
            rec=json.loads(rp.read_text())
            proof=json.loads((exp/('completion-verification.json' if pilot else job+'-verification.json')).read_text())
            assert rec['source_pin']==bank.PIN and not rec['public_prediction_values_read'] and not rec['qualifying_labels_read']
            assert proof['qualifying_targets_zero'] and proof['full_known_projection_recomputed']
            if pilot:assert proof['paired_receipt_sha256']==ne.sha(rp)
            else:assert proof['known_training_disjointness_bound_to_unchanged_original_data'] and proof['both_full_epoch_traces']
            by_name={}
            for k,s in rec['outputs'].items():
                if not k.endswith('.probe_3.npy'):continue
                name=Path(k).name.removesuffix('.probe_3.npy')
                if name!=job and not re.fullmatch(re.escape(job)+r'_ep\d+',name):continue
                q=k.replace('.probe_3.npy','.qual.npy')
                paths=[UP/'artifacts'/study/'source'/z for z in (k,q)]
                if not all(pp.exists() and ne.sha(pp)==sig for pp,sig in zip(paths,(s,rec['outputs'][q]))):continue
                by_name[name]=list(map(str,paths))
            assert job in by_name
            families[job]=by_name
    native=json.loads(MAN.read_text())
    pr=json.loads((UP/'experiments/upstream-remote15h-native-stream/independent-verification.json').read_text())
    assert pr['complete_native_holdout_absent_from_training'] and pr['original_source_data_and_native_receipt_hashes_verified']
    assert not pr['qualifying_labels_read']
    for row in native['rows']:
        stem=row['name'];job=stem.split('_ep')[0]
        if job not in ('s3-attn32','s3-dnn24','s3-rbm200e'):continue
        paths=[row['native_known'],row['paired_files']['qual']]
        if not all(ne.sha(UP/k)==native['source_sha256'][k] for k in paths):continue
        families.setdefault(job,{})[stem]=[str(UP/k) for k in paths]
    return families


def form(phase,families,ids):
    isqual=phase=='qual'
    u,i,d=ids
    train='fulltrain' if isqual else 'train_3'
    tu,ti,stars,uc,ic=[np.load(DATA/train/(k+'.npy'),mmap_mode='r') for k in
                        ('user_idxs','item_idxs','ratings','user_cnts','item_cnts')]
    means=[]
    for ent,cnt in ((tu,uc),(ti,ic)):
        tables=np.empty((len(cnt),5),'f')
        for rating in range(1,6):tables[:,rating-1]=(np.bincount(ent,weights=stars==rating,minlength=len(cnt))+3)/(cnt+15)
        means.append(tables)
    gpath=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(gpath)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    g=np.load(gpath)
    _,vec=np.linalg.eigh(g.T@g);geom=g@vec[:,-16:]
    geom=(geom-geom.mean(0))/np.maximum(geom.std(0),1e-8)
    entries=[];terms=[]
    for job,rows in families.items():
        v=np.clip(np.load(rows[job][int(isqual)],mmap_mode='r'),1,5)
        terms.append(v)
        ep=[k for k in sorted(rows) if '_ep' in k]
        if ep:
            for group in np.array_split(ep,min(3,len(ep))):
                vals=np.zeros(len(v),'f')
                for k in group:vals+=np.clip(np.load(rows[k][int(isqual)],mmap_mode='r'),1,5)/len(group)
                entries.append(vals-v)
    scores=np.column_stack(terms)
    prior=scores.mean(1)
    x=np.column_stack((prior,scores-prior[:,None],*entries,np.log1p(uc[u]),np.log1p(ic[i]),d/2000,
                       means[0][u],means[1][i],geom[i])).astype('f')
    return torch.from_numpy(x),prior


class Model(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.14),
                               nn.Linear(128,64),nn.SiLU(),nn.Dropout(.10),nn.Linear(64,6))
    def forward(self,x):return self.net(x)


def predict(x,qx,targets,prior,query_prior,tr,va):
    torch.manual_seed(20261126)
    model=Model(x.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
    lo,hi=torch.quantile(x[tr[::12]],torch.tensor([.02,.98]),dim=0)
    mid=(lo+hi)*.5
    scale=(hi-lo).clamp(min=.005)*.5
    def norm(z):return ((z-mid)/scale).clamp(-5,5)
    stars=torch.arange(1,6,dtype=torch.float32)
    result=np.zeros((len(va),2),'f')
    for ep in range(19):
        model.train()
        if ep in (10,15):opt.param_groups[0]['lr']*=.5
        for ii in tr[torch.randperm(len(tr))].split(4096):
            vals=model(norm(x[ii]))
            raw=prior[ii]+.4*vals[:,0]
            probs=vals[:,1:]
            mu=probs.softmax(1)@stars
            y=targets[ii]
            loss=(raw-y).square().mean()+.35*(mu-y).square().mean()+.2*F.cross_entropy(probs,y.long()-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in (8,13,18):
            model.eval()
            with torch.inference_mode():
                for a in range(0,len(va),8192):
                    ii=va[a:a+8192]
                    vals=model(norm(qx[ii]))
                    out=torch.column_stack((query_prior[ii]+.4*vals[:,0],vals[:,1:].softmax(1)@stars))
                    result[a:a+len(ii)]+=out.numpy()/3
    return result


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    u,i,d,stars,ix=identities()
    families=family_pairs()
    (OUT/'families.json').write_text(json.dumps(families,indent=2)+'\n')
    x,pr=form('probe_3',families,(u,i,d))
    print('native holdout',len(u),'canonical rows',len(ix),'families',len(families),flush=True)
    fold=ne.user_folds(u)
    np.testing.assert_array_equal(fold[ix],f)
    target=torch.from_numpy(stars.astype('f'))
    prior=torch.from_numpy(pr)
    native=np.empty((len(u),2),'f')
    for j in range(5):
        tr,va=[torch.from_numpy(np.flatnonzero(v)) for v in (fold!=j,fold==j)]
        native[va.numpy()]=predict(x,x,target,prior,prior,tr,va)
        print('native user fold',j,ne.rmse(native[va.numpy(),0],stars[va.numpy()]),flush=True)
    np.save(OUT/'native-oof.npy',native)
    known=native[ix]
    np.save(OUT/'probex.npy',known)
    known=known.astype(float)-base[:,None]
    held=np.empty(len(y))
    for j in range(5):
        tr=f!=j
        held[~tr]=known[~tr]@fit(known[tr],(y-base)[tr])
    w=fit(known,y-base)
    stats=blend.metrics(ne.clipped(base+held),y,f)
    orig=blend.metrics(base,y,f)
    eligible=retained.qualifies(stats,orig)
    np.save(OUT/'oof.npy',held)
    report=dict(selected='candidate' if eligible else 'control',known=stats,control=orig,native_rows=len(u),weights=w.tolist(),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False,parent=str(parent))
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:
        xx,qr=form('qual',families,[ne.history('qual',k) for k in ('user_idxs','item_idxs','dates')])
        qual=predict(x,xx,target,prior,torch.from_numpy(qr),torch.arange(len(u)),torch.arange(len(qb)))
        np.save(OUT/'qual.npy',qual)
        delta=(qual.astype(float)-qb[:,None])@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
