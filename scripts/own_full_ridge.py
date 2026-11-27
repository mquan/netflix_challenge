"""Use every verified own-model checkpoint in a signed ridge fit with model-family context effects."""
import json
from pathlib import Path
import re
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_producer_bank as producer
import own_trajectory_bank as traj
import own_retained_history as retained

OUT=ne.ROOT/'own-full-ridge-20261126'


def sources():
    a=producer.OUT/'sources.json';b=traj.OUT/'sources.json'
    assert ne.sha(a)==json.loads((producer.OUT/'freeze.json').read_text())['source_receipt_sha256']
    assert ne.sha(b)==json.loads((traj.OUT/'freeze.json').read_text())['sources_sha256']
    cols={};seen=set()
    for name,obj in list(json.loads(a.read_text()).items())+list(json.loads(b.read_text())['predictions'].items()):
        pair=tuple(obj['sha256'])
        if pair in seen:continue
        seen.add(pair);cols[name]=obj
    return cols


class Inputs:
    def __init__(self,cols,phase,base,transform=None):
        idx=int(phase=='qual');tr='fulltrain' if idx else 'trainx'
        self.base=base;self.vectors=[];family=[]
        for name,rec in cols.items():
            p=Path(rec['paths'][idx]);assert ne.sha(p)==rec['sha256'][idx]
            v=np.load(p,mmap_mode='r');assert v.shape==base.shape
            self.vectors.append(v);stem=name.split('/')[-1]
            family.append(re.sub(r'_ep[0-9]+.*$','',stem))
        names=sorted(set(family));self.groups=[[i for i,a in enumerate(family) if a==k] for k in names]
        u,m=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
        uc,ic=[ne.history(tr,k) for k in ('user_cnts','item_cnts')]
        self.gates=np.column_stack((np.tanh((base-3.5)/.7),np.tanh((np.log1p(uc[u])-5)/1.4),
                  np.tanh((np.log1p(ic[m])-8)/2),np.tanh((ne.history(phase,'dates')-1750)/400))).astype('f')
        if transform is None:
            sample=slice(None,None,25)
            x=self.raw(sample);transform=(x.mean(0),np.maximum(x.std(0),.035))
        self.transform=transform
    def raw(self,rows):
        x=np.column_stack([v[rows] for v in self.vectors]).astype('f')
        np.clip(x,-1,7,out=x);x-=self.base[rows,None].astype('f')
        assert np.isfinite(x).all()
        return x
    def block(self,rows):
        raw=self.raw(rows);mu,sd=self.transform;z=((raw-mu)/sd).clip(-12,12)
        g=np.column_stack([z[:,ids].mean(1) for ids in self.groups])
        gate=self.gates[rows]
        tails=np.column_stack(((np.maximum(1-self.base[rows,None]-raw,0)).mean(1),
                               (np.maximum(self.base[rows,None]+raw-5,0)).mean(1)))
        # Family-level interaction terms share a coefficient across nearby epoch snapshots.
        return np.column_stack((z,g,.40*(g[:,:,None]*gate[:,None,:]).reshape(len(g),-1),gate,tails,np.ones(len(g)))).astype('f')


def solve(g,b,pen):
    x=g.copy();x.flat[::len(x)+1]+=pen
    return cho_solve(cho_factor(x,check_finite=False),b,check_finite=False)


def main():
    OUT.mkdir(exist_ok=True);y,f,_,_=anchor.history_control()
    root=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((root/'freeze.json').read_text());assert ne.sha(root/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(root/'known-selected.npy');qb=np.fromfile(root/'selected.f32','<f4')
    cols=sources();data=Inputs(cols,'probex',base);print('admitted',len(cols),'group count',len(data.groups),flush=True)
    width=data.block(slice(0,1)).shape[1];gs=np.zeros((5,width,width));rhs=np.zeros((5,width))
    err=y-base
    for start in range(0,len(y),8192):
        sl=slice(start,start+8192);x=data.block(sl)
        for j in range(5):
            g=f[sl]==j;a=x[g];gs[j]+=a.T@a;rhs[j]+=a.T@err[sl][g]
    total=gs.sum(0);rr=rhs.sum(0);pens=(300.,3000.,30000.)
    ws=np.stack([np.column_stack([solve(total-gs[j],rr-rhs[j],pen) for pen in pens]) for j in range(5)])
    held=np.empty((len(y),len(pens)))
    for start in range(0,len(y),8192):
        sl=slice(start,start+8192);x=data.block(sl)
        for j in range(5):
            mask=f[sl]==j;held[sl][mask]=x[mask]@ws[j]
    orig=blend.metrics(base,y,f);stats={str(pen):blend.metrics(ne.clipped(base+held[:,k]),y,f) for k,pen in enumerate(pens)}
    ok=[k for k,pen in enumerate(pens) if retained.qualifies(stats[str(pen)],orig)]
    k=min(ok,key=lambda a:stats[str(pens[a])]['pooled']) if ok else None
    rep=dict(selected=str(pens[k]) if ok else 'control',known=stats,control=orig,parent=str(root),
           columns=len(cols),width=width,source_manifests=[ne.sha(producer.OUT/'sources.json'),ne.sha(traj.OUT/'sources.json')],
           qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        w=solve(total,rr,pens[k]);np.save(OUT/'oof.npy',held[:,k])
        data=Inputs(cols,'qual',qb,data.transform);delta=np.empty(len(qb))
        for j in range(0,len(qb),8192):
            sl=slice(j,j+8192);delta[sl]=data.block(sl)@w
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
