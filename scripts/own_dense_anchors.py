"""Learn residual transfer from real visible anchors and simulated denser within-user histories."""
import json
import numpy as np
from pathlib import Path
import own_anchor_covariance as cov
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
import own_history_context as hist

OUT=ne.ROOT/'own-dense-anchors-20261126'
PARENT=blend.OUT/'stage-07'


def raw():
    u,i,d,prior=cov.original()
    names=retained.MODELS
    mods=[]
    for name in names:
        rec=json.loads((retained.QUAL/name/'report.json').read_text())
        p=Path(rec['column'])
        assert ne.sha(p)==rec['column_sha256']
        mods.append(np.fromfile(p,'<f4'))
    return u,i,d,prior,np.column_stack(mods)


def features(query,base,u,i,d,pred,models,ratings,take):
    qu,qi,qd=[ne.history(query,k) for k in ('user_idxs','item_idxs','dates')]
    g=cov.movie_geometry()
    pop=np.log1p(ne.history('trainx','item_cnts'))
    order=np.flatnonzero(take)
    order=order[np.argsort(u[order],kind='stable')]
    off=np.r_[0,np.bincount(u[order],minlength=480189).cumsum()]
    counts=np.diff(off)
    n=len(base)
    raw=np.zeros((n,8,5),'f')
    sq=np.zeros((n,8),'f')
    for pos in range(int(counts.max())):
        ix=np.flatnonzero(counts[qu]>pos)
        for a in range(0,len(ix),32768):
            j=ix[a:a+32768]
            other=order[off[qu[j]]+pos]
            keep=qi[j]!=i[other]
            j,other=j[keep],other[keep]
            sim=np.einsum('ij,ij->i',g[qi[j]],g[i[other]])
            gap=np.abs(qd[j]-d[other])
            day=(gap==0).astype('f')
            near=1/(1+gap/24.)
            affinity=np.exp(2*(sim-1))
            agreement=np.exp(-np.abs(base[j]-pred[other]))
            wp=np.exp(-np.abs(pop[qi[j]]-pop[i[other]])*.7)
            kernel=np.column_stack((np.ones(len(j)),day,near,affinity,near*affinity,agreement,agreement*near,wp))
            e=ratings[other]-pred[other]
            m=models[other]
            val=np.column_stack((np.ones(len(j)),e,ratings[other]-m[:,0],(e**2)-.8,(ratings[other]>=4)-(pred[other]-2.5)*.45-.25))
            raw[j]+=kernel[:,:,None]*val[:,None,:]
            sq[j]+=kernel**2
    count=raw[:,:,0]
    den=.75+count
    raw[:,:,1:]/=den[:,:,None]
    raw[:,:,0]=np.log1p(count)
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    z=np.column_stack((np.clip((base-3.5)/1.5,-2,2),np.tanh((h[:,7]-4)/2),np.tanh(h[:,13]-1),
                       np.tanh(h[:,1]),np.tanh((h[:,2]-1600)/500)))
    # Pure residual transfer vanishes when no anchor is observed. Shrink the
    # effective count to keep several close anchors from dominating a single user.
    v=raw[:,:,1:].reshape(n,-1)
    return np.column_stack((v,(v[:,:,None]*z[:,None,:]).reshape(n,-1),raw[:,:,0]*raw[:,:,1])).astype('f')


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    rec=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy')
    qb=np.fromfile(PARENT/'selected.f32','<f4')
    u,i,d,pred,m=raw()
    visible,idx=anchor.training_indices('trainx',u,i,d)
    ratings=np.full(len(u),np.nan,'f')
    ratings[visible]=ne.history('trainx','ratings')[idx]
    x=features('probex',base,u,i,d,pred,m,ratings,visible)
    ix=retained.map_rows()
    assert np.all(~visible[ix]) and len(ix)+visible.sum()==len(u)
    ratings[ix]=y
    # Pseudo histories only enter training for users outside the validation
    # fold. Validation always uses x, whose inputs are trainx-only ratings.
    pseudo=features('probex',base,u,i,d,pred,m,ratings,np.ones(len(u),bool))
    def equations(v,sel,e):
        gram=np.zeros((v.shape[1],v.shape[1]))
        rhs=np.zeros(v.shape[1])
        for j in range(0,len(sel),32768):
            rows=sel[j:j+32768]
            a=v[rows].astype(float)
            gram+=a.T@a;rhs+=a.T@e[rows]
        return gram,rhs
    err=y-base
    moments=[[equations(v,np.flatnonzero(f==j),err) for j in range(5)] for v in (x,pseudo)]
    def solve(exclude):
        acc=[]
        for group in moments:
            gg=sum(g for j,(g,b) in enumerate(group) if j!=exclude)
            b=sum(v for j,(g,v) in enumerate(group) if j!=exclude)
            acc.append((gg,b))
        g=acc[0][0]+.20*acc[1][0]
        g.flat[::len(g)+1]+=800.
        return np.linalg.solve(g,acc[0][1]+.20*acc[1][1])
    held=np.empty(len(y))
    for j in range(5):held[f==j]=x[f==j]@solve(j)
    np.save(OUT/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    eligible=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(eligible,key=lambda k:stats[k]['pooled']) if eligible else 'control'
    report=dict(selected=key,known=stats,pseudo_training_weight=.20,qualifying_hashes={},
                qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:
        del x,pseudo
        present,ix=anchor.training_indices('fulltrain',u,i,d)
        np.testing.assert_array_equal(ratings,ne.history('fulltrain','ratings')[ix])
        xq=features('qual',qb,u,i,d,pred,m,ratings,present)
        delta=xq@solve(-1)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
