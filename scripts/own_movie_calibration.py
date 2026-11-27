"""Cross-fit partially pooled per-movie intercepts and contextual slopes on ProbeX."""
import json
import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import cg
import own_neighbor_experiment as ne
import own_history_context as hist
import own_session_latent as latent
from own_kernel_family import control

OUT=ne.ROOT/'own-movie-calibration-20261126'
PENS=(25.,150.,900.)
DIM=8


def design(query,p):
    x=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    # Same columns in both partitions: activity, near-date ratings and dispersion.
    return np.column_stack((np.ones(len(p)), p-3.6, np.clip(x[:,7]-4.5,-4,4)/2,
        np.minimum(x[:,13],5)/2,(x[:,8]-3.6),np.clip((x[:,2]-1750)/500,-3,2),
        np.clip(x[:,1],-1,1)*2, np.minimum(x[:,10],2)-.5)).astype(float)


def graph():
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    v=np.load(path)
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-7)
    r,c,w=[],[],[]
    for a in range(0,len(v),256):
        sl=slice(a,a+256)
        dots=v[sl]@v.T
        dots[np.arange(len(dots)),np.arange(a,a+len(dots))]=-1
        nb=np.argpartition(dots,-24,axis=1)[:,-24:]
        weights=np.maximum(0,np.take_along_axis(dots,nb,axis=1))**4
        r.append(np.repeat(np.arange(a,a+len(dots)),24))
        c.append(nb.ravel());w.append(weights.ravel())
    mat=csr_matrix((np.concatenate(w),(np.concatenate(r),np.concatenate(c))),shape=(len(v),len(v)))
    mat=mat+mat.T
    deg=mat.sum(axis=1).A1
    scale=diags(1/np.sqrt(np.maximum(deg,1e-7)))
    return (scale@mat@scale).tocsr()


def sums(x,ids,e,take,n):
    i=ids[take];v=x[take];ey=e[take]
    g=np.zeros((n,DIM,DIM));b=np.empty((n,DIM))
    for a in range(DIM):
        b[:,a]=np.bincount(i,weights=v[:,a]*ey,minlength=n)
        for j in range(a+1):
            z=np.bincount(i,weights=v[:,a]*v[:,j],minlength=n)
            g[:,a,j]=g[:,j,a]=z
    return g,b


def weights(g,b,pen,adj):
    mat=g.copy()
    ix=np.arange(DIM)
    mat[:,ix,ix]+=pen
    # A graph-smoothed prior shares the intercept and slopes of similar films.
    coef=np.linalg.solve(mat,b[...,None])[...,0]
    for _ in range(4):
        coef=np.linalg.solve(mat,(b+.60*pen*(adj@coef))[...,None])[...,0]
    return coef


def main():
    OUT.mkdir(exist_ok=True)
    y,f,p,q=control()
    x=design('probex',p)
    ids=ne.history('probex','item_idxs')
    n=len(ne.history('trainx','item_cnts'))
    moments=[sums(x,ids,y-p,f==j,n) for j in range(5)]
    gram=sum(a for a,b in moments);rhs=sum(b for a,b in moments)
    adj=graph()
    held=np.empty((len(y),len(PENS)))
    for j in range(5):
        m=f==j
        for k,pen in enumerate(PENS):
            w=weights(gram-moments[j][0],rhs-moments[j][1],pen,adj)
            held[m,k]=np.einsum('ij,ij->i',x[m],w[ids[m]])
    selected,stats=ne.select(held,y,f,p)
    report=dict(selected=selected,known=stats,penalties=PENS,qualifying_hashes={},
                known_comparison_is_nested=False,qualifying_labels_used_for_fitting=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if selected!='control':
        k=int(selected)
        np.save(OUT/'oof.npy',held[:,k])
        w=weights(gram,rhs,PENS[k],adj)
        qi=ne.history('qual','item_idxs')
        delta=np.einsum('ij,ij->i',design('qual',q),w[qi])
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+stats[selected]['weight']*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
