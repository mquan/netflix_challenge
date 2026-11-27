"""Use only forecast contrasts with similar conditional variance in both prediction roles."""
import json
import numpy as np
from scipy.linalg import eigh
import own_anchor_context as anchor
import own_direct_canonical as canonical
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_schema_receipt as receipt

OUT=ne.ROOT/'own-stable-subspace-20261126'
PARENT=blend.OUT/'ablation-06'


class Features:
    def __init__(self,phase,b):
        self.role=canonical.Role(phase);self.b=b;ref=receipt.read(canonical.MARK)
        self.n=len({v['producer'] for v in ref.values()})
    def block(self,sl):
        r=self.role;b=self.b[sl];anchor=r.native[sl]
        z=[v[sl]-anchor for v in r.experts[:self.n]]
        for m,e,t in r.contrasts:z.extend((m[sl]-e[sl],t[sl]-m[sl]))
        raw=np.column_stack(z).clip(-1.8,1.8)
        aux=np.column_stack([v[sl]-anchor for v in r.experts[self.n+1:]])
        u=r.uc[r.u[sl]]/6-1;i=r.ic[r.i[sl]]/9-1;d=r.day[sl].astype('f')/2000-.8;m=b-3.5
        context=np.column_stack((np.ones(len(b)),m,m*m,m*m*m*.3,anchor-b,aux,r.prior[sl],u,i,d,m*u,m*i,m*d,u*i))
        return raw.astype('f'),context.astype('f')
    def moments(self):
        # Fixed uniform metadata-only sample. Random sampling avoids dependence on the storage order.
        rng=np.random.default_rng(39451);rows=np.sort(rng.choice(len(self.b),min(160000,len(self.b)),replace=False))
        raw,context=self.block(rows);g=context.T@context;g.flat[::len(g)+1]+=.2
        self.coef=np.linalg.solve(g,context.T@raw);z=raw-context@self.coef
        return z.T@z/len(z)
    def project(self,sl,axes):
        raw,c=self.block(sl);v=((raw-c@self.coef)@axes).clip(-4,4)
        # Mild activity/score interactions allow a stable contrast to change with the forecast context.
        return np.column_stack((v,.35*v*np.tanh(c[:,1,None]),.35*v*np.tanh(c[:,-2,None]))).astype('f')


def main():
    OUT.mkdir(exist_ok=True);fr=receipt.read(PARENT/'freeze.json');assert receipt.streamed_sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    protocol=dict(parent=str(PARENT),registered_role='Separate diagnostic outside the two comparison branches.',ridge=1200,
                  allowed_generalized_covariance_ratio=[.83,1.20],maximum_absolute_rating_update=.12,
                  development_gate='Lower pooled RMSE and at least four improved user folds.',qualifying_ratings_read=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    y,f,_,_=anchor.history_control();b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    dev=Features('probex',b);qual=Features('qual',qb);dc=dev.moments().astype(float);qc=qual.moments().astype(float)
    ev,axis=eigh((dc+qc)*.5);keep=ev>.000025;basis=axis[:,keep]/np.sqrt(ev[keep])
    left=basis.T@dc@basis;right=basis.T@qc@basis+np.eye(keep.sum())*.0001
    ratio,vec=eigh(left,right);stable=(ratio>.83)&(ratio<1.20);assert stable.any()
    axes=np.ascontiguousarray(basis@vec[:,stable],'f');print('stable directions',stable.sum(),'of',len(ratio),flush=True)
    width=3*stable.sum();g=np.zeros((5,width,width));r=np.zeros((5,width))
    for j in range(0,len(y),16384):
        sl=slice(j,j+16384);x=dev.project(sl,axes)
        for k in range(5):
            ix=f[sl]==k;a=x[ix];g[k]+=a.T@a;r[k]+=a.T@(y[sl]-b[sl])[ix]
    def fit(a,z):
        p=a.copy();p.flat[::len(a)+1]+=1200;return np.linalg.solve(p,z)
    ws=np.stack([fit(g.sum(0)-g[k],r.sum(0)-r[k]) for k in range(5)]);w=fit(g.sum(0),r.sum(0));delta=np.empty(len(y))
    for j in range(0,len(y),16384):
        sl=slice(j,j+16384);x=dev.project(sl,axes);delta[sl]=np.einsum('ij,ij->i',x,ws[f[sl].astype('i')]).clip(-.12,.12)
    stats={str(t):blend.metrics(ne.clipped(b+t*delta),y,f) for t in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];key=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=stats,raw_inputs_sha256=receipt.streamed_sha(canonical.MARK),eigenvalues=ratio.tolist(),retained=int(stable.sum()),
             qualifying_hashes={},qualifying_labels_used_for_fitting_or_selection=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        pred=np.empty_like(qb)
        for j in range(0,len(qb),16384):
            sl=slice(j,j+16384);pred[sl]=ne.clipped(qb[sl]+float(key)*(qual.project(sl,axes)@w).clip(-.12,.12))
        pred.astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=receipt.streamed_sha(OUT/'selected.f32')
        np.savez(OUT/'fit.npz',qual_residualizer=qual.coef,axes=axes,weights=w,dev_residualizer=dev.coef,dev_fold_weights=ws)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
