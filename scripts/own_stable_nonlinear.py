"""Screen nonlinear forecast contrasts for covariance stability before fitting their rating corrections."""
import json
import numpy as np
from scipy.linalg import eigh
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_schema_receipt as receipt
import own_stable_subspace as linear

OUT=ne.ROOT/'own-stable-nonlinear-20261126'


class Role:
    def __init__(self,phase,base,saved,selected):
        self.data=linear.Features(phase,base);self.known=phase=='probex'
        self.data.coef=saved['dev_residualizer' if self.known else 'qual_residualizer'];self.axes=saved['axes']
        self.first=saved['dev_fold_weights' if self.known else 'weights'];self.factor=float(selected)
        n=self.axes.shape[1];rng=np.random.default_rng(74013)
        self.rp=rng.normal(size=(n+5,192)).astype('f')/np.sqrt(n+5);self.offset=rng.uniform(-1,1,192).astype('f')
        self.gate=rng.normal(size=(5,96)).astype('f')*.55;self.pair=rng.integers(0,n,96)
    def features(self,rows):
        raw,c=self.data.block(rows);x=((raw-c@self.data.coef)@self.axes).clip(-4,4)
        meta=np.column_stack((c[:,1],c[:,4]*3,c[:,-7],c[:,-6],c[:,-5])).clip(-4,4)
        z=np.column_stack((x,meta));hidden=np.tanh(z@self.rp+self.offset)
        return np.column_stack((hidden,x[:,self.pair]*np.tanh(meta@self.gate),x*x/(1+x*x))).astype('f')
    def sample(self):
        ids=np.sort(np.random.default_rng(5387).choice(len(self.data.b),120000,replace=False));v=self.features(ids)
        self.center=v.mean(0);v-=self.center;return v.T@v/len(v)
    def prior(self,rows,folds):
        if self.known:x=self.data.project(rows,self.axes);z=np.einsum('ij,ij->i',x,self.first[folds[rows]])
        else:z=self.data.project(rows,self.axes)@self.first
        return ne.clipped(self.data.b[rows]+self.factor*z.clip(-.12,.12))


def main():
    OUT.mkdir(exist_ok=True);src=linear.OUT;parent=receipt.read(src/'freeze.json');assert parent['qualifying_hashes']
    spec=dict(parent=str(src),parent_freeze_sha256=receipt.streamed_sha(src/'freeze.json'),fit_sha256=receipt.streamed_sha(src/'fit.npz'),
        role='Exploratory continuation of the separate covariance-stable linear model.',selection='Fixed one-half dose must improve all five held-user folds and at least three out of four movie popularity groups.',
        random_features=192,ridge=3800,allowed_covariance_ratio=[.90,1.10],maximum_update=.045,qualifying_labels_read=False)
    (OUT/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    y,f,_,_=anchor.history_control();f=f.astype('i');save=np.load(src/'fit.npz');b=np.load(linear.PARENT/'known-selected.npy');qb=np.fromfile(linear.PARENT/'selected.f32','<f4')
    d=Role('probex',b,save,parent['selected']);q=Role('qual',qb,save,parent['selected']);pr=np.empty(len(y));qp=np.fromfile(src/'selected.f32','<f4')
    assert receipt.streamed_sha(src/'selected.f32')==parent['qualifying_hashes']['selected'];exact=0
    for j in range(0,len(qp),16384):
        sl=slice(j,j+16384);exact+=int(np.count_nonzero(q.prior(sl,None).astype('f')!=qp[sl]))
    assert exact==0
    for j in range(0,len(y),16384):
        sl=slice(j,j+16384);pr[sl]=d.prior(sl,f)
    assert abs(ne.rmse(pr,y)-parent['known'][parent['selected']]['pooled'])<1e-12
    ca=d.sample().astype(float);cb=q.sample().astype(float);ev,v=eigh((ca+cb)/2);keep=ev>.001;basis=v[:,keep]/np.sqrt(ev[keep])
    a=basis.T@ca@basis;c=basis.T@cb@basis+np.eye(keep.sum())*.0001;ratio,v=eigh(a,c)
    keep=(ratio>.90)&(ratio<1.10);ax=(basis@v[:,keep]).astype('f');n=int(keep.sum());assert n>0;print('retained nonlinear axes',n,flush=True)
    def projected(role,sl):return ((role.features(sl)-role.center)@ax).clip(-4,4)
    g=np.zeros((5,n,n));r=np.zeros((5,n))
    for j in range(0,len(y),16384):
        sl=slice(j,j+16384);x=projected(d,sl)
        for k in range(5):
            ix=f[sl]==k;v=x[ix];g[k]+=v.T@v;r[k]+=v.T@(y[sl]-pr[sl])[ix]
    def fit(gram,rhs):return np.linalg.solve(gram+np.eye(n)*3800,rhs)
    w=fit(g.sum(0),r.sum(0));ws=np.stack([fit(g.sum(0)-g[k],r.sum(0)-r[k]) for k in range(5)]);out=pr.copy()
    for j in range(0,len(y),16384):
        sl=slice(j,j+16384);out[sl]=ne.clipped(pr[sl]+.5*np.einsum('ij,ij->i',projected(d,sl),ws[f[sl]]).clip(-.09,.09))
    new=blend.metrics(out,y,f);old=blend.metrics(pr,y,f);pop=ne.history('trainx','item_cnts')[d.data.role.i];bins=np.digitize(pop,[1500,11000,55000])
    parts=[dict(rows=int((bins==k).sum()),control=ne.rmse(pr[bins==k],y[bins==k]),candidate=ne.rmse(out[bins==k],y[bins==k])) for k in range(4)]
    ok=all(a<b for a,b in zip(new['folds'],old['folds'])) and sum(v['candidate']<v['control'] for v in parts)>=3
    rep=dict(selected='candidate' if ok else 'control',known=new,control=old,popularity_groups=parts,retained=n,parent_reconstruction_different_float32=exact,
             qualifying_hashes={},qualifying_labels_used_for_fitting_or_selection=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n');print('nonlinear selection',rep['selected'],flush=True)
    if ok:
        qual=np.empty_like(qp)
        for j in range(0,len(qp),16384):
            sl=slice(j,j+16384);qual[sl]=ne.clipped(qp[sl]+.5*(projected(q,sl)@w).clip(-.09,.09))
        qual.astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=receipt.streamed_sha(OUT/'selected.f32')
        np.savez(OUT/'fit.npz',axes=ax,qual_center=q.center,dev_center=d.center,weights=w,fold_weights=ws)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
