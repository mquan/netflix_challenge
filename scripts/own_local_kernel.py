"""Learn partially pooled nonlinear error functions for individual films using held-user rating errors."""
import json
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.linalg import cho_factor,cho_solve
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_extra_joint as extra
import own_joint_cohort_source as joint
import own_joint_jacobian as jac
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as cf
import own_peer_replay as peer
import own_rbm_joint as rbm
import own_retained_history as ret
from own_empirical_neighborhood import fit
OUT=ne.ROOT/'own-local-kernel-20261126'
PARENT=blend.OUT/'schema-07'


class Data:
    def __init__(self):
        self.c=c=peer.Cohort();self.m=c.m;self.u=c.u
        self.fold=np.empty(len(c.rows),'i1')
        canonical=ne.history('trainx','uidx_to_uid');path=ret.QUAL/'dataset/dense-to-raw-user.npy'
        assert ne.sha(path)==json.loads((path.parent/'manifest.json').read_text())['outputs'][path.name]
        mapping=np.searchsorted(canonical,np.load(path));self.fold[:]=ne.user_folds(mapping[c.u])
        ucoord=[np.fromfile(jac.OUT/(name+'.users.f32'),'<f4').reshape(len(mapping),12) for name in jac.MODELS]
        self.user=np.column_stack(ucoord);self.user=((self.user-self.user.mean(0))/self.user.std(0).clip(.001)).clip(-5,5)
        self.activity=np.log1p(np.diff(c.d['train_offsets'])).astype('f')
        rec=json.loads((rbm.OUT/'freeze.json').read_text());parts=[]
        for name in rbm.MODELS:
            p=rbm.OUT/(name+'.f32');assert rbm.ds.sha(p)==rec['signals'][str(p)]
            parts.append(np.memmap(p,'<f4','r').reshape(-1,7)[:,0])
        jr=json.loads((joint.OUT/'manifest.json').read_text());er=json.loads((extra.OUT/'manifest.json').read_text())
        for info in list(jr['models'].values())+list(er['inputs'].values()):
            assert ne.sha(info['vector'])==info['vector_sha256'];parts.append(np.memmap(info['vector'],'<f4','r'))
        h=np.load(peer.OUT/'effects.npy',mmap_mode='r');p=np.load(cf.OUT/'probabilities.npy',mmap_mode='r')
        assert ne.sha(peer.OUT/'effects.npy')==json.loads((peer.OUT/'manifest.json').read_text())['output_sha256']
        assert ne.sha(cf.OUT/'probabilities.npy')==json.loads((cf.OUT/'manifest.json').read_text())['probabilities_sha256']
        parts.extend(h[:,k,0] for k in range(3));parts.append(p[:,0].astype('f')@np.arange(1,6,dtype='f'))
        self.raw=np.column_stack(parts).astype('f');self.ref=.50*self.raw[:,-1]+.25*self.raw[:,:3].mean(1)+.25*self.raw[:,3:11].mean(1)
        self.raw-=self.ref[:,None];self.old=cov.original()[-1]
        _,self.cluster=kmeans2(cov.movie_geometry(),88,iter=14,minit='++',seed=13685)
        self.star=c.rows['rating'].astype('f')
        self.rff=np.random.default_rng(14339).normal(size=(self.user.shape[1]+self.raw.shape[1]+4,48)).astype('f')
        self.phase=np.random.default_rng(1331).uniform(0,2*np.pi,48).astype('f')
        x=self.context(c.ix[::19]);self.center=x.mean(0);self.scale=x.std(0).clip(.045)
    def context(self,rows):
        c=self.c;u=self.u[rows];return np.column_stack((self.user[u],self.raw[rows],self.ref[rows]-3.5,self.activity[u],
                                            (c.day[rows].astype('f')-1950)/500,self.raw[rows].std(1))).astype('f')
    def design(self,rows):
        z=((self.context(rows)-self.center)/self.scale).clip(-4,4)
        ray=z@self.rff/np.sqrt(z.shape[1])
        return np.column_stack((np.ones(len(rows),'f'),z[:,24:],z[:,:24]*.55,.7*np.cos(ray+self.phase),
                               .45*np.cos(2.2*ray+self.phase))).astype('f')
    def user_updates(self,rows,mode):
        c=self.c;ar,masks=c.masks(rows);mask=masks[mode];err=self.star[ar]-self.old[ar]
        # A user's anchor errors inform the weights of models that disagree in the same direction on the target.
        scale=self.scale[24:24+self.raw.shape[1]]
        x=(self.raw[rows]/scale).clip(-4,4);a=(self.raw[ar]/scale).clip(-4,4)
        date=abs(c.day[rows,None].astype('i')-c.day[ar]);near=1/(1+date/35)
        out=[]
        for sl in (slice(0,3),slice(3,11),slice(11,None),slice(None)):
            x0=x[:,sl];a0=a[:,:,sl];dim=x0.shape[1]
            for time in (False,True):
                obs=mask*(np.sqrt(near) if time else 1)
                aa=a0*obs[:,:,None];g=aa@aa.transpose(0,2,1)/dim
                diag=np.arange(g.shape[1]);g[:,diag,diag]+=1.8
                w=np.linalg.solve(g,(err*obs)[:,:,None])[:,:,0]
                cross=np.einsum('bd,bnd->bn',x0,aa)/dim
                out.append((cross*w).sum(1))
        return np.column_stack(out).astype('f')


def solve(gram,rhs,pen):
    a=gram.copy();a.flat[::len(a)+1]+=pen
    return cho_solve(cho_factor(a,check_finite=False),rhs,check_finite=False)


def kernel(d,source,train_y,query,group,pens,same):
    # Fit joint group-level and film-level terms separately. Each held user is absent from every fitted function.
    gsrc=group[d.m[source]];gq=group[d.m[query]]
    tr=np.argsort(gsrc,kind='stable');va=np.argsort(gq,kind='stable');n=int(group.max())+1
    to=np.r_[0,np.bincount(gsrc,minlength=n).cumsum()];vo=np.r_[0,np.bincount(gq,minlength=n).cumsum()]
    result=np.zeros((len(query),len(pens),train_y.shape[1]),'f')
    for k in np.flatnonzero(np.diff(to)*np.diff(vo)):
        a=tr[to[k]:to[k+1]];b=va[vo[k]:vo[k+1]];x=d.design(source[a]);xd=d.design(query[b])
        folds=d.fold[source[a]];tv=d.fold[query[b]]
        gram=x.T@x;rhs=x.T@train_y[a]
        if same:
            for j in range(5):
                v=tv==j;remove=folds==j
                if not v.any():continue
                xx=x[remove];gg=gram-xx.T@xx;rr=rhs-xx.T@train_y[a][remove]
                for c,p in enumerate(pens):result[b[v],c]=xd[v]@solve(gg,rr,p)
        else:
            for c,p in enumerate(pens):result[b,c]=xd@solve(gram,rhs,p)
    return result.reshape(len(query),-1)


def main():
    OUT.mkdir(exist_ok=True);d=Data();c=d.c;y,f,_,_=anchor.history_control();np.testing.assert_array_equal(d.fold[c.ix],f)
    rec=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    source=c.ix;labels=np.column_stack((y-base,y-d.ref[source])).astype('f')
    def make(ids,known):
        groups=[(np.arange(len(d.cluster)),(180.,1150.)),(d.cluster,(1000.,5200.))]
        parts=[kernel(d,source,labels,ids,g,p,known) for g,p in groups]
        local=np.empty((len(ids),8),'f')
        for j in range(0,len(ids),8192):
            sl=slice(j,j+8192);local[sl]=d.user_updates(ids[sl],1 if known else 2)
        return np.column_stack((*parts,local)).astype(float)
    x=make(c.ix,True);held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),qualifying_hashes={},
        qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        delta=make(np.arange(c.n,len(c.rows)),False)@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
