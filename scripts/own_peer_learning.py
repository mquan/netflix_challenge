"""Learn conditional rating forecasts from held users with physically replayed subsets of their visible peers."""
import json
from pathlib import Path
from time import sleep
import numpy as np
from catboost import CatBoostRegressor
from scipy.optimize import lsq_linear
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_extra_joint as extra
import own_full_context as full
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as cf
import own_peer_replay as replay
import own_retained_history as ret
OUT=ne.ROOT/'own-peer-learning-20261126'
PARENT=blend.OUT/'schema-01'
STAR=np.arange(1,6,dtype='f')


class Inputs:
    def __init__(self,cohort):
        self.c=cohort
        rh=json.loads((replay.OUT/'manifest.json').read_text());rc=json.loads((cf.OUT/'manifest.json').read_text())
        assert ne.sha(replay.OUT/'effects.npy')==rh['output_sha256']
        assert ne.sha(cf.OUT/'probabilities.npy')==rc['probabilities_sha256']
        self.hn=np.load(replay.OUT/'effects.npy',mmap_mode='r');self.cf=np.load(cf.OUT/'probabilities.npy',mmap_mode='r')
        self.g=cov.movie_geometry();self.anchor_prior=cov.original()[-1]
        manifest=json.loads((extra.OUT/'manifest.json').read_text())
        models=list(cohort.ref['models'].values())+list(manifest['inputs'].values());sources=[]
        for meta in models:
            assert ne.sha(meta['vector'])==meta['vector_sha256']
            sources.append(np.memmap(meta['vector'],'<f4','r'))
        self.j=np.column_stack(sources);self.jmean=self.j.mean(1)
        self.count=np.log1p(np.diff(cohort.d['train_offsets'])).astype('f')
        self.ic=np.log1p(ne.history('trainx','item_cnts')).astype('f')
        self.movie=cohort.m
    def part(self,rows,mode):
        c=self.c;ar,masks=c.masks(rows);mask=masks[mode];n=mask.sum(1);div=n.clip(min=1)
        mu=self.jmean[rows];prior=self.anchor_prior[ar]
        u,m,d=c.u[rows],c.m[rows],c.day[rows]
        h=self.hn[rows];p=self.cf[rows,mode].astype('f');p0=self.cf[rows,0].astype('f');cm=p@STAR
        dt=d[:,None].astype('i')-c.day[ar];sim=np.einsum('nd,nad->na',self.g[m],self.g[c.m[ar]])
        err=c.rows['rating'][ar].astype('f')-prior
        def mean(v):return (v*mask).sum(1)/div
        norm=1+abs(dt)/20
        feats=np.column_stack((self.j[rows]-mu[:,None],mu,h[:,:,0]-mu[:,None],h[:,:,mode]-h[:,:,0],
            cm,p,p0,p@STAR**2-cm**2,mu-cm,cm-p0@STAR,self.count[u],self.ic[m],(d.astype('f')-2000)/500,n,
            mean(err),mean(err**2),mean(err/norm),mean(err*(dt==0)),mean(sim*err),mean(sim**2*err),
            mean(err*(prior-3.5)),mean(err*(mu[:,None]-prior)),mean(prior)-mu*(n>0),mean(sim),
            mean(1/norm),mean(dt==0),mean(c.rows['rating'][ar]==1),mean(c.rows['rating'][ar]==5),
            mean((self.hn[ar,:,0].mean(2)-prior)*err),mean(abs(err)),mean((err-mean(err)[:,None])**2))).astype('f')
        return feats,cm
    def data(self,rows,mode):
        width=self.part(rows[:1],mode)[0].shape[1]
        x=np.empty((len(rows),width),'f');ref=np.empty(len(rows),'f')
        for j in range(0,len(rows),8192):
            sl=slice(j,j+8192);x[sl],ref[sl]=self.part(rows[sl],mode)
        return x,ref


def upper(v,n,cf,cf0,b):
    e=v-n
    return np.column_stack((v-b,e-(cf-cf0),e,.5*(n-b))).astype(float)


def coefficients(x,err):
    A=x.T@x;A.flat[::len(A)+1]+=480
    L=np.linalg.cholesky(A);return lsq_linear(L.T,np.linalg.solve(L,x.T@err),bounds=([0,-.45,-.25,0],[.45,.45,.25,.3])).x


def main():
    OUT.mkdir(exist_ok=True)
    required=[v.OUT/'manifest.json' for v in (replay,cf,extra)]
    while not all(p.is_file() for p in required):sleep(15)
    y,f,_,_=anchor.history_control()
    rec=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    c=replay.Cohort();d=Inputs(c);np.testing.assert_array_equal(c.rows['rating'][c.ix],y)
    rng=np.random.default_rng(1487);choices=rng.choice([2,3,4,5,6,7],len(y))
    x,cm=d.data(c.ix,1);zero,cm0=d.data(c.ix,0);aug=np.empty_like(x);ac=np.empty_like(cm)
    for mode in np.unique(choices):
        sel=choices==mode;aug[sel],ac[sel]=d.data(c.ix[sel],mode)
    masked,mc=d.data(c.ix,2)
    raw=np.empty_like(cm);empty=raw.copy();hyp=raw.copy();models=[]
    for k in range(5):
        tr=f!=k;va=~tr
        net=CatBoostRegressor(iterations=640,depth=7,learning_rate=.057,l2_leaf_reg=70,
                random_seed=4731+k,verbose=False,allow_writing_files=False,thread_count=2)
        net.fit(np.concatenate((x[tr],aug[tr])),np.r_[(y-cm)[tr],(y-ac)[tr]])
        raw[va]=cm[va]+net.predict(x[va],thread_count=2);empty[va]=cm0[va]+net.predict(zero[va],thread_count=2)
        hyp[va]=mc[va]+net.predict(masked[va],thread_count=2);models.append(net);net.save_model(str(OUT/f'fold-{k}.cbm'))
        print('fold',k,'actual',ne.rmse(raw[va],y[va]),'other observed probe peers',ne.rmse(hyp[va],y[va]),flush=True)
    z=upper(raw,empty,cm,cm0,base);held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@coefficients(z[tr],(y-base)[tr])
    weights=coefficients(z,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f)
    ok=ret.qualifies(known,ctrl)
    report=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=weights.tolist(),
        actual_observation_raw=blend.metrics(raw,y,f),augmented_observation_raw=blend.metrics(hyp,y,f),
        zero_observation_raw=blend.metrics(empty,y,f),qualifying_hashes={},qualifying_labels_used_for_fitting=False,
        training_targets_all_excluded_from_cf_and_history_models=True,known_comparison_is_nested=False,
        inputs={str(p):ne.sha(p) for p in required})
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        del x,zero,aug,masked
        delta=np.empty(len(qb))
        for j in range(0,len(qb),32768):
            rows=np.arange(c.n+j,min(c.n+j+32768,len(c.rows)));sl=slice(j,j+len(rows))
            x,cm=d.data(rows,2);zero,cm0=d.data(rows,0)
            a=cm+sum(net.predict(x,thread_count=2) for net in models)/5
            b=cm0+sum(net.predict(zero,thread_count=2) for net in models)/5
            delta[sl]=upper(a,b,cm,cm0,qb[sl])@weights
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');report['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
