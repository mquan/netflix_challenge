"""Fit four ordinal cumulative-logit corrections initialized at the frozen ensemble mean."""
import json
import numpy as np
from scipy.special import expit,logit
from catboost import CatBoostClassifier,Pool
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret

OUT=ne.ROOT/'own-ordinal-boost-20261126'


def baseline(prob,mu):
    # One common tilt of the four cumulative logits exactly matches the ensemble mean.
    cumulative=np.cumsum(prob[:,::-1],axis=1)[:,:4][:,::-1]
    logits=logit(cumulative.clip(1e-7,1-1e-7))
    lo=np.full(len(mu),-24.,'f');hi=-lo
    for k in range(20):
        mid=(lo+hi)*.5
        up=(1+expit(logits+mid[:,None]).sum(1))<mu
        lo=np.where(up,mid,lo);hi=np.where(up,hi,mid)
    return logits+((lo+hi)*.5)[:,None]


def data(phase,p,q,base,keys):
    raw=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r');known=phase=='probex'
    x=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
    assert ne.sha(full.OUT/(phase+'.npy'))==json.loads((full.OUT/(phase+'-sources.json')).read_text())['feature_sha256']
    parts=[x[:,::8],raw[:,7:37:2],raw[:,37:]]
    dirs=[blend.direction(k,p,q,known)[0].reshape(len(base),-1) for k in keys]
    for name,stride in [('kernel-family',4),('session-pairs',6),('user-gp',2)]:
        v=np.load(ne.ROOT/('own-'+name+'-20261126')/(phase+'.npy'),mmap_mode='r')
        parts.append(v[:,::stride])
    features=np.empty((len(base),sum(t.shape[1] for t in parts+dirs)+1),'f')
    logits=np.empty((len(base),4),'f')
    for j in range(0,len(base),16384):
        sl=slice(j,j+16384)
        prob=raw[sl,7:12];np.testing.assert_allclose(prob.sum(1),1.,atol=2e-5)
        logits[sl]=baseline(prob,base[sl])
        features[sl]=np.column_stack((base[sl],*[v[sl] for v in dirs],*[v[sl] for v in parts]))
    err=np.max(abs(1+expit(logits).sum(1)-base));assert err<.00006,err
    return features,logits


def main():
    OUT.mkdir(exist_ok=True);y,f,p,q=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x,prior=data('probex',p,q,base,fr['columns'])
    held=np.zeros((len(y),4),'f')
    for k in range(4):
        binary=(y>k+1).astype('i')
        for fold in range(5):
            tr=f!=fold;va=~tr
            model=CatBoostClassifier(iterations=520,depth=5,learning_rate=.04,l2_leaf_reg=90,thread_count=2,
                bootstrap_type='Bernoulli',subsample=.82,rsm=.78,border_count=72,random_seed=2291+fold*4+k,
                allow_writing_files=False,verbose=False)
            model.fit(Pool(x[tr],label=binary[tr],baseline=prior[tr,k]))
            model.save_model(str(OUT/f'cut-{k}-fold-{fold}.cbm'))
            raw=model.predict(x[va],prediction_type='RawFormulaVal',thread_count=2)
            if fold==0:
                ix=np.flatnonzero(va)[:512]
                check=model.predict(Pool(x[ix],baseline=prior[ix,k]),prediction_type='RawFormulaVal',thread_count=2)
                np.testing.assert_allclose(check,raw[:512]+prior[ix,k],atol=1e-5)
            held[va,k]=expit(prior[va,k]+raw)-expit(prior[va,k])
            print('threshold',k+1,'fold',fold,flush=True)
    # A shared mean shrinkage and independent threshold reliability both have held-user predictions.
    from own_empirical_neighborhood import fit
    pred=np.empty(len(y))
    for j in range(5):
        tr=f!=j;pred[~tr]=held[~tr]@fit(held[tr].astype(float),(y-base)[tr])
    w=fit(held.astype(float),y-base)
    ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+pred),y,f);take=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),known=known,control=ctrl,weights=w.tolist(),selected='candidate' if take else 'control',
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if take:
        del x,prior
        qx,qprior=data('qual',p,q,qb,fr['columns'])
        delta=np.zeros(len(qb))
        for k in range(4):
            for fold in range(5):
                model=CatBoostClassifier().load_model(str(OUT/f'cut-{k}-fold-{fold}.cbm'))
                for j in range(0,len(qb),65536):
                    sl=slice(j,j+65536);a=qprior[sl,k]
                    raw=model.predict(qx[sl],prediction_type='RawFormulaVal',thread_count=2)
                    delta[sl]+=(expit(a+raw)-expit(a))*(w[k]/5)
        np.save(OUT/'oof.npy',pred);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
