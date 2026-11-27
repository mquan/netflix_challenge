"""Fit error corrections from paired rating forecasts aligned to unchanged original-training predictions."""
import json
from pathlib import Path
import re
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_ridge as bank
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret

OUT=ne.ROOT/'own-aligned-forecasts-20261126'
PARENT=blend.OUT/'stage-24'


def reference(phase):
    rec=json.loads((joint.OUT/'manifest.json').read_text())
    ix=ret.map_rows() if phase=='probex' else slice(1408395,None)
    models=[]
    for m in rec['models'].values():
        assert ne.sha(m['vector'])==m['vector_sha256']
        models.append(np.memmap(m['vector'],'<f4','r')[ix])
    a=np.column_stack(models)
    u,i,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    # Shared history excludes the competition cohort in both roles.
    uc,ic=[ne.history('trainx',k) for k in ('user_cnts','item_cnts')]
    ctx=np.column_stack((a.mean(1)-3.5,a-a.mean(1)[:,None],np.log1p(uc[u])-5,
                    np.log1p(ic[i])-8,(d.astype('f')-1700)/500)).astype('f')
    return a.mean(1),np.column_stack((np.ones(len(u),'f'),ctx,ctx[:,0]**2,ctx[:,0]*ctx[:,-3],
                                    ctx[:,0]*ctx[:,-2],np.tanh(ctx[:,-1])))


def columns():
    keep={};excluded={}
    for key,m in bank.sources().items():
        name=key.rsplit('/',1)[-1]
        if re.match('vf[0-9]+_',name) or re.search('-(mf|nbr|bias|nsvd1)($|_)',name):
            excluded[key]='history statistic or decomposition term';continue
        vectors=[]
        for p,s in zip(m['paths'],m['sha256']):
            assert ne.sha(p)==s
            vectors.append(np.load(p,mmap_mode='r'))
        if any(abs(np.mean(v[::101])-3.5)>1 or not .25<np.std(v[::101])<1.6 for v in vectors):
            excluded[key]='outside the expected mean or variance of rating forecasts';continue
        keep[key]=m
    (OUT/'schema.json').write_text(json.dumps(dict(forecasts=keep,excluded=excluded),indent=2)+'\n')
    return keep


def prepare(phase,cols,base,axis=None):
    ref,ctx=reference(phase)
    vectors=[np.load(m['paths'][int(phase=='qual')],mmap_mode='r') for m in cols.values()]
    def raw(rows):
        return np.column_stack([v[rows] for v in vectors])-ref[rows,None]
    sample=np.arange(0,len(base),29)
    x=ctx[sample];y=raw(sample)
    beta=np.linalg.solve(x.T@x+np.eye(x.shape[1])*.25,x.T@y).astype('f')
    resid=y-x@beta
    if axis is None:
        sd=resid.std(0).clip(.045)
        z=resid/sd
        val,vec=np.linalg.eigh(z.T@z/len(z))
        axis=(sd,vec[:,-48:].copy())
    sd,vec=axis
    # Each role gets its own covariance calibration within the same shared principal subspace.
    z=(resid/sd)@vec;var,basis=np.linalg.eigh(z.T@z/len(z))
    whitening=(basis*(1/np.sqrt(var.clip(.12))))@basis.T
    proj=(vec@whitening)/sd[:,None]
    xout=np.empty((len(base),len(var)+ctx.shape[1]+3),'f')
    for j in range(0,len(base),8192):
        sl=slice(j,j+8192);r=raw(sl)-ctx[sl]@beta
        xout[sl]=np.column_stack((r@proj,ctx[sl],ref[sl]-base[sl],base[sl],r.std(1)))
    assert np.isfinite(xout).all()
    return xout,axis


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    cols=columns();print('forecast pairs',len(cols),flush=True)
    x,axis=prepare('probex',cols,base);held=np.empty(len(y),'f');models=[]
    for k in range(5):
        tr=f!=k
        model=CatBoostRegressor(iterations=480,depth=5,learning_rate=.045,l2_leaf_reg=100,
            thread_count=2,random_seed=1129+k,verbose=False,allow_writing_files=False)
        model.fit(x[tr],(y-base)[tr]);held[~tr]=model.predict(x[~tr],thread_count=2)
        model.save_model(str(OUT/f'fold-{k}.cbm'));models.append(model)
        print('fold',k,ne.rmse(base[~tr]+held[~tr],y[~tr]),flush=True)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[w for w in stats if ret.qualifies(stats[w],stats['0'])]
    key=min(ok,key=lambda w:stats[w]['pooled']) if ok else 'control'
    rep=dict(parent=str(PARENT),selected=key,known=stats,schema_sha256=ne.sha(OUT/'schema.json'),qualifying_hashes={},
        qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del x
        x,_=prepare('qual',cols,qb,axis)
        delta=sum(m.predict(x,thread_count=2) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
