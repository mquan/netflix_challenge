#!/usr/bin/env python3
"""Matched known-user-fold ridge comparison of cached and paired realizations."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_nonlinear as n
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-paired-realization-diagnostic'

def run():
    parent=u.ROOT/'artifacts/upstream-latent90-features';paired=u.ROOT/'artifacts/upstream-rbm50-bank'
    y=np.load(parent/'labels.npy');groups=np.load(parent/'groups.npy');users=np.load(parent/'users.npy')
    if not np.array_equal(y,np.load(paired/'labels.npy')) or not np.array_equal(users,np.load(paired/'users.npy')):
        raise ValueError('Exact known row identities required')
    records=[];sources={}
    for name,path in [('cached',parent/'design.npy'),('paired',paired/'probex-design.npy')]:
        z=np.load(path,mmap_mode='r');sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        if z.shape!=(1128164,183):raise ValueError('Same raw107 ridge schema required')
        grams,rhs=n.grouped_stats(z,y,groups);pred=np.empty(len(y))
        for f in range(5):
            weights=n.ridge_weights(grams,rhs,np.arange(10)//2!=f,1000);held=groups//2==f
            pred[held]=np.clip(z[held]@weights,1.05,4.95)
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(pred,y))/len(y))
        records.append(dict(realization=name,rmse=score,fold_rmse=[n.x.metric(pred[groups//2==f],y[groups//2==f]) for f in range(5)]))
    for p in [parent/'labels.npy',parent/'groups.npy',parent/'users.npy',paired/'labels.npy',paired/'users.npy',Path(__file__),Path(n.__file__)]:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    result=dict(state='complete_matched_known_ridge_diagnostic',observed=dt.datetime.now(dt.timezone.utc).isoformat(),rows=len(y),
        records=records,paired_minus_cached_rmse=records[1]['rmse']-records[0]['rmse'],source_sha256=sources,
        qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Same known rows/user folds, raw107 ridge schema and penalty on two previously disclosed own source realizations. Diagnostic only, not a new model, neural comparison, qualifying score or promotion.')
    u.write(REPORT/'ridge.json',result);print(result,flush=True)

if __name__=='__main__':run()
