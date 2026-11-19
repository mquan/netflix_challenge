#!/usr/bin/env python3
"""Independently replay temporal93 grouped ridge diagnostics on known rows."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_temporal93 as experiment

u,n=experiment.u,experiment.n


def verify():
    manifest=u.read(experiment.REPORT/'manifest.json')
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=experiment.RUN,experiment.REPORT,manifest['config'],experiment.code_hashes
    n.verify_prepared()
    z=np.load(experiment.RUN/'design.npy',mmap_mode='r');y=np.load(experiment.RUN/'labels.npy')
    users=np.load(experiment.RUN/'users.npy');groups=np.load(experiment.RUN/'groups.npy');folds=groups//2
    _,first,inverse=np.unique(users,return_index=True,return_inverse=True)
    if not np.array_equal(folds,folds[first][inverse]):raise ValueError('A user crosses outer folds')
    gram,rhs=n.grouped_stats(z,y,groups)
    with np.load(experiment.RUN/'stats.npz') as saved:
        if not np.allclose(gram,saved['gram'],rtol=1e-12,atol=1e-6) or not np.allclose(rhs,saved['rhs'],rtol=1e-12,atol=1e-6):
            raise ValueError('Grouped sufficient statistics do not reproduce')
    records=[];screen=experiment.REPORT/'ridge-screen.json';published=u.read(screen)
    for arm,indices in [('control',list(range(166))+[168]),('main',list(range(169)))]:
        g=gram[:,indices][:,:,indices];r=rhs[:,indices];penalty=np.eye(len(indices))*manifest['config']['ridge_lambda'];penalty[-1,-1]=0
        prediction=np.empty(len(y))
        for fold in range(5):
            eligible=np.arange(10)//2!=fold
            weights=np.linalg.solve(g[eligible].sum(0)+penalty,r[eligible].sum(0))
            for lo in range(0,len(y),8192):
                hi=min(lo+8192,len(y));held=folds[lo:hi]==fold
                prediction[lo:hi][held]=np.clip(z[lo:hi][held][:,indices]@weights,1.05,4.95)
        path=experiment.RUN/(arm+'-ridge.npy');stored=np.load(path)
        error=float(np.max(np.abs(prediction-stored)))
        if error>1e-10:raise ValueError('Independent held-user ridge replay differs')
        rmse=math.sqrt(math.fsum(float(p-target)**2 for p,target in zip(stored,y))/len(y))
        row=next(r for r in published['records'] if r['arm']==arm)
        if abs(rmse-row['rmse'])>1e-12 or u.sha(path)!=row['prediction_sha256']:
            raise ValueError('Independent ridge score binding differs')
        records.append(dict(arm=arm,independent_rmse=rmse,max_prediction_replay_error=error,prediction_sha256=u.sha(path)))
    result=dict(state='complete_independent_grouped_ridge_replay',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        verifier_sha256=u.sha(Path(__file__)),manifest_sha256=u.sha(experiment.REPORT/'manifest.json'),screen_sha256=u.sha(screen),
        rows=len(y),users=len(first),recomputed_group_stats=True,outer_users_disjoint=True,
        records=records,replay_absolute_tolerance=1e-10,qualifying_labels_read=False,nonlinear_fitting_performed=False,
        scope='Separate sufficient-statistic recomputation, held-user ridge solve/prediction replay and compensated scoring. Only known-ProbeX labels. Diagnostic improvement is not a nonlinear or qualifying promotion.')
    u.write(experiment.REPORT/'ridge-verification.json',result);print(result)


if __name__=='__main__':verify()
