#!/usr/bin/env python3
"""Independent five-fold ridge solve for the fixed completed-output additions."""
import math
from pathlib import Path
import numpy as np
import upstream_ready_public_schema as experiment

u=experiment.u


def main():
    report=experiment.REPORT;run=experiment.RUN
    result=u.read(report/'ridge-screen.json');plan=u.read(report/'plan.json')
    if result['script_sha256']!=u.sha(Path(experiment.__file__)):
        raise ValueError('Screen implementation changed')
    for name,digest in result['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Changed input: '+name)
    parent=experiment.temporal.RUN
    z=np.load(parent/'design.npy',mmap_mode='r');y=np.load(parent/'labels.npy')
    users=np.load(parent/'users.npy');groups=np.load(parent/'groups.npy');folds=groups//2
    _,first,inverse=np.unique(users,return_index=True,return_inverse=True)
    if not np.array_equal(folds,folds[first][inverse]):raise ValueError('Users cross folds')
    selected=[r for r in plan['ready_unused'] if r['duplicate_of'] is None]
    if [r['name'] for r in selected]!=result['fixed_additions']:raise ValueError('Selection changed')
    extra=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in selected]
    width=z.shape[1]+len(extra)
    gram=np.zeros((5,width,width));rhs=np.zeros((5,width))
    def block(lo,hi):
        a=np.empty((hi-lo,width),dtype=np.float64)
        a[:,:z.shape[1]-1]=z[lo:hi,:-1]
        for j,values in enumerate(extra):a[:,z.shape[1]-1+j]=np.clip(values[lo:hi],0,6)
        a[:,-1]=z[lo:hi,-1]
        return a
    for lo in range(0,len(y),4096):
        hi=min(lo+4096,len(y));a=block(lo,hi)
        for fold in range(5):
            keep=folds[lo:hi]==fold;b=a[keep]
            gram[fold]+=b.T@b;rhs[fold]+=b.T@y[lo:hi][keep]
    records=[]
    for arm,indices in [('control',list(range(z.shape[1]-1))+[width-1]),('main',list(range(width)))]:
        g=gram[:,indices][:,:,indices];r=rhs[:,indices]
        penalty=np.eye(len(indices))*1000;penalty[-1,-1]=0
        weights=[np.linalg.solve(g[np.arange(5)!=f].sum(0)+penalty,r[np.arange(5)!=f].sum(0)) for f in range(5)]
        prediction=np.empty(len(y))
        for lo in range(0,len(y),4096):
            hi=min(lo+4096,len(y));a=block(lo,hi)[:,indices]
            for f in range(5):
                held=folds[lo:hi]==f;prediction[lo:hi][held]=np.clip(a[held]@weights[f],1.05,4.95)
        path=run/(arm+'-ridge.npy');stored=np.load(path)
        error=float(np.max(np.abs(prediction-stored)))
        score=math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(stored,y))/len(y))
        published=next(r for r in result['records'] if r['arm']==arm)
        if error>1e-10 or abs(score-published['rmse'])>1e-12 or u.sha(path)!=published['prediction_sha256']:
            raise ValueError('Independent ridge replay differs')
        records.append(dict(arm=arm,rmse=score,max_prediction_error=error,prediction_sha256=u.sha(path)))
    proof=dict(state='complete',verifier_sha256=u.sha(Path(__file__)),screen_sha256=u.sha(report/'ridge-screen.json'),
        plan_sha256=u.sha(report/'plan.json'),records=records,users_disjoint=True,rows=len(y),
        independently_recomputed_five_fold_stats=True,qualifying_labels_read=False,
        public_prediction_values_read=False,scope='Known-ProbeX ridge diagnostic only; no qualifying or nonlinear promotion.')
    u.write(report/'ridge-verification.json',proof);print(proof)


if __name__=='__main__':main()
