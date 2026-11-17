#!/usr/bin/env python3
"""Conditional layer ablations of released inputs versus the fresh fourteen."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import numpy as np
import upstream_expansion as e
import upstream_repro as u
import upstream_exchange as x


def main():
    os.nice(10)
    labels,users,fresh=e.collect_expansion()
    manifest=u.read(u.BASE/'reference-manifest.json')
    if not np.array_equal(labels,np.load(u.REFERENCE/'data/probex/ratings.npy',allow_pickle=False)):
        raise ValueError('Reference/fresh label ordering differs')
    arrays=[]
    for entry in manifest['entries']:
        p=u.REFERENCE/entry['path']
        if u.sha(p)!=entry['sha256']:raise ValueError('Reference fixture changed')
        arrays.append(np.load(p,mmap_mode='r',allow_pickle=False))
    fresh_names=e.ARMS['all-fourteen']
    start=len(arrays)
    arrays += [fresh[name] for name in fresh_names]
    clip=manifest['clip']+[True]*len(fresh_names)
    groups=manifest['groups']
    base=sorted(set(i for group in ['tx','rbm','nn','other'] for i in groups[group]))
    all_reference=list(range(start))
    arms={'fresh-fourteen':list(range(start,len(arrays))),
          'released-base-bank':base,
          'released-base-plus-gbm':sorted(set(base+groups['gbmblend'])),
          'released-base-plus-mlp':sorted(set(base+groups['mlpblend'])),
          'released-base-plus-fwls':sorted(set(base+groups['fwlsblend'])),
          'released-all':all_reference}
    for group in ['gbmblend','mlpblend','fwlsblend']:
        arms['released-without-'+group]=[i for i in all_reference if i not in groups[group]]
    folds=x.user_folds(users)
    width=len(arrays)+1
    grams=np.zeros((5,width,width))
    rhs=np.zeros((5,width))
    def block(lo,hi):
        matrix=np.column_stack([a[lo:hi] for a in arrays]).astype(np.float64)
        for j,should_clip in enumerate(clip):
            if should_clip:matrix[:,j]=np.clip(matrix[:,j],0,6)
        return np.column_stack([matrix,np.ones(len(matrix))])
    for lo in range(0,len(labels),8192):
        hi=min(lo+8192,len(labels)); z=block(lo,hi)
        for fold in range(5):
            m=folds[lo:hi]==fold
            a=z[m]
            grams[fold]+=a.T@a
            rhs[fold]+=a.T@labels[lo:hi][m]
    total=grams.sum(axis=0); target=rhs.sum(axis=0)
    weights={}
    for name,columns in arms.items():
        ix=columns+[len(arrays)]
        penalty=np.eye(len(ix))*1000;penalty[-1,-1]=0
        weights[name]=[np.linalg.solve(total[np.ix_(ix,ix)]+penalty,target[ix])]
        weights[name]+=[np.linalg.solve((total-grams[f])[np.ix_(ix,ix)]+penalty,(target-rhs[f])[ix]) for f in range(5)]
    sse={name:np.zeros(6) for name in arms}
    for lo in range(0,len(labels),8192):
        hi=min(lo+8192,len(labels)); z=block(lo,hi)
        for name,columns in arms.items():
            a=z[:,columns+[len(arrays)]]
            value=np.clip(a@weights[name][0],1.05,4.95)
            sse[name][0]+=np.sum((value-labels[lo:hi])**2)
            for f in range(5):
                m=folds[lo:hi]==f
                value=np.clip(a[m]@weights[name][f+1],1.05,4.95)
                sse[name][f+1]+=np.sum((value-labels[lo:hi][m])**2)
    records=[]
    for name,columns in arms.items():
        records.append(dict(arm=name,columns=len(columns),fitted_rmse=float(np.sqrt(sse[name][0]/len(labels))),
                       final_layer_user_cv_rmse=float(np.sqrt(sse[name][1:].sum()/len(labels))),
                       fold_rmse=[float(np.sqrt(sse[name][f+1]/np.sum(folds==f))) for f in range(5)]))
    scored={r['arm']:r for r in records}
    assert abs(scored['released-all']['fitted_rmse']-0.8561671559335715)<1e-6
    assert abs(scored['fresh-fourteen']['final_layer_user_cv_rmse']-0.8677119362283443)<1e-6
    result=dict(rows=len(labels),source_commit=u.PIN,records=records,
                scope='Diagnostic only: released arrays and their prelearned intermediate blends; only final ridge is user-fold cross-fitted. This is not end-to-end held-out performance or fresh model training.',
                scientific_use='Attribute missing-bank and blending-layer gaps before committing more base-model CPU time')
    u.write(u.ROOT/'experiments/upstream-expansion/gap-audit.json',result)
    import json
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
