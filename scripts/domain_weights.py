#!/usr/bin/env python3
"""Diagnose covariate shift using metadata and unknown target identifiers only.

A balanced logistic domain classifier estimates target/source density ratios.
Rating outcomes, prediction errors and recovered labels are never loaded.
Weights are clipped to [0.5,2] before mean normalization to limit variance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from streaming_blend import digest
from qual_matrix import verified_matrix


def fit_classifier(x, y, penalty=.001):
    def objective(beta):
        z=x@beta[:-1]+beta[-1]
        value=np.mean(np.logaddexp(0,z)-y*z)+penalty*np.dot(beta[:-1],beta[:-1])/2
        error=(expit(z)-y)/len(y)
        gradient=np.r_[x.T@error+penalty*beta[:-1],error.sum()]
        return value,gradient
    result=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=200,ftol=1e-10))
    if not result.success:raise ValueError('domain classifier did not converge: '+result.message)
    return result.x


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--qual-bank',required=True)
    ap.add_argument('--qual',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();output=Path(a.output)
    if output.with_suffix('.json').exists():raise ValueError('output exists')
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    qb=json.loads(Path(a.qual_bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['qualifying_labels_read'] is not False or q['bank_sha256']!=digest(a.qual_bank+'.inputs.json'):
        raise ValueError('unverified unlabeled target bank')
    if b['names'][-18:]!=qb['names'][-18:]:raise ValueError('metadata schemas differ')
    raw=np.load(a.bank+'.x.npy',mmap_mode='r');target=verified_matrix(a.qual,q)
    # The first11 generic names have a fixed schema in gbdt_blend.meta_features.
    names=['log_user_n','log_item_n','log_implicit_n','user_mean','user_std',
           'item_mean','day','day_minus_user_mean','day_minus_first','user_span',
           'log_same_day']+b['extra_meta_names']
    if len(names)!=18:raise ValueError('unsupported metadata schema')
    selected=[i for i,n in enumerate(names) if n!='log_unrated_n']
    source_meta=np.asarray(raw[:,-18:])[:,selected].astype(np.float64)
    target_meta=np.asarray(target[:,-18:])[:,selected].astype(np.float64)
    if not np.isfinite(source_meta).all() or not np.isfinite(target_meta).all():
        raise ValueError('nonfinite metadata')
    rng=np.random.default_rng(20261115)
    count=min(150000,len(raw)//3,len(target)//3)
    s=rng.choice(len(raw),2*count,replace=False);t=rng.choice(len(target),2*count,replace=False)
    train=np.r_[source_meta[s[:count]],target_meta[t[:count]]]
    y=np.r_[np.zeros(count),np.ones(count)]
    low,high=np.quantile(train,[.001,.999],axis=0)
    mean=np.clip(train,low,high).mean(0);std=np.maximum(np.clip(train,low,high).std(0),1e-6)
    train=(np.clip(train,low,high)-mean)/std
    beta=fit_classifier(train,y)
    held=np.r_[source_meta[s[count:]],target_meta[t[count:]]]
    score=((np.clip(held,low,high)-mean)/std)@beta[:-1]+beta[-1]
    auc=float((rankdata(score)[count:].sum()-count*(count+1)/2)/(count*count))
    ratio=np.exp(np.clip(((np.clip(source_meta,low,high)-mean)/std)@beta[:-1]+beta[-1],-20,20))
    weights=np.clip(ratio,.5,2);weights/=weights.mean()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.save(str(output)+'.npy',weights)
    report=dict(state='complete',kind='unlabeled_domain_density_ratio',
        qualifying_labels_read=False,rating_outcomes_loaded=False,
        metadata_features=[names[i] for i in selected],held_domain_auc=auc,
        source_rows=len(raw),target_rows=len(target),balanced_training_rows=2*count,
        effective_sample_fraction=float(weights.sum()**2/(np.dot(weights,weights)*len(weights))),
        weight_quantiles=np.quantile(weights,[0,.01,.1,.5,.9,.99,1]).tolist(),
        coefficients=dict(zip([names[i] for i in selected],beta[:-1].tolist())),
        intercept=float(beta[-1]),low=low.tolist(),high=high.tolist(),mean=mean.tolist(),std=std.tolist(),
        source_bank_sha256=digest(a.bank+'.inputs.json'),
        target_bank_sha256=digest(a.qual_bank+'.inputs.json'),
        target_manifest_sha256=digest(a.qual+'.json'),weights_sha256=digest(str(output)+'.npy'),
        limitation='Domain AUC diagnoses observable covariate shift; it does not prove conditional label invariance or RMSE gain')
    Path(str(output)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
