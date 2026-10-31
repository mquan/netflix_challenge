#!/usr/bin/env python3
"""H9: fixed unlabelled-choice SVD features beyond the validated H8 control.

No rating value enters the embedding producer. All label-dependent stacking,
selection and fitting exclude outer fold 0 until its one final evaluation.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import sparse

from choice_outer_validation import stacked_bases
from gbdt_blend import blend_fold,load
from headroom_choices import features,read_joint_factors,unlabelled_choices
from local_context_chain import paired
from train_holdout import digest


def choice_matrix(d):
    users,items,_=unlabelled_choices(d)
    matrix=sparse.csr_matrix((np.ones(len(users)),(users,items-1)),shape=(d['users'],d['items']))
    matrix.sum_duplicates()
    matrix.data[:]=1. # Presence, not duplicate-event frequency.
    df=np.bincount(matrix.indices,minlength=d['items'])
    idf=np.log1p(d['users']/(1.+df))
    matrix=matrix.multiply(idf).tocsr()
    lengths=np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix=matrix.multiply(1./np.maximum(lengths,1e-12)[:,None]).tocsr()
    return matrix


def embeddings(matrix,rank=32,oversample=8,power=5,seed=20261031):
    if rank<1 or oversample<0 or power<0 or rank>min(matrix.shape):
        raise ValueError('invalid fixed SVD dimensions')
    width=min(rank+oversample,min(matrix.shape))
    rng=np.random.default_rng(seed)
    omega=rng.standard_normal((matrix.shape[1],width))
    q,_=np.linalg.qr(matrix@omega,mode='reduced')
    for _ in range(power):
        z,_=np.linalg.qr(matrix.T@q,mode='reduced')
        q,_=np.linalg.qr(matrix@z,mode='reduced')
    small=np.asarray(matrix.T@q).T
    _,s,vt=np.linalg.svd(small,full_matrices=False)
    raw=vt[:rank].T*np.sqrt(s[:rank])
    unit=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
    if not np.isfinite(raw).all() or not np.isfinite(unit).all():
        raise ValueError('nonfinite choice embedding')
    return raw,unit,s[:rank]


def fit_pair(bank,control_features,extra,y,folds,params,rounds=2000,patience=100):
    inner,outer=stacked_bases(bank,y,folds)
    fit,stop,train,held=folds>=2,folds==1,folds>=1,folds==0
    predictions,iterations={},{}
    for name,values in [('h8',control_features),('h8_svd',np.column_stack([control_features,extra]))]:
        xi=np.column_stack([inner,values])
        model=lgb.train(params,lgb.Dataset(xi[fit],y[fit]-inner[fit]),rounds,
            valid_sets=[lgb.Dataset(xi[stop],y[stop]-inner[stop])],
            callbacks=[lgb.early_stopping(patience,verbose=False)])
        n=int(model.best_iteration)
        xo=np.column_stack([outer,values])
        final=lgb.train(params,lgb.Dataset(xo[train],y[train]-outer[train]),n)
        predictions[name]=outer[held]+final.predict(xo[held],num_iteration=n,
                                                   num_threads=params['num_threads'])
        iterations[name]=n
    return predictions,iterations


def run(plan_path,output):
    plan_path,output=Path(plan_path),Path(output)
    p=json.loads(plan_path.read_text())
    if output.exists():raise FileExistsError(output)
    required=[p['data'],p['joint_model'],p['h8_predictions'],p['h8_report'],*p['bank']]
    required += [str(Path(__file__).resolve().with_name(name)) for name in
        ('choice_cooccurrence.py','choice_outer_validation.py','headroom_choices.py',
         'gbdt_blend.py','headroom_map.py','local_context_chain.py','train_holdout.py')]
    if not set(required)<=set(p['frozen']):raise ValueError('missing frozen H9 input')
    def verify():
        for path,h in p['frozen'].items():
            if digest(path)!=h:raise ValueError('frozen H9 input changed: '+path)
    verify()
    import scipy
    if p['libraries']!=dict(numpy=np.__version__,lightgbm=lgb.__version__,scipy=scipy.__version__):
        raise ValueError('H9 library version differs')
    d=load(p['data']);y=d['probe']['rating'].astype(float);users=d['probe']['user']
    folds=blend_fold(users);held=folds==0
    bank=np.column_stack([np.fromfile(path,dtype='<f4').astype(float) for path in p['bank']])
    matrix=choice_matrix(d)
    print(json.dumps(dict(stage='choice_matrix',rows=matrix.shape[0],items=matrix.shape[1],pairs=matrix.nnz)),flush=True)
    raw,unit,singular=embeddings(matrix,**p['svd'])
    del matrix
    print(json.dumps(dict(stage='choice_embedding_complete',rank=raw.shape[1])),flush=True)
    # Only identifiers/dates are passed to either feature producer.
    identities=np.empty(len(y),dtype=[('user','<u4'),('item','<u2'),('day','<u2')])
    for key in identities.dtype.names:identities[key]=d['probe'][key]
    view=dict(d,probe=identities)
    q,implicit=read_joint_factors(p['joint_model'],d['items'])
    control,names=features(view,q,implicit)
    extra,extra_names=features(view,raw,unit)
    keep=np.array([not name.startswith('n_') for name in extra_names])
    extra=extra[:,keep];extra_names=[name for name,k in zip(extra_names,keep) if k]
    if extra_names!=p['extra_affinity_names']:raise ValueError('H9 affinity schema changed')
    pred,iterations=fit_pair(bank,control,extra,y,folds,p['tree_params'],p['rounds'],p['patience'])
    prior=np.load(p['h8_predictions'])
    if not np.array_equal(prior['rows'],np.flatnonzero(held)):
        raise ValueError('H8 replay rows differ')
    replay_error=float(np.max(np.abs(prior['choice_tree']-pred['h8'])))
    if replay_error>1e-8:raise ValueError('H8 fixed control did not reproduce')
    stats=paired(y[held],pred['h8'],pred['h8_svd'],users[held])
    verify();output.mkdir(parents=True)
    np.savez(output/'embedding.npz',raw=raw,unit=unit,singular=singular)
    np.savez(output/'predictions.npz',rows=np.flatnonzero(held),**pred)
    report=dict(plan_sha256=digest(plan_path),comparison=stats,selected_iterations=iterations,
        h8_replay_max_abs_error=replay_error,passes_p2_gate=stats['delta_rmse']<=-p['minimum_gain'],
        feature_names=names,extra_feature_names=['choice_svd_'+x for x in extra_names],
        files={name:digest(output/name) for name in ('embedding.npz','predictions.npz')},
        scope='One fixed nested P2 fold-0 comparison beyond H8. No original-probe score, GPU use or full-bank promotion.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.plan,args.output)
