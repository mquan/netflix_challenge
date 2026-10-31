#!/usr/bin/env python3
"""Independent H8 validation with fold 0 excluded throughout the stack.

Reuse the original choice features and fixed tree settings. Round selection
uses only folds 2-4/1; fixed-round final refits use folds 1-4. No source
trainer, original H8 output, GPU job or original-probe label is changed.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold, load
from headroom_choices import features, read_joint_factors
from local_context_chain import paired
from train_holdout import digest


def stacked_bases(bank, y, folds, penalty=10.):
    bank, y, folds = np.asarray(bank), np.asarray(y), np.asarray(folds)
    if bank.ndim != 2 or len(bank) != len(y) or folds.shape != y.shape:
        raise ValueError('stack shapes differ')
    if not all(np.any(folds == f) for f in range(5)):
        raise ValueError('all five user folds required')
    x = np.column_stack([np.ones(len(y)), bank])
    def predict(fit, query):
        w = np.linalg.solve(x[fit].T@x[fit]+penalty*np.eye(x.shape[1]),x[fit].T@y[fit])
        return x[query]@w
    inner = np.full(len(y),np.nan)
    for f in (2,3,4):
        query = folds == f
        inner[query] = predict((folds >= 2)&~query,query)
    inner[folds == 1] = predict(folds >= 2,folds == 1)
    outer = np.empty(len(y))
    for f in (1,2,3,4):
        query = folds == f
        outer[query] = predict((folds >= 1)&~query,query)
    outer[folds == 0] = predict(folds >= 1,folds == 0)
    return inner,outer


def fit_outer(bank, choices, y, folds, params, rounds=2000, patience=100):
    if not all(np.isfinite(v).all() for v in (bank,choices,y)):
        raise ValueError('nonfinite stack input')
    inner,outer = stacked_bases(bank,y,folds)
    fit,stop,train,held = folds >= 2,folds == 1,folds >= 1,folds == 0
    predictions = {'base':outer[held]}
    iterations,importance = {},{}
    for name,extra in [('base_tree',choices[:,:0]),('choice_tree',choices)]:
        xi = np.column_stack([inner,extra])
        candidate = lgb.train(params,lgb.Dataset(xi[fit],y[fit]-inner[fit]),rounds,
            valid_sets=[lgb.Dataset(xi[stop],y[stop]-inner[stop])],
            callbacks=[lgb.early_stopping(patience,verbose=False)])
        selected = int(candidate.best_iteration)
        if selected < 1:
            raise ValueError('no selected tree iteration')
        xo = np.column_stack([outer,extra])
        final = lgb.train(params,lgb.Dataset(xo[train],y[train]-outer[train]),selected)
        predictions[name] = outer[held]+final.predict(xo[held],num_iteration=selected)
        iterations[name] = selected
        importance[name] = final.feature_importance('gain').tolist()
    return predictions,iterations,importance


def run(plan_path, output):
    plan_path,output = Path(plan_path),Path(output)
    p = json.loads(plan_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    required = [p['data'],p['joint_model'],*p['bank']]
    required += [str(Path(__file__).resolve().with_name(name)) for name in
        ('choice_outer_validation.py','headroom_choices.py','headroom_map.py',
         'gbdt_blend.py','local_context_chain.py','train_holdout.py')]
    if not set(required) <= set(p['frozen']):
        raise ValueError('missing frozen stack input')
    def verify():
        for path,expected in p['frozen'].items():
            if digest(path) != expected:
                raise ValueError('frozen input changed: '+path)
    verify()
    if p['libraries'] != dict(numpy=np.__version__,lightgbm=lgb.__version__):
        raise ValueError('stack library version changed')
    d = load(p['data'])
    bank = np.column_stack([np.fromfile(path,dtype='<f4').astype(float) for path in p['bank']])
    y,users = d['probe']['rating'].astype(float),d['probe']['user']
    folds = blend_fold(users)
    if len(bank) != len(y):
        raise ValueError('bank does not match P2')
    q,emb = read_joint_factors(p['joint_model'],d['items'])
    choices,names = features(d,q,emb)
    if names != p['choice_feature_names']:
        raise ValueError('choice features differ from fixed protocol')
    pred,iterations,importance = fit_outer(bank,choices,y,folds,p['tree_params'],p['rounds'],p['patience'])
    held = folds == 0
    raw = paired(y[held],pred['base'],pred['choice_tree'],users[held])
    matched = paired(y[held],pred['base_tree'],pred['choice_tree'],users[held])
    verify()
    output.mkdir(parents=True)
    np.savez(output/'predictions.npz',rows=np.flatnonzero(held),**pred)
    report = dict(plan_sha256=digest(plan_path),rows=len(y),held_rows=int(held.sum()),
        selected_iterations=iterations,choices_vs_base=raw,choices_vs_matched_tree=matched,
        passes_original_p2_threshold=raw['delta_rmse']<=-p['minimum_gain'],
        choice_feature_names=names,importance=importance,prediction_sha256=digest(output/'predictions.npz'),
        metric='Independent nested H8 check, P2 fold 0. Fold 0 labels excluded from all ridge and tree fits and selection. No full-bank promotion.',
        difference_from_H8='Proper inner/outer stacking plus final tree refit on folds 1-4; features and tree settings fixed to H8. Not a one-variable numerical reproduction of its original nonnested fit.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run(args.plan,args.output)
