#!/usr/bin/env python3
"""Nested tree ablations on the frozen union bank; no stacked CV inputs."""
import argparse
import hashlib
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
from gbdt_blend import load, blend_fold, meta_features
from moonshot_nn_blend import contexts, design, ridge_fit


def fit_fold(x, y, users, fold, *, rounds=3000, residual_design=None):
    outer = blend_fold(users)
    train, held = outer != fold, outer == fold
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    fit, valid = train & ~inner, train & inner
    base = np.zeros(len(y))
    if residual_design is not None:
        w = ridge_fit(residual_design[fit], y[fit])
        base = 3.6 + residual_design@w
    params = dict(objective='regression', learning_rate=.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=.8, bagging_fraction=.8,
                  bagging_freq=1, lambda_l2=10, num_threads=1, verbose=-1, seed=2026)
    dt = lgb.Dataset(x[fit], y[fit]-base[fit])
    dv = lgb.Dataset(x[valid], y[valid]-base[valid], reference=dt)
    model = lgb.train(params, dt, rounds, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    chosen = model.best_iteration
    selected = base[held] + model.predict(x[held], num_threads=1)
    if residual_design is not None:
        w = ridge_fit(residual_design[train], y[train])
        base = 3.6 + residual_design@w
    model = lgb.train(params, lgb.Dataset(x[train], y[train]-base[train]), chosen)
    refit = base[held] + model.predict(x[held], num_threads=1)
    return selected, refit, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--residual', action='store_true')
    ap.add_argument('--rounds', type=int, default=3000)
    a = ap.parse_args()
    if pathlib.Path(a.output+'.json').exists():
        ap.error('output already exists; use a new experiment name')
    bank = json.loads(pathlib.Path('experiments/UNION-round-manifest.json').read_text())['reports']['union']['inputs']
    d = load(a.data); y = d['probe']['rating'].astype(np.float64)
    cols, hashes = [], dict(bank)
    for name, expected in bank.items():
        raw = pathlib.Path(name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected: raise ValueError(name)
        cols.append(np.frombuffer(raw, dtype='<f4'))
    predictions = np.column_stack(cols)
    support = json.loads(pathlib.Path('experiments/BOOST-support-inputs.json').read_text())
    for name, expected in support.items():
        raw = pathlib.Path(name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected: raise ValueError(name)
        cols.append(np.frombuffer(raw, dtype='<f4'))
    hashes.update(support)
    bank_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    # Metadata depends only on train ratings and unlabeled interaction dates.
    cache = pathlib.Path('artifacts/boost-union-features.npz')
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['bank_hash']) != bank_hash:
                raise ValueError('cached feature bank mismatch')
            x = saved['x']; context = saved['context']
    else:
        for k, v in meta_features(d).items():
            cols.append(v)
        x = np.column_stack(cols).astype(np.float32)
        context = contexts(d)
        np.savez(cache, x=x, context=context, bank_hash=bank_hash)
    rx = design(predictions, context)[0] if a.residual else None
    user = d['probe']['user']; folds = blend_fold(user)
    selected, refit = np.empty(len(y)), np.empty(len(y)); records = []
    print(json.dumps(dict(config=vars(a), columns=x.shape[1], inputs=hashes)), flush=True)
    for f in range(5):
        start = time.monotonic(); held = folds == f
        selected[held], refit[held], chosen = fit_fold(x, y, user, f, rounds=a.rounds, residual_design=rx)
        rec = dict(fold=f, chosen_rounds=chosen, seconds=time.monotonic()-start,
                   selected_rmse=float(np.sqrt(np.mean((np.clip(selected[held],1,5)-y[held])**2))),
                   refit_rmse=float(np.sqrt(np.mean((np.clip(refit[held],1,5)-y[held])**2))))
        records.append(rec); print(json.dumps(rec), flush=True)
    report = dict(config=vars(a), inputs=hashes, folds=records,
                  lightgbm_version=lgb.__version__, feature_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    for name, value in [('selected',selected),('refit',refit)]:
        value.astype('<f4').tofile(a.output+'.'+name+'.f32')
        report[name+'_rmse'] = float(np.sqrt(np.mean((np.clip(value,1,5)-y)**2)))
    pathlib.Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
