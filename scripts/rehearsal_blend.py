#!/usr/bin/env python3
"""Stage C rehearsal blender: fit on P2, apply to two probe banks, compare with dev CV.

Recipe matches round4_boost.py (contextual ridge plus a LightGBM residual tree
with extra meta features), but the blender is fit once on every P2 row instead
of in user folds:

  fit    P2 columns from models trained on train minus P2 (epochs chosen on P2).
  a      original-probe columns from those same models: no retrain.
  b      original-probe columns from models retrained on all training ratings
         at the P2-selected epochs: the Stage C analogue.
  c      the usual five-user-fold development CV on the b columns, for the
         offset between our dev CV metric and a Stage C score.

The spec is JSON: {"fit": SET, "a": SET, "b": SET}, SET = {"data": path,
"columns": [paths], "support": [paths]}, with columns/support in the same
order in every set. Original probe labels are read only for scoring a and b
and for the c folds' own CV.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import time

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold, load, meta_features
from moonshot_nn_blend import contexts, design, ridge_fit
from round4_boost import column, extra_meta

STATE = {}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def features(spec):
    d = load(spec['data'])
    cols = [column(p) for p in spec['columns']]
    support = [column(p) for p in spec.get('support', [])]
    n = len(d['probe'])
    if any(len(c) != n for c in cols + support):
        raise ValueError('column length differs from probe rows of ' + spec['data'])
    predictions = np.column_stack(cols)
    meta = np.column_stack(list(meta_features(d).values())).astype(np.float32)
    more, _ = extra_meta(d)
    x = np.column_stack(cols + support + [meta, more]).astype(np.float32)
    rx = design(predictions, contexts(d))[0]
    return dict(x=x, rx=rx, y=d['probe']['rating'].astype(np.float64), users=d['probe']['user'].copy())


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.clip(pred, 1, 5)-y)**2)))


def train(x, rx, y, users, a, seed):
    """round4_boost.fit_fold on all given rows; returns a predictor."""
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    fit, valid = ~inner, inner
    inner_base = 3.6 + rx@ridge_fit(rx[fit], y[fit])
    weights = ridge_fit(rx, y)
    base = 3.6 + rx@weights
    params = dict(objective='regression', learning_rate=a.learning_rate, num_leaves=a.leaves,
                  min_data_in_leaf=a.min_leaf, feature_fraction=.8, bagging_fraction=.8,
                  bagging_freq=1, lambda_l2=10, num_threads=a.threads, verbose=-1, seed=seed)
    dt = lgb.Dataset(x[fit], y[fit]-inner_base[fit])
    dv = lgb.Dataset(x[valid], y[valid]-inner_base[valid], reference=dt)
    model = lgb.train(params, dt, a.rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = model.best_iteration
    model = lgb.train(params, lgb.Dataset(x, y-base), rounds)
    return lambda tx, trx: 3.6 + trx@weights + model.predict(tx, num_threads=a.threads), rounds


def cv_fold(fold):
    s, a = STATE['set'], STATE['args']
    folds = blend_fold(s['users'])
    tr, held = folds != fold, folds == fold
    pred, rounds = np.zeros(held.sum()), []
    for seed in a.seeds:
        f, r = train(s['x'][tr], s['rx'][tr], s['y'][tr], s['users'][tr], a, seed)
        pred += f(s['x'][held], s['rx'][held])/len(a.seeds); rounds.append(r)
    return fold, pred, rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--rounds', type=int, default=8000)
    ap.add_argument('--learning-rate', type=float, default=.02)
    ap.add_argument('--leaves', type=int, default=63)
    ap.add_argument('--min-leaf', type=int, default=200)
    ap.add_argument('--threads', type=int, default=1, help='LightGBM threads per model')
    ap.add_argument('--processes', type=int, default=5, help='parallel CV folds for c')
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026])
    ap.add_argument('--skip-cv', action='store_true', help='skip the c dev-CV reference')
    a = ap.parse_args()
    if pathlib.Path(a.output + '.json').exists():
        ap.error('output already exists; use a new name')
    spec = json.loads(pathlib.Path(a.spec).read_text())
    if not (len(spec['fit']['columns']) == len(spec['a']['columns']) == len(spec['b']['columns'])):
        raise ValueError('column lists differ in length')
    start = time.monotonic()
    report = dict(config=vars(a), spec=spec, lightgbm_version=lgb.__version__,
                  inputs={p: sha(p) for k in ('fit', 'a', 'b') for p in
                          [spec[k]['data']] + spec[k]['columns'] + spec[k].get('support', [])})
    sets = {k: features(spec[k]) for k in ('fit', 'a', 'b')}
    fit = sets['fit']
    out = {k: np.zeros(len(sets[k]['y'])) for k in ('fit', 'a', 'b')}
    report['fit_rounds'] = []
    for seed in a.seeds:
        f, r = train(fit['x'], fit['rx'], fit['y'], fit['users'], a, seed)
        report['fit_rounds'].append(r)
        for k in out:
            out[k] += f(sets[k]['x'], sets[k]['rx'])/len(a.seeds)
    report['fit_in_sample_rmse'] = rmse(out['fit'], fit['y'])
    for k in ('a', 'b'):
        s = sets[k]
        folds = blend_fold(s['users'])
        report[k + '_rmse'] = rmse(out[k], s['y'])
        report[k + '_fold_rmse'] = [rmse(out[k][folds == f], s['y'][folds == f]) for f in range(5)]
        out[k].astype('<f4').tofile(a.output + '.' + k + '.f32')
        report[k + '_column_rmse'] = [rmse(s['x'][:, j], s['y']) for j in range(len(spec[k]['columns']))]
    print(json.dumps({k: v for k, v in report.items() if k.endswith('rmse')}), flush=True)
    if not a.skip_cv:
        s = sets['b']
        STATE.update(set=s, args=a)
        folds = blend_fold(s['users'])
        cv = np.empty(len(s['y'])); report['c_rounds'] = {}
        with mp.get_context('fork').Pool(a.processes) as pool:
            for fold, pred, rounds in pool.imap_unordered(cv_fold, range(5)):
                cv[folds == fold] = pred; report['c_rounds'][fold] = rounds
        report['c_rmse'] = rmse(cv, s['y'])
        report['c_fold_rmse'] = [rmse(cv[folds == f], s['y'][folds == f]) for f in range(5)]
        cv.astype('<f4').tofile(a.output + '.c.f32')
        report['c_minus_b'] = report['c_rmse'] - report['b_rmse']
    report['a_minus_b'] = report['a_rmse'] - report['b_rmse']
    report['elapsed_seconds'] = time.monotonic() - start
    pathlib.Path(a.output + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k.endswith('rmse') or k.endswith('_b')}), flush=True)


if __name__ == '__main__':
    main()
