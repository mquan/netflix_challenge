#!/usr/bin/env python3
"""Round-4 residual-tree blend: frozen 21-column union bank plus extra columns.

Same nested protocol as moonshot_boost.py --residual: per outer user fold, a
contextual ridge is fit on training users only, a LightGBM model learns its
residual with rounds chosen on a user-disjoint inner split, then is refit on
all outer-training users. Folds run in parallel processes, one thread each.
With several --seeds, each seed repeats that recipe and the refits are averaged.
Union-bank paths resolve under --bank-root and are hash-checked against the
moonshot manifest. Extra columns are hashed into the report.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import time

import lightgbm as lgb
import numpy as np
from gbdt_blend import load, blend_fold, meta_features
from moonshot_nn_blend import contexts, design, ridge_fit

STATE = {}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def column(path):
    col = np.fromfile(path, dtype='<f4')
    assert np.isfinite(col).all(), path
    return col


def extra_meta(d):
    """Tree-only features beyond meta_features: movie age at the rating date,
    user rating-scale extremity, unrated (probe/qualifying) volume and the
    probe day's volume relative to the user's typical rating day."""
    probe = d['probe']
    u = probe['user'].astype(np.int64)
    i = probe['item'].astype(np.int64) - 1
    day = probe['day'].astype(np.float64)
    to, io = d['train_offsets'].astype(np.int64), d['implicit_offsets'].astype(np.int64)
    ratings = d['train_ratings'].astype(np.float64)
    user_n = np.diff(to).astype(np.float64)
    owner = np.repeat(np.arange(d['users']), np.diff(to))
    frac = {k: np.bincount(owner, ratings == k, d['users']) for k in (1, 5)}
    prior = {k: float(np.mean(ratings == k)) for k in (1, 5)}
    items = d['train_items'].astype(np.int64) - 1
    item_n = np.bincount(items, minlength=d['items']).astype(np.float64)
    isum, isq = np.bincount(items, ratings, d['items']), np.bincount(items, ratings*ratings, d['items'])
    istd = np.sqrt(np.maximum(isq/np.maximum(item_n, 1) - (isum/np.maximum(item_n, 1))**2, 0))
    imp_n = np.diff(io).astype(np.float64)
    iowner = np.repeat(np.arange(d['users']), np.diff(io))
    idays = d['implicit_days'].astype(np.int64)
    iitems = d['implicit_items'].astype(np.int64) - 1
    first_item = np.full(d['items'], np.iinfo(np.int64).max); np.minimum.at(first_item, iitems, idays)
    # Implicit days are sorted per user, so a new rating day starts at every change.
    new_day = np.ones(len(idays), dtype=bool); new_day[1:] = (idays[1:] != idays[:-1]) | (iowner[1:] != iowner[:-1])
    user_days = np.bincount(iowner, new_day, d['users'])
    key = iowner.astype(np.int64)*65536 + idays
    pkey = u*65536 + probe['day'].astype(np.int64)
    same_day = np.searchsorted(key, pkey, 'right') - np.searchsorted(key, pkey, 'left')
    per_day = imp_n/np.maximum(user_days, 1)
    cols = dict(movie_age=day - first_item[i],
                user_frac1=((frac[1] + 5*prior[1])/(user_n + 5))[u],
                user_frac5=((frac[5] + 5*prior[5])/(user_n + 5))[u],
                log_unrated_n=np.log1p(imp_n - user_n)[u], log_user_days=np.log1p(user_days)[u],
                same_day_ratio=same_day/per_day[u], item_std=istd[i])
    return np.column_stack(list(cols.values())).astype(np.float32), list(cols)


def fit_fold(fold):
    x, y, users, rx, a = STATE['x'], STATE['y'], STATE['users'], STATE['rx'], STATE['args']
    start = time.monotonic()
    outer = blend_fold(users)
    train, held = outer != fold, outer == fold
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    fit, valid = train & ~inner, train & inner
    inner_base = 3.6 + rx@ridge_fit(rx[fit], y[fit])
    ridge_valid = float(np.sqrt(np.mean((np.clip(inner_base[held], 1, 5)-y[held])**2)))
    base = 3.6 + rx@ridge_fit(rx[train], y[train])
    ridge_held = base[held].copy()
    correction, gain, chosen = np.zeros(held.sum()), 0, []
    for seed in a.seeds:
        params = dict(objective='regression', learning_rate=a.learning_rate, num_leaves=a.leaves,
                      min_data_in_leaf=a.min_leaf, feature_fraction=.8, bagging_fraction=.8,
                      bagging_freq=1, lambda_l2=10, num_threads=1, verbose=-1, seed=seed)
        dt = lgb.Dataset(x[fit], y[fit]-inner_base[fit])
        dv = lgb.Dataset(x[valid], y[valid]-inner_base[valid], reference=dt)
        model = lgb.train(params, dt, a.rounds, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        chosen.append(model.best_iteration)
        model = lgb.train(params, lgb.Dataset(x[train], y[train]-base[train]), chosen[-1])
        correction += model.predict(x[held], num_threads=1)/len(a.seeds)
        gain = gain + model.feature_importance('gain')
    refit = base[held] + correction
    return fold, refit, ridge_held, dict(
        fold=fold, chosen_rounds=chosen if len(chosen) > 1 else chosen[0], seconds=time.monotonic()-start,
        inner_ridge_rmse=ridge_valid,
        ridge_rmse=float(np.sqrt(np.mean((np.clip(ridge_held, 1, 5)-y[held])**2))),
        refit_rmse=float(np.sqrt(np.mean((np.clip(refit, 1, 5)-y[held])**2))),
        gain=gain.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--bank-root', default='/private/tmp/netflix-moonshot')
    ap.add_argument('--no-union', action='store_true', help='omit the 21-column union bank')
    ap.add_argument('--extra', nargs='*', default=[], help='extra prediction columns')
    ap.add_argument('--extra-support', nargs='*', default=[], help='extra tree-only features')
    ap.add_argument('--rounds', type=int, default=3000)
    ap.add_argument('--learning-rate', type=float, default=.05)
    ap.add_argument('--leaves', type=int, default=63)
    ap.add_argument('--min-leaf', type=int, default=200)
    ap.add_argument('--processes', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026], help='LightGBM seeds to average')
    ap.add_argument('--extra-meta', action='store_true', help='add extra_meta() tree features')
    a = ap.parse_args()
    if pathlib.Path(a.output+'.json').exists():
        ap.error('output already exists; use a new experiment name')
    root = pathlib.Path(a.bank_root)
    # A local compact bank can be built from --extra columns when the old
    # moonshot bank has been lost. No external manifest is needed in that case.
    if a.no_union:
        bank, support = {}, {}
    else:
        manifest = json.loads((root/'experiments/UNION-round-manifest.json').read_text())
        bank = manifest['reports']['union']['inputs']
        support = json.loads((root/'experiments/BOOST-support-inputs.json').read_text())
    if not bank and not a.extra:
        ap.error('at least one prediction column is required')
    names, cols, hashes = [], [], {}
    for name, expected in list(bank.items()):
        path = root/name
        if sha(path) != expected: raise ValueError(path)
        names.append(name); cols.append(column(path)); hashes[name] = expected
    for path in a.extra:
        names.append(path); cols.append(column(path)); hashes[path] = sha(path)
    predictions = np.column_stack(cols)
    tree_names, tree_cols = list(names), list(cols)
    for name, expected in support.items():
        path = root/name
        if sha(path) != expected: raise ValueError(path)
        tree_names.append(name); tree_cols.append(column(path)); hashes[name] = expected
    for path in a.extra_support:
        tree_names.append(path); tree_cols.append(column(path)); hashes[path] = sha(path)
    d = load(a.data); y = d['probe']['rating'].astype(np.float64)
    cache = pathlib.Path('artifacts/r4/meta-context.npz')
    if cache.exists():
        with np.load(cache) as saved: meta, context = saved['meta'], saved['context']
    else:
        meta = np.column_stack(list(meta_features(d).values())).astype(np.float32)
        context = contexts(d)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, meta=meta, context=context)
    tree_names += ['meta%d' % k for k in range(meta.shape[1])]
    blocks = [meta]
    if a.extra_meta:
        more, more_names = extra_meta(d)
        blocks.append(more); tree_names += more_names
    x = np.column_stack(tree_cols + blocks).astype(np.float32)
    rx = design(predictions, context)[0]
    STATE.update(x=x, y=y, users=d['probe']['user'], rx=rx, args=a)
    print(json.dumps(dict(config=vars(a), prediction_columns=len(names), tree_features=x.shape[1])), flush=True)
    refit, ridge = np.empty(len(y)), np.empty(len(y)); records = []
    folds = blend_fold(d['probe']['user'])
    with mp.get_context('fork').Pool(a.processes) as pool:
        for f, r, rb, rec in pool.imap_unordered(fit_fold, range(5)):
            refit[folds == f], ridge[folds == f] = r, rb
            gain = rec.pop('gain'); total = sum(gain) or 1
            rec['top_gain'] = sorted(((round(g/total, 4), n) for g, n in zip(gain, tree_names)), reverse=True)[:12]
            records.append(rec); print(json.dumps(rec), flush=True)
    records.sort(key=lambda r: r['fold'])
    report = dict(config=vars(a), inputs=hashes, folds=records, lightgbm_version=lgb.__version__)
    for name, value in [('refit', refit), ('ridge', ridge)]:
        value.astype('<f4').tofile(a.output+'.'+name+'.f32')
        report[name+'_rmse'] = float(np.sqrt(np.mean((np.clip(value, 1, 5)-y)**2)))
        report[name+'_fold_rmse'] = [float(np.sqrt(np.mean((np.clip(value[folds == f], 1, 5)-y[folds == f])**2)))
                                     for f in range(5)]
    pathlib.Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in report if k.endswith('rmse')}), flush=True)


if __name__ == '__main__': main()
