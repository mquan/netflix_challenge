#!/usr/bin/env python3
"""Gradient-boosted blend of probe prediction columns with user/movie meta-features.

Uses the same five user folds as `netflix ensemble` (blend_fold), so a row's
label never fits the model that scores it. Base models still select epochs on
probe: this is development CV, not an untouched test score.
Requires numpy and lightgbm.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HEADER = np.dtype([('magic', 'S8'), ('users', '<u4'), ('items', '<u4'),
                   ('train_count', '<u8'), ('probe_count', '<u8'), ('implicit_count', '<u8'),
                   ('min_day', '<u2'), ('max_day', '<u2'), ('pad', '<u4'), ('mean', '<f8')])
PROBE = np.dtype([('user', '<u4'), ('item', '<u2'), ('day', '<u2'), ('rating', 'u1'), ('pad', 'V3')])


def load(path):
    raw = np.memmap(path, mode='r', dtype=np.uint8)
    h = raw[:HEADER.itemsize].view(HEADER)[0]
    users, n, p, m = int(h['users']), int(h['train_count']), int(h['probe_count']), int(h['implicit_count'])
    pos = HEADER.itemsize

    def take(dtype, count):
        nonlocal pos
        size = np.dtype(dtype).itemsize*count
        out = raw[pos:pos+size].view(dtype)
        pos += size
        return out
    d = dict(users=users, items=int(h['items']), mean=float(h['mean']),
             min_day=int(h['min_day']), max_day=int(h['max_day']))
    d['train_offsets'] = take('<u8', users+1)
    d['train_items'] = take('<u2', n)
    d['train_days'] = take('<u2', n)
    d['train_ratings'] = take('u1', n)
    d['probe'] = take(PROBE, p)
    d['implicit_offsets'] = take('<u8', users+1)
    d['implicit_items'] = take('<u2', m)
    d['implicit_days'] = take('<u2', m)
    if pos != raw.size:
        sys.exit('dataset layout mismatch')
    return d


def blend_fold(user):
    x = user.astype(np.uint64) + np.uint64(0x9e3779b97f4a7c15)
    with np.errstate(over='ignore'):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) % np.uint64(5)).astype(np.int64)


def meta_features(d):
    probe = d['probe']
    u = probe['user'].astype(np.int64)
    i = probe['item'].astype(np.int64) - 1
    day = probe['day'].astype(np.float64)
    to, io = d['train_offsets'].astype(np.int64), d['implicit_offsets'].astype(np.int64)
    ratings = d['train_ratings'].astype(np.float64)
    user_n = np.diff(to).astype(np.float64)
    owner = np.repeat(np.arange(d['users']), np.diff(to))
    rsum = np.bincount(owner, ratings, d['users'])
    rsq = np.bincount(owner, ratings*ratings, d['users'])
    umean = (rsum + 3.6*5)/(user_n + 5)
    ustd = np.sqrt(np.maximum(rsq/np.maximum(user_n, 1) - (rsum/np.maximum(user_n, 1))**2, 0))
    items = d['train_items'].astype(np.int64) - 1
    item_n = np.bincount(items, minlength=d['items']).astype(np.float64)
    imean = (np.bincount(items, ratings, d['items']) + 3.6*25)/(item_n + 25)
    imp_n = np.diff(io).astype(np.float64)
    iowner = np.repeat(np.arange(d['users']), np.diff(io))
    idays = d['implicit_days'].astype(np.float64)
    first = np.full(d['users'], np.inf); np.minimum.at(first, iowner, idays)
    last = np.full(d['users'], -np.inf); np.maximum.at(last, iowner, idays)
    uday = np.bincount(iowner, idays, d['users'])/np.maximum(imp_n, 1)
    # Ratings the user made on the same day as the probe row (implicit days are sorted per user).
    key = iowner.astype(np.int64)*65536 + d['implicit_days'].astype(np.int64)
    pkey = u*65536 + probe['day'].astype(np.int64)
    same_day = np.searchsorted(key, pkey, 'right') - np.searchsorted(key, pkey, 'left')
    cols = dict(log_user_n=np.log1p(user_n[u]), log_item_n=np.log1p(item_n[i]),
                log_implicit_n=np.log1p(imp_n[u]), user_mean=umean[u], user_std=ustd[u],
                item_mean=imean[i], day=day, day_minus_user_mean=day - uday[u],
                day_minus_first=day - first[u], user_span=last[u] - first[u],
                log_same_day=np.log1p(same_day))
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--prediction', action='append', required=True)
    ap.add_argument('--feature', action='append', default=[], help='extra float32 per-probe column, not a prediction')
    ap.add_argument('--predictions', help='write CV predictions (float32)')
    ap.add_argument('--rounds', type=int, default=1500)
    ap.add_argument('--leaves', type=int, default=63)
    ap.add_argument('--learning-rate', type=float, default=.03)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-meta', action='store_true')
    args = ap.parse_args()
    import lightgbm as lgb

    d = load(args.data)
    y = d['probe']['rating'].astype(np.float64)
    folds = blend_fold(d['probe']['user'])
    names, cols = [], []
    for path in args.prediction:
        col = np.fromfile(path, dtype='<f4')
        if col.size != y.size or not np.isfinite(col).all():
            sys.exit(f'bad prediction column {path}')
        print(f'column={path} probe_rmse={np.sqrt(np.mean((y - np.clip(col, 1, 5))**2)):.10f}')
        names.append(pathlib.Path(path).stem); cols.append(col.astype(np.float64))
    for path in args.feature:
        col = np.fromfile(path, dtype='<f4')
        if col.size != y.size or not np.isfinite(col).all():
            sys.exit(f'bad feature column {path}')
        names.append(pathlib.Path(path).stem); cols.append(col.astype(np.float64))
    if not args.no_meta:
        for k, v in meta_features(d).items():
            names.append(k); cols.append(v)
    x = np.column_stack(cols).astype(np.float32)
    params = dict(objective='regression', learning_rate=args.learning_rate, num_leaves=args.leaves,
                  min_data_in_leaf=200, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                  lambda_l2=10, num_threads=args.threads, verbose=-1, seed=2026)
    cv = np.zeros_like(y)
    rounds = []
    for fold in range(5):
        train, held = folds != fold, folds == fold
        # Early stopping on a user-disjoint slice of the training folds, never the held fold.
        inner = blend_fold(d['probe']['user'][train].astype(np.uint64)*np.uint64(7) + np.uint64(1)) == 0
        tx, ty = x[train], y[train]
        dt = lgb.Dataset(tx[~inner], ty[~inner], feature_name=names, free_raw_data=False)
        dv = lgb.Dataset(tx[inner], ty[inner], reference=dt)
        model = lgb.train(params, dt, args.rounds, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds.append(model.best_iteration)
        cv[held] = model.predict(x[held], num_iteration=model.best_iteration)
        rmse = np.sqrt(np.mean((y[held] - np.clip(cv[held], 1, 5))**2))
        print(f'heldout_user_fold={fold} count={held.sum()} rounds={model.best_iteration} rmse={rmse:.10f}', flush=True)
        if fold == 0:
            gain = model.feature_importance('gain')
            order = np.argsort(-gain)
            print('importance', json.dumps({names[j]: round(float(gain[j]/gain.sum()), 4) for j in order}))
    score = np.sqrt(np.mean((y - np.clip(cv, 1, 5))**2))
    print(f'gbdt_blend_cv_rmse={score:.10f} columns={len(args.prediction)} meta={not args.no_meta} rounds={rounds}')
    if args.predictions:
        cv.astype('<f4').tofile(args.predictions)


if __name__ == '__main__':
    main()
