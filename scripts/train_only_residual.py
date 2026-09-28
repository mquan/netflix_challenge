#!/usr/bin/env python3
"""Fit a residual tree on a training-only pseudo-probe; apply it to real probe.

The source factor model for pseudo-probe rows must be trained without those
ratings and with a fixed epoch. Target probe labels are read for reporting only.
"""
import argparse
import hashlib
import json
import pathlib

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold, load, meta_features


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def features(d, base):
    return np.column_stack([base] + list(meta_features(d).values())).astype(np.float32)


def read_column(path, size):
    col = np.fromfile(path, dtype='<f4')
    if len(col) != size or not np.isfinite(col).all():
        raise ValueError('invalid prediction column: ' + str(path))
    return col


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.clip(pred, 1, 5)-y)**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--holdout-data', required=True)
    p.add_argument('--holdout-pred', required=True)
    p.add_argument('--target-data', required=True)
    p.add_argument('--target-pred', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--rounds', type=int, default=2000)
    p.add_argument('--min-leaf', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260927)
    a = p.parse_args()
    output = pathlib.Path(a.output)
    if output.exists() or pathlib.Path(str(output) + '.json').exists():
        p.error('output already exists')
    holdout, target = load(a.holdout_data), load(a.target_data)
    hb = read_column(a.holdout_pred, len(holdout['probe']))
    tb = read_column(a.target_pred, len(target['probe']))
    hx, tx = features(holdout, hb), features(target, tb)
    hy = holdout['probe']['rating'].astype(np.float32)
    ty = target['probe']['rating'].astype(np.float32)
    users = holdout['probe']['user']
    valid = blend_fold(users) == 0
    params = dict(objective='regression', learning_rate=.03, num_leaves=31,
                  min_data_in_leaf=a.min_leaf, feature_fraction=.9, bagging_fraction=.9,
                  bagging_freq=1, lambda_l2=10, num_threads=1, verbose=-1,
                  seed=a.seed)
    inner_train = lgb.Dataset(hx[~valid], hy[~valid]-hb[~valid])
    inner_valid = lgb.Dataset(hx[valid], hy[valid]-hb[valid], reference=inner_train)
    inner = lgb.train(params, inner_train, a.rounds, valid_sets=[inner_valid],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    chosen = inner.best_iteration
    inner_pred = hb[valid] + inner.predict(hx[valid], num_threads=1)
    final = lgb.train(params, lgb.Dataset(hx, hy-hb), chosen)
    corrected = tb + final.predict(tx, num_threads=1)
    corrected.astype('<f4').tofile(output)
    report = dict(config=vars(a), inputs={str(path): digest(path) for path in
                  (a.holdout_data, a.holdout_pred, a.target_data, a.target_pred)},
                  lightgbm_version=lgb.__version__, chosen_rounds=chosen,
                  holdout_inner_base_rmse=rmse(hb[valid], hy[valid]),
                  holdout_inner_corrected_rmse=rmse(inner_pred, hy[valid]),
                  target_base_rmse=rmse(tb, ty), target_corrected_rmse=rmse(corrected, ty),
                  target_user_fold_rmse=[rmse(corrected[blend_fold(target['probe']['user']) == f],
                                              ty[blend_fold(target['probe']['user']) == f]) for f in range(5)],
                  output_sha256=digest(output))
    pathlib.Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
