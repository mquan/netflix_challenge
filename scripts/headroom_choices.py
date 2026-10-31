#!/usr/bin/env python3
"""H8 (HEADROOM.md): deployable choice-identity affinity beyond N1 on P2.

Reads K1's item factors q and SVD++ implicit factors y from a joint model
(trained on T - P2), projects both to the top principal directions of y,
and builds per-row features from the user's *unlabelled* choices (implicit
pairs absent from training, target movie excluded) within day windows. Then
it fits a LightGBM residual model over a cross-fitted ridge of the base bank
(fit on folds 2-4, early-stopped on 1, evaluated on 0), as headroom_proxies.py.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold, load
from headroom_map import crossfit
from local_context_chain import paired
from train_holdout import digest

BIG = 1 << 20
WINDOWS = (3, 30, 100000)
DIMS = 32


def read_joint_factors(path, items):
    with open(path, 'rb') as f:
        magic, _fp = np.frombuffer(f.read(16), '<u8')
        rank, _epoch = np.frombuffer(f.read(8), '<u4')
        vectors = []
        for _ in range(12):
            n = int(np.frombuffer(f.read(8), '<u8')[0])
            vectors.append(np.frombuffer(f.read(4 * n), '<f4'))
    q, y = vectors[9], vectors[10]
    rank = int(rank)
    if q.size != items * rank or y.size != items * rank:
        raise ValueError('unexpected factor sizes')
    return q.reshape(items, rank), y.reshape(items, rank)


def unlabelled_choices(d):
    users = int(d['users'])
    iu = np.repeat(np.arange(users), np.diff(d['implicit_offsets']).astype(np.int64))
    ikey = iu * BIG + d['implicit_items'].astype(np.int64)
    tu = np.repeat(np.arange(users), np.diff(d['train_offsets']).astype(np.int64))
    tkey = np.sort(tu * BIG + d['train_items'].astype(np.int64))
    pos = np.searchsorted(tkey, ikey)
    rated = (pos < len(tkey)) & (tkey[np.minimum(pos, len(tkey) - 1)] == ikey)
    keep = ~rated
    return iu[keep], d['implicit_items'][keep].astype(np.int64), d['implicit_days'][keep].astype(np.int64)


def features(d, q, y):
    p = d['probe']
    u, i, t = p['user'].astype(np.int64), p['item'].astype(np.int64), p['day'].astype(np.int64)
    # principal directions of y; project both embeddings
    yc = y - y.mean(0)
    _, _, vt = np.linalg.svd(yc[np.linalg.norm(y, axis=1) > 0], full_matrices=False)
    basis = vt[:DIMS].T
    yp, qp = y @ basis, q @ basis
    cu, ci, cd = unlabelled_choices(d)
    order = np.lexsort((cd, cu))
    cu, ci, cd = cu[order], ci[order], cd[order]
    key = cu * BIG + cd
    zeros = np.zeros((1, DIMS))
    prefix_y = np.vstack([zeros, np.cumsum(yp[ci - 1], 0)])
    prefix_q = np.vstack([zeros, np.cumsum(qp[ci - 1], 0)])
    target_y, target_q = yp[i - 1], qp[i - 1]
    # Is the target pair itself among the unlabelled choices? (true for every P2/probe row)
    self_key = u * BIG + t
    cols, names = [], []
    for w in WINDOWS:
        lo = np.searchsorted(key, u * BIG + np.maximum(t - w, 0), 'left')
        hi = np.searchsorted(key, u * BIG + np.minimum(t + w, BIG - 1), 'right')
        n = (hi - lo).astype(float)
        sy = prefix_y[hi] - prefix_y[lo]
        sq = prefix_q[hi] - prefix_q[lo]
        # exclude the target movie itself (present iff the target pair is an unlabelled choice)
        inside = (hi > lo)
        sy = sy - target_y * inside[:, None]
        sq = sq - target_q * inside[:, None]
        n = np.maximum(n - inside, 0)
        safe = np.maximum(n, 1)[:, None]
        my, mq = sy / safe, sq / safe
        cols += [np.log1p(n), np.sum(target_q * my, 1), np.sum(target_q * mq, 1), np.sum(target_y * my, 1)]
        names += [f'n_w{w}', f'q_dot_meany_w{w}', f'q_dot_meanq_w{w}', f'y_dot_meany_w{w}']
    del self_key
    return np.column_stack(cols), names


def run(plan_path, model_path, output, seed=20261031):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    d = load(plan['data'])
    p = d['probe']
    y, users = p['rating'].astype(float), p['user']
    bank = np.column_stack([np.fromfile(c, dtype='<f4').astype(float) for c in plan['bank']])
    folds = blend_fold(users)
    held = folds == 0
    base = crossfit(bank, y, folds)
    q, emb = read_joint_factors(model_path, int(d['items']))
    x, names = features(d, q, emb)
    x = np.column_stack([base, x]); names = ['base'] + names
    e = y - base
    train, stop = folds >= 2, folds == 1
    params = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=500,
                  feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=10, seed=seed,
                  num_threads=2, verbose=-1)
    model = lgb.train(params, lgb.Dataset(x[train], e[train]), 2000, valid_sets=[lgb.Dataset(x[stop], e[stop])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    corrected = base[held] + model.predict(x[held], num_iteration=model.best_iteration)
    stats = paired(y[held], base[held], corrected, users[held])
    # control: the same residual model with only the base feature (measures tree recalibration alone)
    m0 = lgb.train(params, lgb.Dataset(x[train][:, :1], e[train]), 2000,
                   valid_sets=[lgb.Dataset(x[stop][:, :1], e[stop])], callbacks=[lgb.early_stopping(100, verbose=False)])
    base_only = base[held] + m0.predict(x[held][:, :1], num_iteration=m0.best_iteration)
    vs_recal = paired(y[held], base_only, corrected, users[held])
    gain = model.feature_importance('gain')
    report = dict(plan=str(plan_path), bank=plan['bank'], model=str(model_path), model_sha256=digest(model_path),
                  rows=int(len(y)), held_rows=int(held.sum()), best_iteration=int(model.best_iteration),
                  choices_vs_base=stats, choices_vs_base_recalibrated=vs_recal,
                  importance={n: float(g) for n, g in sorted(zip(names, gain), key=lambda t: -t[1])},
                  metric='P2 fold-0 RMSE change from label-free choice-affinity features over bank + N1.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(delta_vs_base=stats['delta_rmse'], delta_vs_recalibrated=vs_recal['delta_rmse'],
                          ci=vs_recal['cluster_normal_95_mse_interval'], top=list(report['importance'])[:6]), indent=1))
    return report


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--plan', required=True)
    a.add_argument('--model', required=True, help='joint model trained on the plan dataset (T - P2)')
    a.add_argument('--output', required=True)
    args = a.parse_args()
    run(args.plan, args.model, args.output)


if __name__ == '__main__':
    main()
