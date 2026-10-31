#!/usr/bin/env python3
"""Headroom map H4: deployable proxies for the user's current shift.

H1-H3 (HEADROOM.md) found that the remaining shared error is user-level and
current. This measures how much of it is predictable from information
available at test time, using no held-out label:

- size of the user's held-out set (all rows, same-day rows)
- mean base prediction over the user's *other* held-out movies, relative to
  the target's prediction and to the user's training mean
- days since the user's last training rating; target day minus the user's
  mean training day
- user training count, mean and std, and the mean of the last 10/50 training
  ratings minus the overall mean (recent drift)
- unlabelled implicit choices: total beyond training, and on the target day
- movie training popularity, and the user's held-out popularity mean vs training

A LightGBM model on the cross-fitted base residual is trained on user folds
2-4, early-stopped on fold 1 and evaluated on fold 0, as in the fixed gate.
The reported Δ RMSE is the deployable gain over the base on fold 0.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold, load
from headroom_map import crossfit, leave_one_out
from local_context_chain import paired
from selection_objective_gate import ridge_predict
from train_holdout import digest

BIG = 1 << 20


def per_user_training(d):
    users = int(d['users'])
    counts = np.diff(d['train_offsets']).astype(np.int64)
    uid = np.repeat(np.arange(users), counts)
    days = d['train_days'].astype(np.int64)
    ratings = d['train_ratings'].astype(float)
    order = np.lexsort((days, uid))  # by user, then date
    uid_s, days_s, r_s = uid[order], days[order], ratings[order]
    safe = np.maximum(counts, 1)
    mean = np.bincount(uid, ratings, users) / safe
    sq = np.bincount(uid, ratings ** 2, users) / safe
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    mean_day = np.bincount(uid, days, users) / safe
    last_day = np.full(users, -1, dtype=np.int64)
    np.maximum.at(last_day, uid, days)
    # rank from the end within each user (0 = most recent rating)
    ends = np.cumsum(counts)
    from_end = ends[uid_s] - 1 - np.arange(len(uid_s))
    recent = {}
    for k in (10, 50):
        keep = from_end < k
        n = np.bincount(uid_s[keep], minlength=users)
        recent[k] = np.where(n > 0, np.bincount(uid_s[keep], r_s[keep], users) / np.maximum(n, 1), mean) - mean
    item_pop = np.bincount(d['train_items'].astype(np.int64), minlength=int(d['items']) + 1)
    return dict(counts=counts, mean=mean, std=std, mean_day=mean_day, last_day=last_day,
                recent10=recent[10], recent50=recent[50], item_pop=item_pop)


def features(d, base):
    p = d['probe']
    u, i, day = p['user'].astype(np.int64), p['item'].astype(np.int64), p['day'].astype(np.int64)
    t = per_user_training(d)
    s_pred, n_other = leave_one_out(u, base)
    other_mean_pred = np.where(n_other > 0, s_pred / np.maximum(n_other, 1), base)
    _, n_same_day = leave_one_out(u * BIG + day, np.zeros(len(u)))
    pop = np.log1p(t['item_pop'][i])
    s_pop, _ = leave_one_out(u, pop)
    other_pop = np.where(n_other > 0, s_pop / np.maximum(n_other, 1), pop)
    imp_u = np.repeat(np.arange(int(d['users'])), np.diff(d['implicit_offsets']).astype(np.int64))
    imp_count = np.bincount(imp_u, minlength=int(d['users']))
    imp_keys = np.sort(imp_u * BIG + d['implicit_days'].astype(np.int64))
    q = u * BIG + day
    imp_same_day = np.searchsorted(imp_keys, q, 'right') - np.searchsorted(imp_keys, q, 'left')
    cols = dict(
        base=base, n_other=np.log1p(n_other), n_same_day=np.log1p(n_same_day),
        other_mean_pred=other_mean_pred, other_minus_target=other_mean_pred - base,
        other_minus_user_mean=other_mean_pred - t['mean'][u],
        gap_days=np.where(t['last_day'][u] >= 0, day - t['last_day'][u], -1),
        day_minus_mean_day=day - t['mean_day'][u],
        train_count=np.log1p(t['counts'][u]), user_mean=t['mean'][u], user_std=t['std'][u],
        recent10=t['recent10'][u], recent50=t['recent50'][u],
        unlabelled_choices=np.log1p(np.maximum(imp_count[u] - t['counts'][u], 0)),
        implicit_same_day=np.log1p(imp_same_day),
        item_pop=pop, other_pop_minus_train_pop=other_pop - pop.mean(),
    )
    return np.column_stack(list(cols.values())), list(cols)


def run(data_path, bank_paths, output, seed=20261031):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    d = load(data_path)
    p = d['probe']
    y, users = p['rating'].astype(float), p['user']
    bank = np.column_stack([np.fromfile(c, dtype='<f4').astype(float) for c in bank_paths])
    folds = blend_fold(users)
    held = folds == 0
    base_cf = crossfit(bank, y, folds)          # out-of-fold base for every row
    base_gate, _ = ridge_predict(bank, y, folds)  # the fixed gate's fold-0 base
    x, names = features(d, base_cf)
    e = y - base_cf
    train, stop = folds >= 2, folds == 1
    params = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=500,
                  feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=10, seed=seed,
                  num_threads=2, verbose=-1)
    model = lgb.train(params, lgb.Dataset(x[train], e[train]), 2000,
                      valid_sets=[lgb.Dataset(x[stop], e[stop])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    corrected = base_cf[held] + model.predict(x[held], num_iteration=model.best_iteration)
    stats_cf = paired(y[held], base_cf[held], corrected, users[held])
    gain = model.feature_importance('gain')
    report = dict(data=str(data_path), data_sha256=digest(data_path), bank=list(map(str, bank_paths)),
                  rows=int(len(y)), held_rows=int(held.sum()), best_iteration=int(model.best_iteration),
                  base_crossfit_fold0_rmse=stats_cf['control_rmse'],
                  base_gate_fold0_rmse=float(np.sqrt(np.mean((np.clip(base_gate, 1, 5) - y[held]) ** 2))),
                  deployable=stats_cf,
                  importance={n: float(g) for n, g in sorted(zip(names, gain), key=lambda t: -t[1])},
                  metric='Fold-0 RMSE change from a LightGBM residual model on label-free test-time features.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(delta_rmse=stats_cf['delta_rmse'], base=stats_cf['control_rmse'],
                          best_iteration=report['best_iteration'],
                          top=list(report['importance'])[:8]), indent=1), flush=True)
    return report


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--plan', required=True, help='JSON with "data" and "bank" (SO1-plan style)')
    a.add_argument('--output', required=True)
    args = a.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    run(plan['data'], plan['bank'], args.output)


if __name__ == '__main__':
    main()
