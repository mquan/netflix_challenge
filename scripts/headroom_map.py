#!/usr/bin/env python3
"""Headroom map (STRATEGY.md, part A): oracle value of information sources on P2.

For each source, pool the cross-fitted residuals of the bank11 ridge blend
over the target row's group (for example the user's other P2 ratings on the
same day), always leaving the target row out. Add those pooled features to the
same fixed ridge gate SO1 uses: fit on user folds 2-4, pick the penalty on
fold 1, refit on 1-4, evaluate on fold 0. The RMSE gain says how much of the
bank's error is *shared within that group*, i.e. an approximate ceiling on what
a model of that source could remove.

These features read other held-out P2 labels, so they are **oracles, not
deployable predictors**. Only P2 (training ratings) is used; the original
probe is never read. A placebo reassigns rows to random groups with the same
size distribution; its gain is the floor from adding columns at all.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from selection_objective_gate import ridge_predict
from train_holdout import digest

SHRINK = (1, 5, 25)


def crossfit(x, y, folds, penalty=10.):
    """Out-of-fold ridge blend for every row (folds are by user)."""
    design = np.column_stack([np.ones(len(y)), x])
    out = np.empty(len(y))
    for f in range(5):
        fit = folds != f
        w = np.linalg.solve(design[fit].T @ design[fit] + penalty * np.eye(design.shape[1]),
                            design[fit].T @ y[fit])
        out[~fit] = design[~fit] @ w
    return out


def leave_one_out(keys, e):
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=e)
    return sums[inverse] - e, counts[inverse] - 1.


def pooled_features(keys, e):
    s, n = leave_one_out(keys, e)
    return np.column_stack([s / (n + k) for k in SHRINK] + [(n > 0).astype(float), np.log1p(n)])


def groups(p):
    u, i, d = (p['user'].astype(np.int64), p['item'].astype(np.int64), p['day'].astype(np.int64))
    big = 1 << 20
    return {
        'user': u,
        'user_month': u * big + d // 30,
        'user_week': u * big + d // 7,
        'user_day': u * big + d,
        'item': i,
        'item_month': i * big + d // 30,
        'item_week': i * big + d // 7,
        'item_day': i * big + d,
        'calendar_day': d,
    }


COMBINED = {
    'all_user_sources': ('user', 'user_month', 'user_week', 'user_day'),
    'all_item_sources': ('item', 'item_month', 'item_week', 'item_day'),
    'everything': ('user', 'user_month', 'user_week', 'user_day', 'item', 'item_month', 'item_week',
                   'item_day', 'calendar_day'),
}


def run(plan_path, output, spec_bank=None):
    """plan_path: SO1-style plan (data + bank), or with spec_bank a rehearsal spec
    (artifacts/rehearsal/spec-*.json) whose `spec_bank` entry ('a' or 'b') is scored."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    if spec_bank:
        entry = plan[spec_bank]
        plan = dict(data=entry['data'], bank=entry['columns'] + entry.get('support', []))
    data = load(plan['data'])
    p = data['probe']
    y, users = p['rating'].astype(float), p['user']
    bank = np.column_stack([np.fromfile(c, dtype='<f4').astype(float) for c in plan['bank']])
    if bank.shape[0] != len(y) or not np.isfinite(bank).all():
        raise ValueError('bank does not match P2 rows')
    folds = blend_fold(users)
    held = folds == 0
    base_pred, base_penalty = ridge_predict(bank, y, folds)
    base_rmse = float(np.sqrt(np.mean((np.clip(base_pred, 1, 5) - y[held]) ** 2)))
    e = y - crossfit(bank, y, folds)
    g = groups(p)
    rng = np.random.default_rng(20261031)
    feats = {name: pooled_features(keys, e) for name, keys in g.items()}
    for placebo_of in ('user_day', 'item_week'):  # same group-size distribution, random membership
        feats['placebo_' + placebo_of] = pooled_features(rng.permutation(g[placebo_of]), e)
    for name, parts in COMBINED.items():
        feats[name] = np.column_stack([feats[x] for x in parts])
    results = {}
    for name, f in feats.items():
        pred, penalty = ridge_predict(np.column_stack([bank, f]), y, folds)
        stats = paired(y[held], base_pred, pred, users[held])
        s, n = leave_one_out(g[name], e) if name in g else (None, None)
        results[name] = dict(delta_rmse=stats['delta_rmse'], penalty=penalty, stats=stats,
                             rows_with_other_members=None if n is None else float(np.mean(n > 0)))
        print(f"{name:22s} delta_rmse={stats['delta_rmse']:+.6f}", flush=True)
    report = dict(plan=str(plan_path), plan_sha256=digest(plan_path), spec_bank=spec_bank, data=plan['data'],
                  bank=plan['bank'],
                  rows=int(len(y)), held_rows=int(held.sum()), base_rmse=base_rmse, base_penalty=base_penalty,
                  shrink=SHRINK, results=results,
                  metric='P2 fold-0 RMSE change from adding leave-one-out pooled residual features. Oracle '
                         'upper bounds that read other held-out labels of the scored set; not deployable.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--plan', default='experiments/SO1-plan.json', help='source of the frozen P2 data and bank11')
    a.add_argument('--spec-bank', choices=('a', 'b'), help='treat --plan as a rehearsal spec and score this bank')
    a.add_argument('--output', required=True)
    args = a.parse_args()
    run(args.plan, args.output, args.spec_bank)


if __name__ == '__main__':
    main()
