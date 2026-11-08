#!/usr/bin/env python3
"""Blender training-data scaling diagnostic (OTHER_APPROACHES_20261108.md, item 1).

Runs scripts/round4_boost.py unchanged, except that each outer fold's
*training* users are subsampled to --keep-fraction by a fixed user hash that
is independent of the fold and inner-split hashes. Held-out rows, columns,
features, seeds and budgets are identical, so the refit RMSE compares directly
with the 100% reference. --keep-fraction 1 must reproduce the reference exactly.

Usage: blend_data_scaling.py --keep-fraction F <round4_boost.py arguments>
"""
import sys
import time

import numpy as np

import round4_boost as rb


def keep_mask(users, fraction):
    x = users.astype(np.uint64) * np.uint64(31) + np.uint64(0x5bd1e995)
    with np.errstate(over='ignore'):
        x = (x ^ (x >> np.uint64(33))) * np.uint64(0xff51afd7ed558ccd)
        x = (x ^ (x >> np.uint64(33))) * np.uint64(0xc4ceb9fe1a85ec53)
    return ((x ^ (x >> np.uint64(33))) % np.uint64(1_000_000)) < np.uint64(round(fraction * 1_000_000))


def fit_fold(fold):
    """round4_boost.fit_fold with `train &= keep`; everything else unchanged."""
    x, y, users, rx, a = rb.STATE['x'], rb.STATE['y'], rb.STATE['users'], rb.STATE['rx'], rb.STATE['args']
    start = time.monotonic()
    outer = rb.blend_fold(users)
    train, held = (outer != fold) & rb.STATE['keep'], outer == fold
    inner = rb.blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    fit, valid = train & ~inner, train & inner
    inner_base = 3.6 + rx@rb.ridge_fit(rx[fit], y[fit])
    ridge_valid = float(np.sqrt(np.mean((np.clip(inner_base[held], 1, 5)-y[held])**2)))
    base = 3.6 + rx@rb.ridge_fit(rx[train], y[train])
    ridge_held = base[held].copy()
    correction, gain, chosen = np.zeros(held.sum()), 0, []
    for seed in a.seeds:
        params = dict(objective='regression', learning_rate=a.learning_rate, num_leaves=a.leaves,
                      min_data_in_leaf=a.min_leaf, feature_fraction=.8, bagging_fraction=.8,
                      bagging_freq=1, lambda_l2=10, num_threads=1, verbose=-1, seed=seed)
        dt = rb.lgb.Dataset(x[fit], y[fit]-inner_base[fit])
        dv = rb.lgb.Dataset(x[valid], y[valid]-inner_base[valid], reference=dt)
        model = rb.lgb.train(params, dt, a.rounds, valid_sets=[dv],
                             callbacks=[rb.lgb.early_stopping(100, verbose=False)])
        chosen.append(model.best_iteration)
        model = rb.lgb.train(params, rb.lgb.Dataset(x[train], y[train]-base[train]), chosen[-1])
        correction += model.predict(x[held], num_threads=1)/len(a.seeds)
        gain = gain + model.feature_importance('gain')
    refit = base[held] + correction
    return fold, refit, ridge_held, dict(
        fold=fold, chosen_rounds=chosen if len(chosen) > 1 else chosen[0], seconds=time.monotonic()-start,
        train_rows=int(train.sum()), inner_ridge_rmse=ridge_valid,
        ridge_rmse=float(np.sqrt(np.mean((np.clip(ridge_held, 1, 5)-y[held])**2))),
        refit_rmse=float(np.sqrt(np.mean((np.clip(refit, 1, 5)-y[held])**2))),
        gain=gain.tolist())


class KeepState(dict):
    """Adds the keep mask when round4_boost.main() stores the probe users."""
    fraction = 1.

    def update(self, *args, **kw):
        super().update(*args, **kw)
        if 'users' in self and 'keep' not in self:
            self['keep'] = keep_mask(self['users'], self.fraction)


def main():
    argv = sys.argv[1:]
    i = argv.index('--keep-fraction')
    fraction = float(argv[i + 1])
    if not 0 < fraction <= 1:
        raise ValueError('--keep-fraction must be in (0, 1]')
    sys.argv = [sys.argv[0]] + argv[:i] + argv[i + 2:]
    KeepState.fraction = fraction
    rb.STATE = KeepState()
    rb.fit_fold = fit_fold
    print('{"keep_fraction": %r}' % fraction, flush=True)
    rb.main()


if __name__ == '__main__':
    main()
