#!/usr/bin/env python3
"""Frozen-geometry recurrent two-persona residual model for Netflix Probe.

This is a separate additive column, not a refit of the existing ensemble.
Probe labels are read only after all predictions have been written.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gbdt_blend import load
from round9_latent import read_model


def ridge(x, y, weight, penalty):
    gram = x.T @ (weight[:, None] * x)
    gram.flat[::len(gram) + 1] += penalty
    gram[-1, -1] -= penalty * .9  # retain a regularized persona intercept
    return np.linalg.solve(gram, x.T @ (weight * y))


def softmax2(logit):
    z = np.clip(logit, -30, 30)
    p = 1 / (1 + np.exp(-z))
    return np.column_stack((p, 1-p))


def fit_user(x, y, day, penalty, iterations, day_weight=.7):
    """Return a one-person ridge and two recurrent personas with day coupling."""
    n = len(y)
    one = ridge(x, y, np.ones(n), penalty)
    q = x[:, :-1]
    norm = np.maximum(np.linalg.norm(q, axis=1), 1e-6)
    unit = q / norm[:, None]
    # Deterministic distinct initialization from a joint item/rating axis.
    axis = unit[:, 0] + .15 * np.clip(y, -2, 2)
    split = axis >= np.median(axis)
    weights = np.column_stack((np.where(split, .8, .2), np.where(split, .2, .8)))
    _, day_id = np.unique(day, return_inverse=True)
    groups = int(day_id.max()) + 1
    for _ in range(iterations):
        beta = np.stack([ridge(x, y, weights[:, k], penalty) for k in (0, 1)])
        centers = weights.T @ unit
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)
        item_logit = 2.0 * (unit @ (centers[0] - centers[1]))
        pred = x @ beta.T
        rating_logit = .6 * ((y - pred[:, 1])**2 - (y - pred[:, 0])**2)
        day_sum = np.bincount(day_id, weights[:, 0], minlength=groups)
        day_n = np.bincount(day_id, minlength=groups)
        # Leave the target rating's own responsibility out of its day evidence.
        day_logit = np.log((day_sum[day_id]-weights[:, 0]+1.5) /
                           (day_n[day_id]-day_sum[day_id]-weights[:, 1]+1.5))
        weights = softmax2(item_logit + rating_logit + day_weight*day_logit)
    beta = np.stack([ridge(x, y, weights[:, k], penalty) for k in (0, 1)])
    centers = weights.T @ unit
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)
    return one, beta, centers, weights


def probe_correction(x, days, same_train_days, train_weights, implicit_days,
                     implicit_q, one, beta, centers, day_weight=.7, implicit_weight=.5):
    q = x[:, :-1]
    unit = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
    logit = 2.0 * (unit @ (centers[0] - centers[1]))
    for j, day in enumerate(days):
        lo = np.searchsorted(same_train_days, day, 'left')
        hi = np.searchsorted(same_train_days, day, 'right')
        if hi > lo:
            p = train_weights[lo:hi, 0].sum()
            logit[j] += day_weight * np.log((p + 1.5)/(hi-lo-p+1.5))
        il = np.searchsorted(implicit_days, day, 'left')
        ih = np.searchsorted(implicit_days, day, 'right')
        if ih > il + 1:
            # Query item is excluded from this pooled day signal by subtracting
            # its embedding once, including when the same movie is duplicated.
            pooled = implicit_q[il:ih].sum(axis=0) - q[j]
            length = max(ih-il-1, 1)
            pooled /= length
            pooled /= max(np.linalg.norm(pooled), 1e-6)
            logit[j] += implicit_weight * pooled @ (centers[0]-centers[1])
    w = softmax2(logit)
    return x @ one, np.sum((x @ beta.T) * w, axis=1), w[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--train-residual', required=True)
    ap.add_argument('--base-probe', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-history', type=int, default=100)
    ap.add_argument('--dimensions', type=int, default=8)
    ap.add_argument('--penalty', type=float, default=5.)
    ap.add_argument('--iterations', type=int, default=4)
    ap.add_argument('--day-weight', type=float, default=.7)
    ap.add_argument('--implicit-weight', type=float, default=.5)
    ap.add_argument('--max-users', type=int, default=0, help='pilot, deterministic first eligible users')
    a = ap.parse_args()
    start = time.monotonic()
    d = load(a.data)
    m, info = read_model(a.model, d['users'], d['items'])
    residual = np.memmap(a.train_residual, dtype='<f4', mode='r')
    base = np.fromfile(a.base_probe, dtype='<f4')
    if len(residual) != len(d['train_items']) or len(base) != len(d['probe']):
        raise ValueError('input length mismatch')
    q = np.asarray(m['item_factors'], dtype=np.float64)
    centered = q - q.mean(0)
    vals, vectors = np.linalg.eigh(centered.T @ centered)
    basis = vectors[:, -a.dimensions:]
    item = q @ basis
    print(f'loaded rank={info["rank"]} item_pc_variance={vals[-a.dimensions:].sum()/vals.sum():.3f}', flush=True)
    probe = d['probe']
    order = np.argsort(probe['user'], kind='stable')
    ends = np.r_[0, np.bincount(probe['user'], minlength=d['users']).cumsum()]
    corrected_one = base.copy()
    corrected_two = base.copy()
    confidence = np.zeros(len(base), np.float32)
    handled = 0
    stats = {'eligible_users': 0, 'covered_rows': 0, 'mean_abs_correction': 0.}
    for u in range(d['users']):
        lo, hi = map(int, d['train_offsets'][u:u+2])
        if hi-lo < a.min_history:
            continue
        rows = order[ends[u]:ends[u+1]]
        if not len(rows):
            continue
        if a.max_users and handled >= a.max_users:
            break
        handled += 1
        x = np.column_stack((item[d['train_items'][lo:hi]-1], np.ones(hi-lo)))
        y = np.asarray(residual[lo:hi], dtype=np.float64)
        one, beta, centers, weights = fit_user(x, y, d['train_days'][lo:hi], a.penalty,
                                                a.iterations, a.day_weight)
        px = np.column_stack((item[probe['item'][rows]-1], np.ones(len(rows))))
        il, ih = map(int, d['implicit_offsets'][u:u+2])
        imp_item = item[d['implicit_items'][il:ih]-1]
        p1, p2, conf = probe_correction(px, probe['day'][rows], d['train_days'][lo:hi], weights,
                                         d['implicit_days'][il:ih], imp_item, one, beta, centers,
                                         a.day_weight, a.implicit_weight)
        shrink = (hi-lo)/(hi-lo+200)
        corrected_one[rows] += np.float32(np.clip(p1*shrink, -.5, .5))
        corrected_two[rows] += np.float32(np.clip(p2*shrink, -.5, .5))
        confidence[rows] = np.float32(np.abs(conf-.5)*2)
        stats['covered_rows'] += len(rows)
        stats['mean_abs_correction'] += float(np.abs(p2*shrink).sum())
        if handled % 10000 == 0:
            print(f'users={handled} elapsed={time.monotonic()-start:.1f}s', flush=True)
    stats['eligible_users'] = handled
    stats['mean_abs_correction'] /= max(stats['covered_rows'], 1)
    prefix = pathlib.Path(a.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    corrected_one.astype('<f4').tofile(str(prefix)+'.one.f32')
    corrected_two.astype('<f4').tofile(str(prefix)+'.two.f32')
    confidence.astype('<f4').tofile(str(prefix)+'.confidence.f32')
    y = probe['rating'].astype(np.float64)
    for name, pred in [('base', base), ('one', corrected_one), ('two', corrected_two)]:
        stats[name+'_rmse'] = float(np.sqrt(np.mean((np.clip(pred, 1, 5)-y)**2)))
    stats['seconds'] = time.monotonic()-start
    stats['config'] = vars(a)
    pathlib.Path(str(prefix)+'.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
