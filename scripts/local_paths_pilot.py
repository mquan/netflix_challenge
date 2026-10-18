#!/usr/bin/env python3
"""Bounded, training-only three-edge retrieval and a P2 rating-signal pilot.

No pretrained inputs or residual targets. P2 user folds 2--4 fit ridge,
fold 1 selects regularization, and fold 0 evaluates a refit on folds 1--4.
This is a pilot on P2, not an original-probe ensemble evaluation.
"""
import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np

from gbdt_blend import blend_fold, load
from train_holdout import digest


class TrainingIndex:
    def __init__(self, d):
        self.d = d
        # Stable item sort preserves user order from the user-major training CSR.
        self.rows = np.argsort(d['train_items'], kind='stable').astype(np.uint32)
        owner = np.repeat(np.arange(d['users'], dtype=np.uint32),
                          np.diff(d['train_offsets']).astype(np.int64))
        self.users = owner[self.rows]
        counts = np.bincount(d['train_items'], minlength=d['items'] + 1)
        self.offsets = np.r_[0, np.cumsum(counts)]
        self.item_mean = (np.bincount(d['train_items'], d['train_ratings'],
                                      minlength=d['items'] + 1) + 25 * d['mean']) / (counts + 25)

    def item_rows(self, item):
        a, b = self.offsets[item:item + 2]
        return self.rows[a:b], self.users[a:b]

    def retrieve(self, user, item, day, *, history=32, neighbors=32,
                 candidates=256, seed=20261012):
        d = self.d
        a, b = map(int, d['train_offsets'][user:user + 2])
        rows = np.arange(a, b)
        if np.any(d['train_items'][rows] == item):
            raise ValueError('query target is present in training context')
        # Nearest dates, then source-row order; neither decision reads ratings.
        gap = np.abs(d['train_days'][rows].astype(np.int32) - day)
        rows = rows[np.argsort(gap, kind='stable')[:history]]
        target_rows, target_users = self.item_rows(item)
        rng = np.random.default_rng(np.random.SeedSequence([seed, user, item, day]))
        if len(target_rows) > candidates:
            pick = rng.choice(len(target_rows), candidates, replace=False)
            target_rows, target_users = target_rows[pick], target_users[pick]
        # All candidate intersection work is bounded by history * candidates.
        cross = np.full((len(target_rows), len(rows)), -1, dtype=np.int64)
        for j, row in enumerate(rows):
            ir, iu = self.item_rows(int(d['train_items'][row]))
            pos = np.searchsorted(iu, target_users)
            valid = np.flatnonzero(pos < len(iu))
            valid = valid[iu[pos[valid]] == target_users[valid]]
            cross[valid, j] = ir[pos[valid]]
        overlap = (cross >= 0).sum(axis=1)
        distance = np.abs(d['train_days'][target_rows].astype(np.int32) - day)
        ranked = np.lexsort((target_users, distance, -overlap))
        ranked = ranked[overlap[ranked] > 0][:neighbors]
        return dict(history_rows=rows, target_rows=target_rows[ranked],
                    cross_rows=cross[ranked], candidate_count=len(target_rows),
                    user_count=b - a, item_mean=float(self.item_mean[item]),
                    user_mean=float((d['train_ratings'][a:b].sum() + 5 * d['mean']) / (b - a + 5)))


def distribution(ratings):
    return np.bincount(ratings, minlength=6)[1:6] / max(len(ratings), 1)


def summarize(d, graph, day):
    hr, tr, cross = (graph[k] for k in ('history_rows', 'target_rows', 'cross_rows'))
    mask = cross >= 0
    v, j = np.nonzero(mask)
    ur = d['train_ratings'][hr].astype(np.float64)
    vr = d['train_ratings'][tr].astype(np.float64)
    shared = d['train_ratings'][cross[mask]].astype(np.float64)
    support = mask.sum(axis=1)
    baseline = np.clip(graph['user_mean'] + graph['item_mean'] - d['mean'], 1, 5)
    if len(v):
        difference = ur[j] - shared
        means = np.bincount(v, difference, minlength=len(tr)) / support
        mse = np.bincount(v, difference ** 2, minlength=len(tr)) / support
        weights = support / (support + 5) * np.exp(-mse / 2)
        knn = np.clip(np.dot(weights, vr + means) / weights.sum(), 1, 5)
        # Each path contributes exactly once to a rating(u,j), rating(v,j), rating(v,i) cell.
        cell = (ur[j].astype(int) - 1) * 25 + (shared.astype(int) - 1) * 5 + vr[v].astype(int) - 1
        histogram = np.bincount(cell, minlength=125) / len(v)
        agreement = [difference.mean(), np.mean(difference ** 2),
                     np.mean(difference == 0), means.std(), mse.mean(),
                     np.max(support), np.mean(support), np.std(support)]
        time_gaps = [np.mean(np.abs(d['train_days'][cross[mask]].astype(float) - day)) / 365,
                     np.mean(np.abs(d['train_days'][hr[j]].astype(float) -
                                    d['train_days'][cross[mask]].astype(float))) / 365,
                     np.mean(np.abs(d['train_days'][tr].astype(float) - day)) / 365]
    else:
        knn = baseline
        histogram, agreement, time_gaps = np.zeros(125), np.zeros(8), np.zeros(3)
    # Control contains the same retrieved ratings, pooled without triplet correspondence,
    # plus the matched simple user-kNN and all support/date summaries.
    control = np.r_[baseline, knn, graph['user_mean'], graph['item_mean'],
                    np.log1p([graph['user_count'], graph['candidate_count'], len(hr), len(tr), len(v)]),
                    distribution(ur.astype(int)), distribution(vr.astype(int)),
                    distribution(shared.astype(int)), time_gaps]
    return control, np.r_[histogram, agreement], float(knn)


def ridge_predict(x, y, fit, held, regularization):
    mean = x[fit].mean(axis=0)
    scale = x[fit].std(axis=0)
    scale[scale < 1e-8] = 1
    tx = (x[fit] - mean) / scale
    center = y[fit].mean()
    weights = np.linalg.solve(tx.T @ tx + regularization * np.eye(x.shape[1]),
                              tx.T @ (y[fit] - center))
    return np.clip(center + ((x[held] - mean) / scale) @ weights, 1, 5)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def signal_test(control, paths, y, users, knn):
    folds = blend_fold(users)
    fit, select, held = folds >= 2, folds == 1, folds == 0
    if min(fit.sum(), select.sum(), held.sum()) == 0:
        raise ValueError('pilot needs examples in fit, selection and evaluation user folds')
    scores, predictions = {}, {}
    for name, x in [('pooled_knn', control), ('pooled_knn_paths', np.column_stack([control, paths]))]:
        selection = {str(reg): rmse(y[select], ridge_predict(x, y, fit, select, reg))
                     for reg in (1, 10, 100, 1000)}
        reg = float(min(selection, key=selection.get))
        pred = ridge_predict(x, y, ~held, held, reg)
        predictions[name] = pred
        scores[name] = dict(selected_regularization=reg, selection_rmse=selection,
                            evaluation_rmse=rmse(y[held], pred))
    diff = ((y[held] - predictions['pooled_knn_paths']) ** 2 -
            (y[held] - predictions['pooled_knn']) ** 2)
    _, cluster = np.unique(users[held], return_inverse=True)
    count = np.bincount(cluster)
    sums = np.bincount(cluster, diff)
    centered = sums - diff.mean() * count
    groups = len(count)
    if groups < 2:
        raise ValueError('at least two evaluation users required for clustered uncertainty')
    se = np.sqrt(groups / (groups - 1) * np.sum(centered ** 2)) / len(diff)
    return dict(protocol='P2 user folds 2-4 fit, 1 select, 0 evaluate; refit on 1-4',
                fit_rows=int(fit.sum()), selection_rows=int(select.sum()), evaluation_rows=int(held.sum()),
                evaluation_users=groups, models=scores, matched_knn_rmse=rmse(y[held], knn[held]),
                delta_rmse=scores['pooled_knn_paths']['evaluation_rmse'] - scores['pooled_knn']['evaluation_rmse'],
                delta_mse=float(diff.mean()), cluster_normal_95_mse_interval=[float(diff.mean() - 1.96 * se),
                                                                            float(diff.mean() + 1.96 * se)]), predictions


def stats(values):
    return dict(mean=float(np.mean(values)), p50=float(np.quantile(values, .5)),
                p95=float(np.quantile(values, .95)), max=int(np.max(values)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True, help='new directory for report and sampled features')
    p.add_argument('--queries', type=int, default=20000)
    p.add_argument('--history', type=int, default=32)
    p.add_argument('--neighbors', type=int, default=32)
    p.add_argument('--candidates', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261012)
    a = p.parse_args()
    if min(a.queries, a.history, a.neighbors, a.candidates) < 1 or a.candidates < a.neighbors:
        p.error('positive caps required, with candidates >= neighbors')
    output = pathlib.Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    d = load(a.data)
    queries = np.sort(np.random.default_rng(a.seed).choice(len(d['probe']),
                       min(a.queries, len(d['probe'])), replace=False))
    print(f'building training index: {len(d["train_ratings"])} ratings', flush=True)
    index = TrainingIndex(d)
    index_seconds = time.monotonic() - started
    print(f'index_seconds={index_seconds:.2f}; extracting {len(queries)} queries', flush=True)
    controls, paths, knns, metrics = [], [], [], []
    extract_start = time.monotonic()
    for n, q in enumerate(queries):
        row = d['probe'][q]
        user, item, day = int(row['user']), int(row['item']), int(row['day'])
        graph = index.retrieve(user, item, day, history=a.history, neighbors=a.neighbors,
                               candidates=a.candidates, seed=a.seed)
        control, path, knn = summarize(d, graph, day)
        controls.append(control); paths.append(path); knns.append(knn)
        metrics.append([len(graph['history_rows']), len(graph['target_rows']),
                        int((graph['cross_rows'] >= 0).sum()), graph['candidate_count'], graph['user_count']])
        if (n + 1) % 2000 == 0:
            print(f'queries={n + 1} seconds={time.monotonic() - extract_start:.1f}', flush=True)
    extraction_seconds = time.monotonic() - extract_start
    controls, paths, knns, metrics = map(np.asarray, (controls, paths, knns, metrics))
    # Labels first enter the pipeline here, after every graph and feature is frozen.
    y = d['probe']['rating'][queries].astype(float)
    users = d['probe']['user'][queries]
    signal_start = time.monotonic()
    signal, predictions = signal_test(controls, paths, y, users, knns)
    signal_seconds = time.monotonic() - signal_start
    for label, keep in [('history_le_20', metrics[:, 4] <= 20), ('history_gt_20', metrics[:, 4] > 20)]:
        held_keep = keep[blend_fold(users) == 0]
        signal[label] = dict(evaluation_rows=int(held_keep.sum()))
        if held_keep.any():
            yy = y[blend_fold(users) == 0][held_keep]
            signal[label]['delta_rmse'] = (rmse(yy, predictions['pooled_knn_paths'][held_keep]) -
                                           rmse(yy, predictions['pooled_knn'][held_keep]))
    np.savez_compressed(output / 'sample.npz', queries=queries, users=users, control=controls,
                        paths=paths, knn=knns, metrics=metrics, **predictions)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(arguments=vars(a), data_sha256=digest(a.data),
                  source_sha256={str(path): digest(path) for path in
                                 (pathlib.Path(__file__), pathlib.Path(__file__).with_name('gbdt_blend.py'),
                                  pathlib.Path(__file__).with_name('train_holdout.py'))},
                  metric='training-only P2 pilot; no original-probe or ensemble improvement claim',
                  query_count=len(queries), full_query_count=len(d['probe']),
                  index_seconds=index_seconds, extraction_seconds=extraction_seconds,
                  signal_seconds=signal_seconds, queries_per_second=len(queries) / extraction_seconds,
                  projected_full_extraction_seconds=extraction_seconds / len(queries) * len(d['probe']),
                  peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
                  index_bytes=index.rows.nbytes + index.users.nbytes + index.offsets.nbytes + index.item_mean.nbytes,
                  path_coverage=float(np.mean(metrics[:, 2] > 0)),
                  retained_intersection_density=float(metrics[:, 2].sum() / max(np.sum(metrics[:, 0] * metrics[:, 1]), 1)),
                  history=stats(metrics[:, 0]), neighbors=stats(metrics[:, 1]), paths=stats(metrics[:, 2]),
                  candidates=stats(metrics[:, 3]), signal=signal,
                  sample_sha256=digest(output / 'sample.npz'))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
