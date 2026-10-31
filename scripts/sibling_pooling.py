#!/usr/bin/env python3
"""H6: sibling residual pooling rehearsal (HEADROOM.md).

In the real submission, probe and qualifying ratings are a random split of the
same held-out pool, so a qualifying row's user has *labelled sibling* probe
rows from the same period. Dev-CV models never trained on the probe, so those
siblings' residuals are honest. This rehearses using them.

1. Split each user's original-probe rows at random (fixed seed) into half A
   (stand-in for the labelled probe) and half B (stand-in for qualifying).
2. Feature for a row: pooled dev residual of the user's A rows (sum over n+k,
   for k in SHRINK, an indicator for n>0, log1p n), plus the same pooled at
   user-day and user-week level. For A rows the target is left out.
3. Fit the ridge coefficients on A rows only (leave-one-out features within
   A). No B label is read for fitting.
4. Apply to B with features pooled from all A siblings. Report B's RMSE
   change versus the unchanged base, overall and by user fold.

The base is a dev-CV prediction (out of fold by user, trained without the
probe), so A residuals are honest. This measures the pre-Stage C effect only.
The interaction with retraining on the probe is a separate rehearsal.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from train_holdout import digest

SHRINK = (1, 5, 25)
BIG = 1 << 20


def pooled(keys_a, e_a, keys_query, leave_out=None):
    """Pool e over A rows sharing each query key; optionally drop the query's own A value."""
    order = np.argsort(keys_a, kind='stable')
    ks, es = keys_a[order], e_a[order]
    uniq, start = np.unique(ks, return_index=True)
    sums = np.add.reduceat(es, start) if len(es) else np.zeros(0)
    counts = np.diff(np.append(start, len(ks)))
    pos = np.searchsorted(uniq, keys_query)
    found = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == keys_query)
    s = np.where(found, sums[np.minimum(pos, len(uniq) - 1)], 0.)
    n = np.where(found, counts[np.minimum(pos, len(uniq) - 1)], 0).astype(float)
    if leave_out is not None:
        s, n = s - leave_out, n - 1
    return np.column_stack([s / (n + k) for k in SHRINK] + [(n > 0).astype(float), np.log1p(np.maximum(n, 0))])


def design(p, base, e, a_mask, query_mask, leave_out):
    u, d = p['user'].astype(np.int64), p['day'].astype(np.int64)
    cols = [np.ones(query_mask.sum()), base[query_mask]]
    for keys in (u, u * BIG + d // 7, u * BIG + d):
        cols.append(pooled(keys[a_mask], e[a_mask], keys[query_mask], e[query_mask] if leave_out else None))
    return np.column_stack(cols)


def run(data_path, base_path, output, seed=20261031, penalty=10.):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = load(data_path)
    p = data['probe']
    y = p['rating'].astype(float)
    base = np.fromfile(base_path, dtype='<f4').astype(float)
    if base.shape != y.shape:
        raise ValueError('base does not match probe rows')
    a_mask = np.random.default_rng(seed).random(len(y)) < .5
    b_mask = ~a_mask
    e = y - base  # honest for every row: dev-CV base never trained on probe labels
    xa = design(p, base, e, a_mask, a_mask, leave_out=True)
    w = np.linalg.solve(xa.T @ xa + penalty * np.eye(xa.shape[1]), xa.T @ y[a_mask])
    xb = design(p, base, e, a_mask, b_mask, leave_out=False)
    corrected = xb @ w
    # Control: the same ridge without pooled features (intercept + base), fit on A.
    xa0, xb0 = xa[:, :2], xb[:, :2]
    w0 = np.linalg.solve(xa0.T @ xa0 + penalty * np.eye(2), xa0.T @ y[a_mask])
    control = xb0 @ w0
    stats = paired(y[b_mask], control, corrected, p['user'][b_mask])
    raw = paired(y[b_mask], base[b_mask], corrected, p['user'][b_mask])
    report = dict(data=str(data_path), base=str(base_path), base_sha256=digest(base_path), seed=seed,
                  a_rows=int(a_mask.sum()), b_rows=int(b_mask.sum()), penalty=penalty,
                  b_vs_recalibrated_base=stats, b_vs_raw_base=raw, coefficients=w.tolist(),
                  metric='Half-B RMSE change from pooled honest residuals of the user\'s half-A probe siblings; '
                         'coefficients fit on half A only. Pre-Stage C; diagnostic use of original-probe labels.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(delta_vs_recalibrated=stats['delta_rmse'], delta_vs_raw=raw['delta_rmse'],
                          b_base_rmse=raw['control_rmse'], b_corrected_rmse=raw['treatment_rmse'],
                          fold_deltas=stats['fold_deltas'], ci_mse=stats['cluster_normal_95_mse_interval']), indent=1))
    return report


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', default='data/processed/netflix.bin')
    a.add_argument('--base', required=True, help='dev-CV prediction of the original probe (float32)')
    a.add_argument('--output', required=True)
    a.add_argument('--seed', type=int, default=20261031)
    args = a.parse_args()
    run(args.data, args.base, args.output, args.seed)


if __name__ == '__main__':
    main()
