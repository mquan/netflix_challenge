#!/usr/bin/env python3
"""Measure coverage for the predeclared recurring-persona experiment."""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gbdt_blend import load


def same_day_counts(data, field):
    probe = data['probe']
    order = np.argsort(probe['user'], kind='stable')
    counts = np.bincount(probe['user'], minlength=data['users'])
    ends = np.r_[0, counts.cumsum()]
    out = np.zeros(len(probe), dtype=np.int32)
    offsets = data[field + '_offsets']
    days = data[field + '_days']
    for user in np.flatnonzero(counts):
        rows = order[ends[user]:ends[user + 1]]
        history = days[offsets[user]:offsets[user + 1]]
        qdays = probe['day'][rows]
        out[rows] = np.searchsorted(history, qdays, 'right') - np.searchsorted(history, qdays, 'left')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--prediction', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    d = load(args.data)
    probe = d['probe']
    n = len(probe)
    pred = np.fromfile(args.prediction, dtype='<f4')
    if len(pred) != n:
        raise ValueError('prediction length mismatch')
    train_n = np.diff(d['train_offsets'])[probe['user']]
    train_day = same_day_counts(d, 'train')
    implicit_day = same_day_counts(d, 'implicit')
    err2 = (np.clip(pred, 1, 5).astype(np.float64) - probe['rating']) ** 2
    report = {'rows': n, 'reference_rmse': float(np.sqrt(err2.mean())), 'groups': {}}
    for minimum in (100, 500):
        for context, mask in [('any', np.ones(n, bool)), ('other_implicit', implicit_day > 1),
                              ('training_rating', train_day > 0)]:
            eligible = (train_n >= minimum) & mask
            p = float(eligible.mean())
            key = f'history_{minimum}_{context}'
            report['groups'][key] = {
                'rows': int(eligible.sum()), 'fraction': p,
                'reference_rmse': float(np.sqrt(err2[eligible].mean())) if p else None,
                'needed_eligible_mse_reduction_for_proxy': 0.003897 / p if p else None,
                'possible_total_mse_reduction_to_zero_error': float(err2[eligible].sum() / n),
            }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
