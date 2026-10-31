#!/usr/bin/env python3
"""Write per-training-row loss weights for an E1 Stage C weighting arm.

E1 (NEXT_EXPLORATIONS.md) retrains on all training ratings (the Phase 0 "b"
retrain) but gives the P2 rows, the stand-ins for the probe, a larger loss
weight. This writes a float32 vector, one weight per training row of
`--full` in its stored order, for `netflix joint --row-weights`.

P2 rows are the `--holdout` dataset's probe rows. Each must match exactly one
training row of `--full` on (user, item, day, rating), and the rest of
`--full`'s training rows must equal the holdout's training rows in the same
order. Any mismatch is an error. A JSON report with input/output hashes sits
beside the output.
"""
import argparse
import json
import pathlib

import numpy as np

from gbdt_blend import load
from train_holdout import digest


def build(full, holdout, weight):
    f, h = load(full), load(holdout)
    for key in ('users', 'items'):
        if f[key] != h[key]:
            raise ValueError(key)
    users = np.repeat(np.arange(f['users'], dtype=np.int64), np.diff(f['train_offsets']).astype(np.int64))
    items = np.asarray(f['train_items'], dtype=np.int64)
    key = users * (int(f['items']) + 1) + items
    p = h['probe']
    probe_key = p['user'].astype(np.int64) * (int(f['items']) + 1) + p['item'].astype(np.int64)
    order = np.argsort(key, kind='stable')
    found = np.searchsorted(key, probe_key, sorter=order)
    if np.any(found >= len(key)):
        raise ValueError('P2 row missing from full training')
    rows = order[found]
    if not np.array_equal(key[rows], probe_key):
        raise ValueError('P2 row missing from full training')
    if len(np.unique(rows)) != len(rows) or len(np.unique(key)) != len(key):
        raise ValueError('duplicate (user, item) rows')
    if not (np.array_equal(f['train_days'][rows], p['day']) and np.array_equal(f['train_ratings'][rows], p['rating'])):
        raise ValueError('P2 day or rating differs from full training')
    marked = np.zeros(len(key), dtype=bool)
    marked[rows] = True
    for name in ('train_items', 'train_days', 'train_ratings'):
        if not np.array_equal(f[name][~marked], h[name]):
            raise ValueError('remaining training rows differ from holdout: ' + name)
    weights = np.ones(len(key), dtype='<f4')
    weights[marked] = weight
    return weights, int(marked.sum())


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--full', required=True, help='dataset retrained in Stage C (all training ratings)')
    a.add_argument('--holdout', required=True, help='Phase 0 holdout dataset whose probe is P2')
    a.add_argument('--weight', type=float, required=True, help='loss weight for P2 rows (others get 1)')
    a.add_argument('--output', required=True)
    args = a.parse_args()
    if not 0 <= args.weight <= 16:
        raise SystemExit('--weight must be in [0, 16]')
    output = pathlib.Path(args.output)
    report_path = pathlib.Path(str(output) + '.json')
    if output.exists() or report_path.exists():
        raise SystemExit('choose a new output path')
    weights, count = build(args.full, args.holdout, args.weight)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights.tofile(output)
    report = dict(full=args.full, full_sha256=digest(args.full), holdout=args.holdout,
                  holdout_sha256=digest(args.holdout), weight=args.weight, weighted_rows=count,
                  train_rows=int(len(weights)), output=str(output), output_sha256=digest(output))
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
