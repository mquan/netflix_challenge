#!/usr/bin/env python3
"""Screen candidate prediction columns against R9-LAT-A2h's out-of-fold residual.

For each candidate column c (label-free base-model output on probe), fit
residual ~ a + b*(c - bank_mean) + g*(c - bank_mean)*support_bin on four user
folds and score the fifth (the blend's own blend_fold split). Reports the RMSE
change of control_avg + correction. It is a linear screen only: a column that
cannot move the residual here is unlikely to pay in the tree, and the batch
blend test is the real gate.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from gbdt_blend import blend_fold, load

R9 = pathlib.Path('/Users/quan/workspace/netflix_round9')
CONTROL = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-NN2.avg.f32'
BASE = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-s3.json'


def rmse(e):
    return float(np.sqrt(np.mean(e * e)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    d = load(a.data)
    q = d['probe']
    y = q['rating'].astype(np.float64)
    folds = blend_fold(q['user'])
    control = np.fromfile(CONTROL, dtype='<f4').astype(np.float64)
    resid = y - np.clip(control, 1, 5)
    base = json.loads(BASE.read_text())['config']
    bank_paths = []
    for x in base['extra']:
        for root in (R9, R9 / base['bank_root']):
            if (root / x).exists():
                bank_paths.append(root / x)
                break
    bank_hash = {hashlib.sha256(pt.read_bytes()).hexdigest() for pt in bank_paths}
    mean = np.mean([np.fromfile(pt, dtype='<f4') for pt in bank_paths], axis=0).astype(np.float64)
    n_u = np.diff(d['train_offsets'].astype(np.int64))[q['user'].astype(np.int64)]
    bins = np.digitize(np.log1p(n_u), np.log1p([20, 60, 150, 400]))
    onehot = np.eye(5)[bins]
    rows = []
    for path in a.candidates:
        raw = pathlib.Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() in bank_hash:
            continue
        c = np.frombuffer(raw, dtype='<f4').astype(np.float64)
        if len(c) != len(y) or not np.isfinite(c).all():
            continue
        z = c - mean
        x = np.column_stack([onehot, onehot * z[:, None]])
        pred = np.empty(len(y))
        for f in range(5):
            tr = folds != f
            w = np.linalg.solve(x[tr].T @ x[tr] + 10 * np.eye(x.shape[1]), x[tr].T @ resid[tr])
            pred[~tr] = x[~tr] @ w
        fold_delta = [rmse(np.clip(control[folds == f] + pred[folds == f], 1, 5) - y[folds == f])
                      - rmse(np.clip(control[folds == f], 1, 5) - y[folds == f]) for f in range(5)]
        rows.append(dict(path=path, standalone=rmse(np.clip(c, 1, 5) - y),
                         corr_with_residual=float(np.corrcoef(z, resid)[0, 1]),
                         delta=float(np.mean(fold_delta)), fold_deltas=fold_delta,
                         folds_better=int(sum(x < 0 for x in fold_delta))))
        print(json.dumps({k: rows[-1][k] for k in ('path', 'standalone', 'delta', 'folds_better')}), flush=True)
    rows.sort(key=lambda r: r['delta'])
    out.write_text(json.dumps(dict(control=str(CONTROL), bank_columns=len(bank_paths),
                                   note='Linear OOF screen on the blend folds; not a promotion test.',
                                   results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
