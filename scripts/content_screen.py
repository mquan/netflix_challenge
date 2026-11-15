#!/usr/bin/env python3
"""Fixed one-weight development screen of a content feature on R9 OOF scores."""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gbdt_blend import blend_fold, load


def read(path, n):
    x = np.fromfile(path, dtype='<f4')
    if len(x) != n or not np.isfinite(x).all():
        raise ValueError(f'bad column: {path}')
    return x.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, required=True)
    ap.add_argument('--baseline', type=pathlib.Path, required=True)
    ap.add_argument('--feature', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = load(args.data)
    y = d['probe']['rating'].astype(np.float64)
    n = len(y)
    base, x = read(args.baseline, n), read(args.feature, n)
    folds = blend_fold(d['probe']['user'])
    pred = np.empty(n, dtype=np.float64)
    records = []
    for fold in range(5):
        fit, held = folds != fold, folds == fold
        # Fixed 1% relative shrink; no per-fold or probe-tuned hyperparameter.
        dot = np.dot(x[fit], y[fit] - base[fit])
        norm = np.dot(x[fit], x[fit])
        weight = dot / (1.01 * norm) if norm else 0.
        pred[held] = np.clip(base[held] + weight*x[held], 1, 5)
        old = np.sqrt(np.mean((y[held]-np.clip(base[held], 1, 5))**2))
        new = np.sqrt(np.mean((y[held]-pred[held])**2))
        records.append(dict(fold=fold, rows=int(held.sum()), weight=float(weight),
                            baseline_rmse=float(old), treatment_rmse=float(new),
                            gain=float(old-new)))
    old = np.sqrt(np.mean((y-np.clip(base, 1, 5))**2))
    new = np.sqrt(np.mean((y-pred)**2))
    result = dict(kind='diagnostic_not_promotion', baseline_rmse=float(old),
                  treatment_rmse=float(new), gain=float(old-new), folds=records,
                  gate_pass=bool(old-new >= 0.0001 and sum(r['gain'] > 0 for r in records) >= 4),
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.baseline, args.feature)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
