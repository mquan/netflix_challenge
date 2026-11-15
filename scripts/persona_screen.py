#!/usr/bin/env python3
"""Cross-fit one scalar correction weight against a frozen OOF ensemble."""
import argparse
import json
import numpy as np
from gbdt_blend import blend_fold, load


def evaluate(y, user, reference, base, candidate):
    delta = candidate.astype(np.float64)-base
    fold = blend_fold(user)
    oof = np.empty(len(y), np.float64)
    weights = []
    for k in range(5):
        fit, held = fold != k, fold == k
        # MSE-optimal coefficient for the raw additive correction, bounded to
        # prevent one noisy component dominating the existing ensemble.
        w = np.clip(np.dot(delta[fit], y[fit]-reference[fit]) /
                    max(np.dot(delta[fit], delta[fit]), 1e-12), -2., 2.)
        weights.append(float(w))
        oof[held] = np.clip(reference[held]+w*delta[held], 1, 5)
    score = lambda p, target=y: float(np.sqrt(np.mean((np.clip(p, 1, 5)-target)**2)))
    return {'reference_rmse': score(reference), 'treatment_rmse': score(oof),
            'delta_rmse': score(oof)-score(reference), 'weights': weights,
            'fold_delta': [score(oof[fold==k], y[fold==k])-score(reference[fold==k], y[fold==k])
                           for k in range(5)],
            'raw_correction_rmse': score(candidate)}


def main():
    ap = argparse.ArgumentParser()
    for key in ('data', 'reference', 'base', 'one', 'two', 'output'):
        ap.add_argument('--'+key, required=True)
    a = ap.parse_args()
    d = load(a.data)
    y = d['probe']['rating'].astype(np.float64)
    ref = np.fromfile(a.reference, dtype='<f4').astype(np.float64)
    base = np.fromfile(a.base, dtype='<f4').astype(np.float64)
    if len(ref) != len(y) or len(base) != len(y):
        raise ValueError('input length mismatch')
    report = {}
    for name in ('one', 'two'):
        candidate = np.fromfile(getattr(a, name), dtype='<f4')
        if len(candidate) != len(y) or not np.isfinite(candidate).all():
            raise ValueError(name+' invalid')
        report[name] = evaluate(y, d['probe']['user'], ref, base, candidate)
    with open(a.output, 'w') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
