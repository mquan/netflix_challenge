#!/usr/bin/env python3
"""Screen a new probe column against an existing out-of-fold reference.

Fits one blend weight on four user folds and scores the fifth. The reference
is already out of fold, but this remains development screening because model
epochs and candidate selection may use the full probe.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from gbdt_blend import blend_fold, load


def evaluate(y, user, reference, candidate):
    fold = blend_fold(user)
    oof = np.empty_like(y, dtype=np.float64)
    weights = []
    for f in range(5):
        train, held = fold != f, fold == f
        def objective(w):
            pred = np.clip(reference[train] + w * (candidate[train] - reference[train]), 1, 5)
            return np.mean((pred - y[train]) ** 2)
        w = float(minimize_scalar(objective, bounds=(-.5, 1.5), method='bounded').x)
        weights.append(w)
        oof[held] = np.clip(reference[held] + w * (candidate[held] - reference[held]), 1, 5)
    baseline = float(np.sqrt(np.mean((np.clip(reference, 1, 5) - y) ** 2)))
    candidate_score = float(np.sqrt(np.mean((np.clip(candidate, 1, 5) - y) ** 2)))
    score = float(np.sqrt(np.mean((oof - y) ** 2)))
    return dict(reference_rmse=baseline, candidate_rmse=candidate_score,
                blend_rmse=score, improvement=baseline - score,
                weights=weights)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--candidate', required=True)
    a = ap.parse_args()
    d = load(a.data)
    y = d['probe']['rating'].astype(np.float64)
    reference = np.fromfile(a.reference, dtype='<f4')
    candidate = np.fromfile(a.candidate, dtype='<f4')
    if len(reference) != len(y) or len(candidate) != len(y):
        ap.error('prediction length does not match probe')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        ap.error('predictions contain nonfinite values')
    print(json.dumps(evaluate(y, d['probe']['user'], reference, candidate), indent=2))


if __name__ == '__main__':
    main()
