#!/usr/bin/env python3
"""Cross-fit one weight for a candidate's correction to an OOF reference.

Candidate and candidate-base columns must be aligned to the probe. The
candidate's own unsampled rows may equal its base; a mask restricts reporting
to sampled rows. This is a development screen, not independent validation.
"""
import argparse
import hashlib
import json

import numpy as np
from scipy.optimize import minimize_scalar

from gbdt_blend import blend_fold, load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--candidate-base', required=True)
    ap.add_argument('--mask', required=True)
    a = ap.parse_args()
    d = load(a.data)
    y = d['probe']['rating'].astype(np.float64)
    reference = np.fromfile(a.reference, dtype='<f4').astype(np.float64)
    candidate = np.fromfile(a.candidate, dtype='<f4').astype(np.float64)
    base = np.fromfile(a.candidate_base, dtype='<f4').astype(np.float64)
    mask = np.fromfile(a.mask, dtype='<f4') > 0
    if any(len(x) != len(y) for x in (reference, candidate, base, mask)):
        ap.error('probe column length mismatch')
    if not all(np.isfinite(x).all() for x in (reference, candidate, base)):
        ap.error('non-finite input')
    delta = candidate-base
    if not np.any(mask) or np.any(delta[~mask]):
        ap.error('mask does not match correction support')
    fold = blend_fold(d['probe']['user'])
    output = np.empty_like(y)
    weights = []
    for f in range(5):
        train, held = fold != f, fold == f
        def mse(w):
            return np.mean((y[train]-np.clip(reference[train]+w*delta[train], 1, 5))**2)
        w = float(minimize_scalar(mse, bounds=(-1, 1), method='bounded').x)
        weights.append(w)
        output[held] = np.clip(reference[held]+w*delta[held], 1, 5)
    old = np.clip(reference, 1, 5)
    rmse = lambda x, take: float(np.sqrt(np.mean((y[take]-x[take])**2)))
    report = dict(reference_rmse=rmse(old, slice(None)), blend_rmse=rmse(output, slice(None)),
                  sampled_reference_rmse=rmse(old, mask), sampled_blend_rmse=rmse(output, mask),
                  sampled_rows=int(mask.sum()), weights=weights,
                  candidate_sha256=hashlib.sha256(np.asarray(candidate, '<f4').tobytes()).hexdigest())
    report['improvement'] = report['reference_rmse']-report['blend_rmse']
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
