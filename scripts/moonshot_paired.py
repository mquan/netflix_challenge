#!/usr/bin/env python3
"""Paired comparisons of already cross-fitted predictions; uncertainty by user."""
import argparse
import json
import pathlib
import numpy as np
from gbdt_blend import load, blend_fold

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', required=True)
p.add_argument('--control', required=True)
p.add_argument('--candidate', action='append', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
d = load(a.data)
y = d['probe']['rating'].astype(np.float64)
u = d['probe']['user']

def read(path):
    x = np.fromfile(path, dtype='<f4').astype(np.float64)
    if len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError(f'invalid predictions: {path}')
    return np.clip(x, 1, 5)

base = read(a.control)
result = []
for path in a.candidate:
    candidate = read(path)
    for subset, keep in [('full', np.ones(len(y), dtype=bool)),
                         ('outside_pilot', (u.astype(np.uint64)*2654435761 % 10) != 0)]:
        yy, bb, cc, uu = y[keep], base[keep], candidate[keep], u[keep]
        error_diff = (cc-yy)**2-(bb-yy)**2
        users, inverse = np.unique(uu, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=error_diff)
        mean = error_diff.mean()
        se = np.sqrt(len(users)/(len(users)-1)*np.sum((sums-counts*mean)**2))/len(yy)
        bscore, cscore = np.sqrt(np.mean((bb-yy)**2)), np.sqrt(np.mean((cc-yy)**2))
        f = blend_fold(uu)
        result.append(dict(candidate=path, subset=subset, rows=len(yy), users=len(users),
                           control_rmse=float(bscore), candidate_rmse=float(cscore),
                           delta_rmse=float(cscore-bscore),
                           delta_mse=float(mean), paired_user_cluster_95_mse=[float(mean-1.96*se), float(mean+1.96*se)],
                           fold_delta_rmse=[float(np.sqrt(np.mean((cc[f==i]-yy[f==i])**2))-
                                                   np.sqrt(np.mean((bb[f==i]-yy[f==i])**2))) for i in range(5)]))
pathlib.Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
