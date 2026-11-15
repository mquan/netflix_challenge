#!/usr/bin/env python3
"""Paired user-cluster comparison of a candidate to the frozen R9 control."""
import argparse
import json
import numpy as np
from gbdt_blend import blend_fold, load


def compare(data, control, candidate):
    d = load(data)
    probe = d['probe']
    y = probe['rating'].astype(np.float64)
    a = np.fromfile(control, dtype='<f4').astype(np.float64)
    b = np.fromfile(candidate, dtype='<f4').astype(np.float64)
    if len(a) != len(y) or len(b) != len(y) or not np.isfinite(b).all():
        raise ValueError('invalid prediction column')
    ae = (np.clip(a, 1, 5)-y)**2
    be = (np.clip(b, 1, 5)-y)**2
    delta = be-ae
    user_sum = np.bincount(probe['user'], delta, minlength=d['users'])
    active = np.bincount(probe['user'], minlength=d['users']) > 0
    sums = user_sum[active]
    se = np.sqrt(len(sums)/(len(sums)-1)*np.sum((sums-sums.mean())**2))/len(y)
    fold = blend_fold(probe['user'])
    score = lambda z: float(np.sqrt(z.mean()))
    out = {'control_rmse': score(ae), 'candidate_rmse': score(be),
           'delta_rmse': score(be)-score(ae), 'delta_mse': float(delta.mean()),
           'cluster_se_mse': float(se), 'cluster_95ci_mse': [float(delta.mean()-1.96*se),
                                                           float(delta.mean()+1.96*se)],
           'fold_delta_rmse': [score(be[fold==k])-score(ae[fold==k]) for k in range(5)]}
    out['promote_all_folds'] = all(x < 0 for x in out['fold_delta_rmse'])
    return out


def main():
    ap = argparse.ArgumentParser()
    for key in ('data', 'control', 'candidate', 'output'):
        ap.add_argument('--'+key, required=True)
    a = ap.parse_args()
    result = compare(a.data, a.control, a.candidate)
    with open(a.output, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
