#!/usr/bin/env python3
"""Paired user-fold development CV for frozen controls versus local GP additions.

Uses only sampled users, never scores the base-filled unsampled rows. Ridge is
fixed before seeing results. Base checkpoint selection still used probe labels.
"""
import argparse
import json
import pathlib
import numpy as np
from gbdt_blend import load, blend_fold


def evaluate(d, controls, candidates, mask, contexts):
    rows = d['probe'][mask]
    y = rows['rating'].astype(np.float64)
    folds = blend_fold(rows['user'])
    cs = [np.ones(len(y))]
    if contexts:
        nu = np.diff(d['train_offsets']).astype(np.float64)
        ni = np.bincount(d['train_items'], minlength=d['items']+1)
        cs += [np.log1p(nu[rows['user']])/6-1, np.log1p(ni[rows['item']])/10-1]
    context = np.column_stack(cs)

    def fit(cols):
        centered = np.column_stack([np.ones(len(y))]+[c[mask].astype(np.float64)-3.6 for c in cols])
        x = (context[:, :, None]*centered[:, None, :]).reshape(len(y), -1)
        cv = np.empty(len(y))
        for fold in range(5):
            train, held = folds != fold, folds == fold
            a, b = x[train], y[train]-3.6
            ridge = np.eye(x.shape[1])*1e-5*train.sum(); ridge[0, 0] = 0
            w = np.linalg.solve(a.T@a+ridge, a.T@b)
            cv[held] = 3.6+x[held]@w
        return np.clip(cv, 1, 5)

    base = fit(controls)
    records = []
    for name, candidates_for_run in [('control', [])]+[(n, [c]) for n, c in candidates]+[('all_gp', [c for n, c in candidates if not n.endswith('.3.f32')])]:
        pred = base if not candidates_for_run else fit(controls+candidates_for_run)
        delta = (pred-y)**2-(base-y)**2
        # Paired uncertainty with users as independent clusters, not rating rows.
        users, inverse = np.unique(rows['user'], return_inverse=True)
        cluster_sum = np.bincount(inverse, weights=delta)
        cluster_n = np.bincount(inverse)
        mean = delta.mean()
        se = np.sqrt(len(users)/(len(users)-1)*np.sum((cluster_sum-mean*cluster_n)**2))/len(y)
        score = lambda z: float(np.sqrt(np.mean(z)))
        records.append(dict(name=name, rmse=score((pred-y)**2), delta_mse=float(mean),
                            delta_mse_95=[float(mean-1.96*se), float(mean+1.96*se)],
                            folds=[score((pred[folds==f]-y[folds==f])**2) for f in range(5)],
                            fold_delta_mse=[float(delta[folds==f].mean()) for f in range(5)]))
    return dict(rows=len(y), users=len(users), contexts=contexts, results=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--control', action='append', required=True)
    p.add_argument('--candidate', action='append', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    d = load(a.data)
    def column(path):
        c = np.fromfile(path, dtype='<f4')
        if len(c) != len(d['probe']) or not np.isfinite(c).all():
            raise ValueError(f'invalid prediction: {path}')
        return c
    mask = column(a.mask) == 1
    controls = [column(path) for path in a.control]
    candidates = [(pathlib.Path(path).name, column(path)) for path in a.candidate]
    result = [evaluate(d, controls, candidates, mask, contexts) for contexts in (False, True)]
    pathlib.Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
