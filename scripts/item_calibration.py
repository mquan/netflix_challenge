#!/usr/bin/env python3
"""Nested item-specific calibration of a frozen raw-model contextual blend.

Learns movie-specific residual means/slopes from the outer training users.
Inner users select shrinkage, including a zero-correction control. No fitted
OOF column is used as training input to another fold's blender.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from gbdt_blend import load, blend_fold
from streaming_blend import weights, predict_ridge, rmse, digest


def features(base, context, disagreement):
    center = base-3.6
    return np.column_stack([np.ones(len(base)), center, center**2, context[:, 1],
                            context[:, 4], center*context[:, 4], disagreement*10])


def moments(x, residual, items, mask, nitems):
    a = np.zeros((nitems, x.shape[1], x.shape[1]))
    b = np.zeros((nitems, x.shape[1]))
    xx, ii, rr = x[mask], items[mask], residual[mask]
    for j in range(x.shape[1]):
        b[:, j] = np.bincount(ii, xx[:, j]*rr, minlength=nitems)
        for k in range(j+1):
            a[:, j, k] = a[:, k, j] = np.bincount(ii, xx[:, j]*xx[:, k], minlength=nitems)
    return a, b


def fit(stats, ridge):
    a, b = stats
    return np.linalg.solve(a+ridge*np.eye(a.shape[-1]), b[..., None])[..., 0]


def apply(x, item, beta): return np.einsum('ij,ij->i', x, beta[item])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    if Path(a.output+'.json').exists(): ap.error('output exists')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(Path(a.bank+'.inputs.json').read_text())
    d = load(record['config']['data'])
    y = d['probe']['rating'].astype(np.float64)
    users = d['probe']['user'].copy(); items = d['probe']['item'].astype(np.int64)-1
    nitems = d['items']; del d
    outer = blend_fold(users)
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    raw = np.load(a.bank+'.x.npy', mmap_mode='r'); p = raw[:, :record['columns']]
    context = np.load(record['config']['meta'])['context']
    ss = np.load(a.bank+'.stats.npz'); stats = tuple(ss[k] for k in ('gram', 'rhs', 'count'))
    # Bank plan places all 79 local columns first, then the public columns.
    if not record['config']['include_local'] or p.shape[1] <= 79: raise ValueError('expected local/public union bank')
    disagreement = p[:, :79].mean(1)-p[:, 79:].mean(1)
    out, rb = np.empty(len(y)), np.empty(len(y)); records = []
    for fold in range(5):
        start = time.monotonic()
        tr, held = outer != fold, outer == fold
        fit_rows, valid = tr & ~inner, tr & inner
        ib = predict_ridge(p, context, weights(stats, fold, True))
        base = predict_ridge(p, context, weights(stats, fold))
        ix = features(ib, context, disagreement)
        ms = moments(ix, y-ib, items, fit_rows, nitems)
        candidates = [(rmse(ib[valid], y[valid]), None)]
        for ridge in (10, 100, 1000):
            beta = fit(ms, ridge)
            score = rmse(ib[valid]+apply(ix[valid], items[valid], beta), y[valid])
            candidates.append((score, ridge))
        chosen = min(candidates, key=lambda v:v[0])[1]
        out[held] = rb[held] = base[held]
        if chosen is not None:
            x = features(base, context, disagreement)
            beta = fit(moments(x, y-base, items, tr, nitems), chosen)
            out[held] += apply(x[held], items[held], beta)
        rec = dict(fold=fold, ridge_rmse=rmse(rb[held],y[held]),
                   calibrated_rmse=rmse(out[held],y[held]), selected_ridge=chosen,
                   inner_scores=candidates, seconds=time.monotonic()-start)
        records.append(rec); print(json.dumps(rec), flush=True)
    out.astype('<f4').tofile(a.output+'.f32')
    report = dict(config=vars(a), bank_sha256=digest(a.bank+'.inputs.json'), folds=records,
                  rmse=rmse(out,y), control_rmse=rmse(rb,y),
                  delta=rmse(out,y)-rmse(rb,y),
                  fold_delta=[r['calibrated_rmse']-r['ridge_rmse'] for r in records],
                  metric='Original Probe nested user-fold development, no private-test claim',
                  target_rmse=.85725)
    Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
