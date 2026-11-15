#!/usr/bin/env python3
"""Train a zero-diagonal dense residual autoencoder inspired by EASE (2019).

This is an explicit-rating adaptation, not the paper's implicit-ranking result.
Only training ratings fit the biases and Gram matrix. Query labels are read
after the prediction file is complete, solely for development scoring.
"""
import argparse
import json
import os
from pathlib import Path
import time
for name in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import numpy as np
from scipy import sparse, linalg
from gbdt_blend import load, blend_fold
from streaming_blend import digest, rmse


def ease_weights(gram, ridge):
    a = np.array(gram, dtype=np.float64, order='F', copy=True)
    a.flat[::len(a)+1] += ridge
    inverse = linalg.cho_solve(linalg.cho_factor(a, overwrite_a=True, check_finite=False),
                             np.eye(len(a)), overwrite_b=True, check_finite=False)
    inverse /= -np.diag(inverse).copy()[None, :]
    np.fill_diagonal(inverse, 0)
    return inverse


def fit_bias(d, iterations=10):
    n = np.diff(d['train_offsets']).astype(np.int64)
    owner = np.repeat(np.arange(d['users'], dtype=np.int32), n)
    items = d['train_items'].astype(np.int32)-1
    r = d['train_ratings'].astype(np.float32)-d['mean']
    counts = np.bincount(items, minlength=d['items'])
    bu = np.zeros(d['users']); bi = np.zeros(d['items'])
    nonempty = np.flatnonzero(n)
    for _ in range(iterations):
        bi = np.bincount(items, weights=r-bu[owner], minlength=d['items'])/(counts+25)
        bu[nonempty] = np.add.reduceat(r-bi[items], d['train_offsets'][nonempty].astype(np.int64))/(n[nonempty]+10)
    residual = r-bu[owner]-bi[items]
    return bu, bi, residual.astype(np.float32), owner, items, n, counts


def fit(d, limit, ridge, output):
    start = time.monotonic()
    bu, bi, residual, owner, items, n, count = fit_bias(d)
    selected = np.argsort(-count, kind='stable')[:min(limit, d['items'])]
    mapper = np.full(d['items'], -1, dtype=np.int32); mapper[selected] = np.arange(len(selected))
    target = mapper[items]; keep = target >= 0
    norm = np.sqrt(np.maximum(n, 1)).astype(np.float32)
    x = sparse.csr_matrix((residual[keep]/norm[owner[keep]], (owner[keep], target[keep])),
                          shape=(d['users'], len(selected)), dtype=np.float32)
    del residual, owner, items, target, keep
    print(json.dumps(dict(stage='csr', seconds=time.monotonic()-start, nnz=x.nnz, items=len(selected))), flush=True)
    gram = (x.T@x).toarray()
    print(json.dumps(dict(stage='gram', seconds=time.monotonic()-start)), flush=True)
    w = ease_weights(gram, ridge).astype(np.float32)
    del gram
    np.savez(output+'.model.npz', weights=w, selected=selected, user_bias=bu, item_bias=bi,
             norm=norm, ridge=ridge)
    print(json.dumps(dict(stage='inverse', seconds=time.monotonic()-start)), flush=True)
    probe = d['probe']; users = probe['user']; query_items = probe['item'].astype(np.int32)-1
    base = d['mean']+bu[users]+bi[query_items]
    correction = np.zeros(len(probe), dtype=np.float32)
    order = np.argsort(query_items, kind='stable')
    sorted_i = query_items[order]
    edges = np.r_[0, np.flatnonzero(sorted_i[1:] != sorted_i[:-1])+1, len(order)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = order[lo:hi]; j = mapper[query_items[rows[0]]]
        if j < 0: continue
        u = users[rows]
        correction[rows] = (x[u]@w[:, j])*norm[u]
    return base+correction, correction, dict(items=len(selected), covered=float(np.mean(mapper[query_items]>=0)),
                                            seconds=time.monotonic()-start)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--items', type=int, default=8192)
    ap.add_argument('--ridge', type=float, default=100)
    a = ap.parse_args()
    if a.items <= 0 or a.ridge <= 0: ap.error('positive item count and ridge required')
    if Path(a.output+'.json').exists() or Path(a.output+'.model.npz').exists(): ap.error('output exists')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    d = load(a.data)
    p, correction, report = fit(d, a.items, a.ridge, a.output)
    p.astype('<f4').tofile(a.output+'.f32'); correction.tofile(a.output+'.correction.f32')
    y = d['probe']['rating'].astype(np.float64); f = blend_fold(d['probe']['user'])
    report.update(config=vars(a), data_sha256=digest(a.data), predictions_sha256=digest(a.output+'.f32'),
                  probe_rmse=rmse(p,y), fold_rmse=[rmse(p[f==k],y[f==k]) for k in range(5)],
                  metric='Original Probe development; all Probe values excluded from fitting',
                  method='EASE-inspired dense residual autoencoder, user-normalized, zero diagonal')
    Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
