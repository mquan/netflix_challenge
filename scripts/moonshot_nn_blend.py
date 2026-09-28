#!/usr/bin/env python3
"""Nested user-fold neural residual blending over a contextual ridge control.

Only raw base-model columns are inputs. Each outer fold independently fits its
ridge, normalizer, inner epoch selection, and final neural model. Do not supply
cross-fitted ensemble columns whose training spans an outer held-out fold.
Base checkpoint selection on probe still makes these development estimates.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import time
import numpy as np
import torch
from gbdt_blend import load, blend_fold


def contexts(d):
    r = d['probe']; u, i = r['user'], r['item']
    offsets = d['implicit_offsets'].astype(np.int64)
    counts = np.diff(offsets)
    day_sum = np.zeros(d['users'], dtype=np.float64)
    nonempty = np.flatnonzero(counts)
    # Empty users contribute no entries; unique nonempty starts partition CSR.
    day_sum[nonempty] = np.add.reduceat(d['implicit_days'].astype(np.float64), offsets[nonempty])
    means = (day_sum/np.maximum(counts, 1)).astype(np.float32)
    popularity = np.bincount(d['train_items'], minlength=d['items']+1)
    same = np.empty(len(r), dtype=np.float32)
    # Probe order is by item, so gather by user before looking up session counts.
    order = np.argsort(u, kind='stable')
    sorted_u = u[order]
    starts = np.r_[0, np.flatnonzero(sorted_u[1:] != sorted_u[:-1])+1, len(r)]
    for start, end in zip(starts[:-1], starts[1:]):
        indices = order[start:end]; user = int(sorted_u[start])
        days = d['implicit_days'][offsets[user]:offsets[user+1]]
        query = r['day'][indices]
        same[indices] = np.searchsorted(days, query, 'right')-np.searchsorted(days, query, 'left')
    return np.column_stack([
        np.ones(len(r), dtype=np.float32),
        (np.log1p(np.diff(d['train_offsets'])[u].astype(np.float64))/6-1).astype(np.float32),
        (np.log1p(popularity[i].astype(np.float64))/10-1).astype(np.float32),
        (r['day'].astype(np.float32)-means[u])/1000,
        (np.log1p(same.astype(np.float64))/4-.5).astype(np.float32),
    ])


def design(predictions, context):
    centered = np.column_stack([np.ones(len(predictions)), predictions.astype(np.float64)-3.6])
    ridge = (context[:, :, None].astype(np.float64)*centered[:, None, :]).reshape(len(predictions), -1)
    mean = predictions.mean(axis=1, keepdims=True)
    neural = np.column_stack([mean-3.6, predictions-mean, context[:, 1:]]).astype(np.float32)
    return ridge, neural


def ridge_fit(x, y):
    a = x.T@x
    penalty = np.eye(x.shape[1])*1e-5*len(y); penalty[0, 0] = 0
    return np.linalg.solve(a+penalty, x.T@(y-3.6))


def fit_nn(x, y, baseline, *, hidden, epochs, seed, validation=None, patience=6):
    torch.manual_seed(seed)
    mean, std = x.mean(0, dtype=np.float64), np.maximum(x.std(0, dtype=np.float64), .01)
    tx = torch.from_numpy(((x-mean)/std).astype(np.float32))
    residual = torch.from_numpy((y-baseline).astype(np.float32))
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
    torch.nn.init.zeros_(model[-1].weight); torch.nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best_epoch, best_score, best_state = 0, float('inf'), copy.deepcopy(model.state_dict())
    if validation is not None:
        vx, vy, vb = validation
        vx = torch.from_numpy(((vx-mean)/std).astype(np.float32))
        best_score = float(np.mean((np.clip(vb, 1, 5)-vy)**2))
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(tx))
        for batch in order.split(8192):
            optimizer.zero_grad(set_to_none=True)
            error = model(tx[batch]).squeeze(1)-residual[batch]
            error.square().mean().backward(); optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                pred = np.concatenate([model(batch).squeeze(1).numpy() for batch in vx.split(65536)])+vb
            score = float(np.mean((np.clip(pred, 1, 5)-vy)**2))
            if score < best_score:
                best_epoch, best_score = epoch, score
                best_state = copy.deepcopy(model.state_dict())
            if epoch-best_epoch >= patience:
                break
        else:
            best_epoch = epoch
    if validation is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_epoch


def infer(model, mean, std, x):
    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(((x-mean)/std).astype(np.float32))
        return np.concatenate([model(batch).squeeze(1).numpy() for batch in tx.split(65536)])


def cross_fit(predictions, context, y, users, *, hidden=16, epochs=60, seed=20260928):
    torch.set_num_threads(1)
    x, nx = design(predictions, context)
    outer = blend_fold(users)
    cv, control = np.empty(len(y)), np.empty(len(y))
    records = []
    for fold in range(5):
        start = time.monotonic()
        train, held = outer != fold, outer == fold
        # All selection labels belong to outer training users.
        inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
        fit, valid = train & ~inner, train & inner
        wi = ridge_fit(x[fit], y[fit])
        _, _, _, chosen = fit_nn(nx[fit], y[fit], 3.6+x[fit]@wi,
                                hidden=hidden, epochs=epochs, seed=seed+fold,
                                validation=(nx[valid], y[valid], 3.6+x[valid]@wi))
        weights = ridge_fit(x[train], y[train])
        control[held] = 3.6+x[held]@weights
        if chosen:
            model, mean, std, _ = fit_nn(nx[train], y[train], 3.6+x[train]@weights,
                                        hidden=hidden, epochs=chosen, seed=seed+fold)
            cv[held] = control[held]+infer(model, mean, std, nx[held])
        else:
            cv[held] = control[held]
        record = dict(fold=fold, chosen_epochs=chosen, rows=int(held.sum()),
                      ridge_rmse=float(np.sqrt(np.mean((np.clip(control[held], 1, 5)-y[held])**2))),
                      neural_rmse=float(np.sqrt(np.mean((np.clip(cv[held], 1, 5)-y[held])**2))),
                      seconds=time.monotonic()-start)
        records.append(record); print(json.dumps(record), flush=True)
    return cv, control, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--prediction', action='append', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=20260928)
    a = p.parse_args()
    if a.hidden < 1 or a.epochs < 1: p.error('hidden and epochs must be positive')
    d = load(a.data); y = d['probe']['rating'].astype(np.float64)
    columns, hashes = [], {}
    for name in a.prediction:
        raw = pathlib.Path(name).read_bytes(); col = np.frombuffer(raw, dtype='<f4')
        if len(col) != len(y) or not np.isfinite(col).all(): raise ValueError(name)
        columns.append(col); hashes[name] = hashlib.sha256(raw).hexdigest()
    result, control, folds = cross_fit(np.column_stack(columns), contexts(d), y, d['probe']['user'],
                                       hidden=a.hidden, epochs=a.epochs, seed=a.seed)
    result.astype('<f4').tofile(a.output+'.f32'); control.astype('<f4').tofile(a.output+'.control.f32')
    report = dict(config=vars(a), inputs=hashes, folds=folds,
                  ridge_cv_rmse=float(np.sqrt(np.mean((np.clip(control, 1, 5)-y)**2))),
                  neural_cv_rmse=float(np.sqrt(np.mean((np.clip(result, 1, 5)-y)**2))))
    pathlib.Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
