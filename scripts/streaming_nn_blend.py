#!/usr/bin/env python3
"""NN2 recipe on a streaming_blend bank without the full dense ridge design."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from gbdt_blend import load, blend_fold
from round8_nn_blend import train, predict
from streaming_blend import digest, weights, predict_ridge, rmse


def standardized(features, fit, block=16, output=None):
    out = (np.lib.format.open_memmap(output, mode='w+', dtype='<f4',
                                    shape=features.shape, fortran_order=True)
           if output else np.empty(features.shape, dtype=np.float32))
    for start in range(0, features.shape[1], block):
        source = features[:, start:start+block].astype(np.float64)
        lo, hi = np.percentile(source[fit], [1, 99], axis=0)
        source = np.clip(source, lo, hi)
        mean, std = source[fit].mean(0), np.maximum(source[fit].std(0), 1e-6)
        out[:, start:start+block] = (source-mean)/std
    if output: out.flush()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', required=True, help='streaming tree prefix, not its .json')
    ap.add_argument('--output', required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=.2)
    ap.add_argument('--lr', type=float, default=.0003)
    ap.add_argument('--weight-decay', type=float, default=.01)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seeds', nargs='+', type=int, default=[2026, 7, 13])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--disk-features', action='store_true', help='bound wide-bank RAM with file-backed features')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    prefix = Path(a.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.output+'.json').exists(): ap.error('output already exists')
    record = json.loads(Path(a.bank+'.inputs.json').read_text())
    if digest(record['config']['data']) != record['data_sha256']: raise ValueError('dataset changed')
    if digest(record['config']['meta']) != record['meta_sha256']: raise ValueError('context changed')
    d = load(record['config']['data']); y = d['probe']['rating'].astype(np.float64)
    users = d['probe']['user'].copy(); del d
    outer = blend_fold(users)
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    raw = np.load(a.bank+'.x.npy', mmap_mode='r')
    p = raw[:, :record['columns']]
    context = np.load(record['config']['meta'])['context']
    saved = np.load(a.bank+'.stats.npz'); stats = tuple(saved[k] for k in ('gram', 'rhs', 'count'))
    shape = (len(y), raw.shape[1]+1)
    features = (np.lib.format.open_memmap(a.output+'.features.npy', mode='w+', dtype='<f4',
                                        shape=shape, fortran_order=True)
                if a.disk_features else np.empty(shape, dtype=np.float32))
    mean = p.mean(1)
    features[:, 0] = mean-3.6
    for j in range(p.shape[1]): features[:, 1+j] = p[:, j]-mean
    features[:, 1+p.shape[1]:] = raw[:, p.shape[1]:]
    out = np.empty(len(y)); records = []
    print(json.dumps(dict(stage='start', config=vars(a), columns=p.shape[1], features=features.shape[1])), flush=True)
    for fold in range(5):
        start = time.monotonic()
        tr, held = outer != fold, outer == fold
        fit, valid = tr & ~inner, tr & inner
        x = standardized(features, tr, output=a.output+'.normalized.npy' if a.disk_features else None)
        inner_base = predict_ridge(p, context, weights(stats, fold, True))
        base = predict_ridge(p, context, weights(stats, fold))
        correction = np.zeros(held.sum()); chosen = []
        for seed in a.seeds:
            _, epochs = train(x[fit], y[fit]-inner_base[fit], a, seed+fold, a.epochs,
                              valid=(x[valid], y[valid]-inner_base[valid]))
            chosen.append(epochs)
            print(json.dumps(dict(stage='selection', fold=fold, seed=seed, epochs=epochs)), flush=True)
            if epochs:
                model, _ = train(x[tr], y[tr]-base[tr], a, seed+fold, epochs)
                correction += predict(model, x[held])/len(a.seeds)
            print(json.dumps(dict(stage='refit', fold=fold, seed=seed, seconds=time.monotonic()-start)), flush=True)
        out[held] = base[held]+correction
        out[held].astype('<f4').tofile(a.output+f'.fold{fold}.f32')
        rec = dict(fold=fold, nn_rmse=rmse(out[held], y[held]),
                   ridge_rmse=rmse(base[held], y[held]), chosen_epochs=chosen,
                   seconds=time.monotonic()-start)
        records.append(rec); print(json.dumps(rec), flush=True)
        del x
    out.astype('<f4').tofile(a.output+'.nn.f32')
    report = dict(config=vars(a), bank_input_sha256=digest(a.bank+'.inputs.json'),
                  stats_sha256=digest(a.bank+'.stats.npz'), folds=records,
                  nn_rmse=rmse(out, y), nn_fold_rmse=[rmse(out[outer == k], y[outer == k]) for k in range(5)],
                  metric='Original Probe user-fold development CV; no private-test claim',
                  torch_version=torch.__version__, target_rmse=.85725)
    tree = Path(a.bank+'.refit.f32')
    if tree.exists():
        avg = (out+np.fromfile(tree, dtype='<f4'))/2
        avg.astype('<f4').tofile(a.output+'.avg.f32')
        report.update(avg_rmse=rmse(avg, y), avg_fold_rmse=[rmse(avg[outer == k], y[outer == k]) for k in range(5)])
    Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k.endswith('rmse')}), flush=True)


if __name__ == '__main__': main()
