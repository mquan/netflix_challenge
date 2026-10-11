#!/usr/bin/env python3
"""Neural residual blend over a round4_boost.py bank, averaged with its LightGBM blend.

Reads the inputs of an existing round4_boost.py report (--like): the same
prediction columns, tree features and contextual ridge. Per outer user fold
(the same blend_fold split), an MLP learns the ridge residual from the
prediction columns' mean and deviations plus every standardized tree feature.
Epochs are chosen on the same user-disjoint inner split round4_boost.py uses,
then the network is refit on all outer-training users. Several --seeds are
averaged. The output is scored alone and averaged 50/50 (predeclared) with the
--like report's cross-fitted LightGBM refit column.
"""
import argparse
import copy
import json
import pathlib
import time

import numpy as np
import torch
from gbdt_blend import load, blend_fold
from moonshot_nn_blend import design, ridge_fit
from round4_boost import column, extra_meta, sha


def bank(config, d):
    root = pathlib.Path(config['bank_root'])
    manifest = json.loads((root/'experiments/UNION-round-manifest.json').read_text())
    union = {} if config['no_union'] else manifest['reports']['union']['inputs']
    support = json.loads((root/'experiments/BOOST-support-inputs.json').read_text())
    cols, hashes = [], {}
    for name, expected in union.items():
        if sha(root/name) != expected: raise ValueError(name)
        cols.append(column(root/name)); hashes[name] = expected
    for path in config['extra']:
        cols.append(column(path)); hashes[path] = sha(path)
    tree = []
    for name, expected in support.items():
        if sha(root/name) != expected: raise ValueError(name)
        tree.append(column(root/name)); hashes[name] = expected
    for path in config['extra_support']:
        tree.append(column(path)); hashes[path] = sha(path)
    with np.load('artifacts/r4/meta-context.npz') as saved: meta, context = saved['meta'], saved['context']
    blocks = [np.column_stack(tree), meta] if tree else [meta]
    if config['extra_meta']: blocks.append(extra_meta(d)[0])
    return np.column_stack(cols), np.column_stack(blocks).astype(np.float64), context, hashes


def standardize(x, fit):
    """Robust per-feature scaling from the fit rows only; heavy tails clipped."""
    lo, hi = np.percentile(x[fit], [1, 99], axis=0)
    x = np.clip(x, lo, hi)
    mean, std = x[fit].mean(0), np.maximum(x[fit].std(0), 1e-6)
    return ((x-mean)/std).astype(np.float32)


def net(width, hidden, layers, dropout):
    mods, size = [], width
    for _ in range(layers):
        mods += [torch.nn.Linear(size, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout)]; size = hidden
    mods.append(torch.nn.Linear(size, 1))
    torch.nn.init.zeros_(mods[-1].weight); torch.nn.init.zeros_(mods[-1].bias)
    return torch.nn.Sequential(*mods)


def train(x, r, a, seed, epochs, valid=None):
    """Fit the residual r; with valid=(x, r) pick the best epoch (patience a.patience)."""
    torch.manual_seed(seed)
    model = net(x.shape[1], a.hidden, a.layers, a.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tx, tr = torch.from_numpy(x), torch.from_numpy(r.astype(np.float32))
    best, best_epoch, state = float('inf'), 0, None
    for epoch in range(1, epochs+1):
        model.train()
        for batch in torch.randperm(len(tx)).split(a.batch):
            opt.zero_grad(set_to_none=True)
            (model(tx[batch]).squeeze(1)-tr[batch]).square().mean().backward(); opt.step()
        if valid is None: continue
        score = float(np.mean((predict(model, valid[0])-valid[1])**2))
        if score < best: best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch-best_epoch >= a.patience: break
    if valid is not None: model.load_state_dict(state)
    return model, best_epoch


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(b).squeeze(1).numpy() for b in torch.from_numpy(x).split(65536)])


def rmse(p, y): return float(np.sqrt(np.mean((np.clip(p, 1, 5)-y)**2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--like', required=True, help='round4_boost.py report (.json) whose inputs and refit to use')
    ap.add_argument('--output', required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=.1)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--weight-decay', type=float, default=.01)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026])
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    if pathlib.Path(a.output+'.json').exists(): ap.error('output already exists; use a new experiment name')
    torch.set_num_threads(a.threads)
    like = json.loads(pathlib.Path(a.like).read_text())
    d = load(a.data); y = d['probe']['rating'].astype(np.float64); users = d['probe']['user']
    preds, tree, context, hashes = bank(like['config'], d)
    if set(hashes) != set(like['inputs']) or any(like['inputs'][k] != v for k, v in hashes.items()):
        raise ValueError('inputs differ from --like report')
    boost = np.fromfile(a.like[:-len('.json')]+'.refit.f32', dtype='<f4').astype(np.float64)
    rx = design(preds, context)[0]
    mean = preds.mean(1, keepdims=True)
    features = np.column_stack([mean-3.6, preds-mean, tree])
    print(json.dumps(dict(config=vars(a), prediction_columns=preds.shape[1], features=features.shape[1])), flush=True)
    outer = blend_fold(users)
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    out, records = np.empty(len(y)), []
    for fold in range(5):
        start = time.monotonic()
        tr, held = outer != fold, outer == fold
        fit, valid = tr & ~inner, tr & inner
        x = standardize(features, tr)
        inner_base = 3.6+rx@ridge_fit(rx[fit], y[fit])
        base = 3.6+rx@ridge_fit(rx[tr], y[tr])
        correction, chosen = np.zeros(held.sum()), []
        for seed in a.seeds:
            _, epochs = train(x[fit], y[fit]-inner_base[fit], a, seed+fold, a.epochs,
                              valid=(x[valid], y[valid]-inner_base[valid]))
            chosen.append(epochs)
            if epochs:
                model, _ = train(x[tr], y[tr]-base[tr], a, seed+fold, epochs)
                correction += predict(model, x[held])/len(a.seeds)
        out[held] = base[held]+correction
        rec = dict(fold=fold, chosen_epochs=chosen, seconds=time.monotonic()-start,
                   ridge_rmse=rmse(base[held], y[held]), nn_rmse=rmse(out[held], y[held]),
                   boost_rmse=rmse(boost[held], y[held]), avg_rmse=rmse((out[held]+boost[held])/2, y[held]))
        records.append(rec); print(json.dumps(rec), flush=True)
    folds = blend_fold(users)
    avg = (out+boost)/2
    report = dict(config=vars(a), inputs=hashes, folds=records, torch_version=torch.__version__)
    for name, value in [('nn', out), ('avg', avg), ('boost', boost)]:
        report[name+'_rmse'] = rmse(value, y)
        report[name+'_fold_rmse'] = [rmse(value[folds == f], y[folds == f]) for f in range(5)]
    out.astype('<f4').tofile(a.output+'.nn.f32'); avg.astype('<f4').tofile(a.output+'.avg.f32')
    pathlib.Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in report if k.endswith('rmse')}), flush=True)


if __name__ == '__main__': main()
