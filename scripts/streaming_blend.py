#!/usr/bin/env python3
"""Memory-bounded nested user-fold ridge/tree blending of local + public bases.

The contextual ridge uses exact sufficient statistics in ten disjoint user
groups (outer fold x inner membership). Neither its fit nor early stopping
reads the scored outer fold's labels. This avoids materializing the 1000-wide
float64 ridge design for all 1.4 million rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
import numpy as np
from gbdt_blend import blend_fold, load


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def design(p, c):
    z = np.column_stack([np.ones(len(p)), p.astype(np.float64)-3.6])
    return (c[:, :, None].astype(np.float64)*z[:, None, :]).reshape(len(p), -1)


def grouped_statistics(predictions, context, y, groups, chunks=8192, sample_weights=None):
    width = (predictions.shape[1]+1)*context.shape[1]
    gram = np.zeros((10, width, width), dtype=np.float64)
    rhs = np.zeros((10, width), dtype=np.float64)
    count = np.bincount(groups, weights=sample_weights, minlength=10)
    for group in range(10):
        rows = np.flatnonzero(groups == group)
        for start in range(0, len(rows), chunks):
            take = rows[start:start+chunks]
            x = design(predictions[take], context[take])
            if sample_weights is None:
                gram[group] += x.T@x
                rhs[group] += x.T@(y[take]-3.6)
            else:
                sw=sample_weights[take]
                gram[group] += x.T@(x*sw[:,None])
                rhs[group] += x.T@((y[take]-3.6)*sw)
    return gram, rhs, count


def weights(stats, fold, inner=False):
    gram, rhs, count = stats
    selected = np.arange(10)//2 != fold
    if inner: selected &= np.arange(10)%2 == 0
    a = gram[selected].sum(0)
    penalty = np.ones(a.shape[0])*count[selected].sum()*1e-5
    penalty[0] = 0
    a.flat[::len(a)+1] += penalty
    return np.linalg.solve(a, rhs[selected].sum(0))


def predict_ridge(p, context, w, chunks=8192, rows=None):
    size=len(p) if rows is None else len(rows)
    out = np.empty(size)
    for start in range(0, size, chunks):
        stop = start+chunks
        take=slice(start,stop) if rows is None else rows[start:stop]
        out[start:stop] = 3.6+design(p[take], context[take])@w
    return out


def rmse(p, y): return float(np.sqrt(np.mean((np.clip(p, 1, 5)-y)**2)))


def prepare(a, y, data):
    dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    base = Path(a.base_bank)
    if a.include_local:
        manifest = json.loads((base/'experiments/UNION-round-manifest.json').read_text())
        entries += [(str(base/k), v) for k, v in manifest['reports']['union']['inputs'].items()]
    if a.public:
        public = json.loads(Path(a.public).read_text())
        entries += [(v['path'], v['sha256']) for v in public['columns']]
    entries += [(str(Path(p).resolve()), digest(p)) for p in a.extra]
    support = ({} if a.no_local_support else
               json.loads((base/'experiments/BOOST-support-inputs.json').read_text()))
    supports = [(str(base/k), v) for k, v in support.items()]
    if a.support_manifest:
        extra_support = json.loads(Path(a.support_manifest).read_text())
        supports += [(v['path'], v['sha256']) for v in extra_support['columns']]
    if not entries: raise ValueError('empty bank')
    cache = np.load(a.meta)
    meta, context = cache['meta'], cache['context']
    data_sha = digest(a.data)
    if 'data_sha256' in cache and str(cache['data_sha256'].item()) != data_sha:
        raise ValueError('metadata cache belongs to a different dataset')
    if len(context) != len(y): raise ValueError('meta/context row mismatch')
    from round4_boost import extra_meta
    more, more_names = extra_meta(data)
    all_entries = entries+supports
    feature_path = a.output+'.x.npy'
    x = np.lib.format.open_memmap(feature_path, mode='w+', dtype='<f4',
                                 shape=(len(y), len(all_entries)+meta.shape[1]+more.shape[1]),
                                 fortran_order=True)
    names = []
    for j, (path, expected) in enumerate(all_entries):
        if digest(path) != expected: raise ValueError('input hash mismatch '+path)
        col = np.fromfile(path, dtype='<f4')
        if len(col) != len(y) or not np.isfinite(col).all(): raise ValueError(path)
        x[:, j] = col
        names.append(path)
    x[:, len(all_entries):len(all_entries)+meta.shape[1]] = meta
    x[:, len(all_entries)+meta.shape[1]:] = more
    x.flush()
    record = dict(config=vars(a), inputs=dict(all_entries), names=names,
                  columns=len(entries), support=len(supports), extra_meta_names=more_names,
                  meta_sha256=digest(a.meta),
                  data_sha256=data_sha, public_manifest_sha256=digest(a.public) if a.public else None)
    Path(a.output+'.inputs.json').write_text(json.dumps(record, indent=2)+'\n')
    return x, context, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--base-bank', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--public')
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--support-manifest')
    ap.add_argument('--no-local-support', action='store_true')
    ap.add_argument('--metric-label', default='Original Probe')
    ap.add_argument('--include-local', action='store_true')
    ap.add_argument('--output', required=True)
    ap.add_argument('--ridge-only', action='store_true')
    ap.add_argument('--rounds', type=int, default=5000)
    ap.add_argument('--seeds', nargs='+', type=int, default=[2026])
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    if Path(a.output+'.json').exists() or Path(a.output+'.inputs.json').exists():
        ap.error('output exists; choose a new name')
    d = load(a.data)
    y = d['probe']['rating'].astype(np.float64)
    users = d['probe']['user'].copy()
    outer = blend_fold(users)
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    x, context, record = prepare(a, y, d)
    del d
    p = x[:, :record['columns']]
    print(json.dumps(dict(stage='statistics', columns=p.shape[1], features=x.shape[1])), flush=True)
    start = time.monotonic()
    stats = grouped_statistics(p, context, y, outer*2+inner)
    np.savez(a.output+'.stats.npz', gram=stats[0], rhs=stats[1], count=stats[2])
    print(json.dumps(dict(stage='statistics_complete', seconds=time.monotonic()-start)), flush=True)
    out, rb = np.empty(len(y)), np.empty(len(y))
    records = []
    if not a.ridge_only: import lightgbm as lgb
    for fold in range(5):
        start = time.monotonic()
        train, held = outer != fold, outer == fold
        fit, valid = train & ~inner, train & inner
        w = weights(stats, fold)
        if a.ridge_only:
            # There is no downstream residual learner needing inner/all-row
            # predictions in this branch. Score only the held rows.
            base=predict_ridge(p,context,w,rows=np.flatnonzero(held))
            rb[held]=out[held]=base
            rec=dict(fold=fold,ridge_rmse=rmse(base,y[held]))
        else:
            iw=weights(stats,fold,True)
            base=predict_ridge(p,context,w);inner_base=predict_ridge(p,context,iw)
            rb[held]=base[held]
            rec=dict(fold=fold,ridge_rmse=rmse(base[held],y[held]))
            correction = np.zeros(held.sum()); chosen = []
            for seed in a.seeds:
                params = dict(objective='regression', learning_rate=.02, num_leaves=63,
                              min_data_in_leaf=200, feature_fraction=.8, bagging_fraction=.8,
                              bagging_freq=1, lambda_l2=10, num_threads=a.threads, verbose=-1, seed=seed)
                dt = lgb.Dataset(x[fit], y[fit]-inner_base[fit])
                dv = lgb.Dataset(x[valid], y[valid]-inner_base[valid], reference=dt)
                model = lgb.train(params, dt, a.rounds, valid_sets=[dv],
                                  callbacks=[lgb.early_stopping(100, verbose=False)])
                chosen.append(model.best_iteration)
                del dt, dv, model
                model = lgb.train(params, lgb.Dataset(x[train], y[train]-base[train]), chosen[-1])
                correction += model.predict(x[held], num_threads=a.threads)/len(a.seeds)
                del model
            out[held] = base[held]+correction
            rec.update(chosen_rounds=chosen, tree_rmse=rmse(out[held], y[held]))
        rec['seconds'] = time.monotonic()-start
        records.append(rec)
        print(json.dumps(rec), flush=True)
        out[held].astype('<f4').tofile(a.output+f'.fold{fold}.f32')
    for name, col in [('ridge', rb), ('refit', out)]:
        col.astype('<f4').tofile(a.output+'.'+name+'.f32')
        record[name+'_rmse'] = rmse(col, y)
        record[name+'_fold_rmse'] = [rmse(col[outer == k], y[outer == k]) for k in range(5)]
    record.update(folds=records, metric=a.metric_label+' nested user-fold development CV; no private-test claim',
                  target_rmse=.85725, target_reached=record['refit_rmse'] <= .85725)
    Path(a.output+'.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k.endswith('rmse') or k=='target_reached'}), flush=True)


if __name__ == '__main__': main()
