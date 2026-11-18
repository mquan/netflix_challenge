#!/usr/bin/env python3
"""Fixed, nested residual-blend comparisons on the receipt-verified fresh bank."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_repro as u
import upstream_exchange as x
import upstream_expansion as e
import upstream_snapshot_blend as b

REPORT = u.ROOT/'experiments/upstream-nonlinear'
RUN = u.ROOT/'artifacts/upstream-nonlinear'
CONFIG = dict(seeds=[2026, 7, 13], threads=2, tree_rounds=2000,
              tree_patience=75, tree_leaves=31, tree_min_leaf=1000,
              tree_lr=.03, nn_epochs=40, nn_patience=4, nn_hidden=64,
              nn_batch=8192, nn_lr=.0003, nn_dropout=.2,
              nn_weight_decay=.01, ridge_lambda=1000)


def event(**values):
    print(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), **values)), flush=True)


def grouped_stats(z, y, groups):
    grams = np.zeros((10, z.shape[1], z.shape[1]))
    rhs = np.zeros((10, z.shape[1]))
    for lo in range(0, len(y), 8192):
        hi = min(lo+8192, len(y)); block = np.asarray(z[lo:hi], dtype=np.float64)
        for g in range(10):
            keep = groups[lo:hi] == g; a = block[keep]
            grams[g] += a.T@a; rhs[g] += a.T@y[lo:hi][keep]
    return grams, rhs


def ridge_weights(grams, rhs, selected, penalty=1000):
    reg = np.eye(rhs.shape[1])*penalty; reg[-1, -1] = 0
    # Sum only eligible groups: excluded labels cannot affect rounding either.
    return np.linalg.solve(grams[selected].sum(0)+reg, rhs[selected].sum(0))


def scaled(features, fit, scaler=None):
    if scaler is None:
        lo, hi = np.percentile(features[fit], [1, 99], axis=0)
        clipped = np.clip(features[fit], lo, hi).astype(np.float64)
        scaler = dict(lo=lo, hi=hi, mean=clipped.mean(0), std=np.maximum(clipped.std(0), 1e-6))
    return np.asarray((np.clip(features, scaler['lo'], scaler['hi'])-scaler['mean'])/
                      scaler['std'], dtype=np.float32, order='C'), scaler


def train_nn(features, residual, cfg, seed, epochs, valid=None):
    # Imported only in the NN subprocess; LightGBM has its own process/runtime.
    import torch
    torch.set_num_threads(cfg['threads']); torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(features.shape[1], cfg['nn_hidden']),
        torch.nn.GELU(), torch.nn.Dropout(cfg['nn_dropout']),
        torch.nn.Linear(cfg['nn_hidden'], cfg['nn_hidden']), torch.nn.GELU(),
        torch.nn.Dropout(cfg['nn_dropout']), torch.nn.Linear(cfg['nn_hidden'], 1))
    torch.nn.init.zeros_(model[-1].weight); torch.nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['nn_lr'], weight_decay=cfg['nn_weight_decay'])
    tx = torch.from_numpy(np.ascontiguousarray(features))
    ty = torch.from_numpy(np.asarray(residual, dtype=np.float32))
    best, chosen = (float(np.mean(valid[1]**2)), 0) if valid is not None else (None, epochs)
    for epoch in range(1, epochs+1):
        model.train()
        for ix in torch.randperm(len(tx)).split(cfg['nn_batch']):
            optimizer.zero_grad(set_to_none=True)
            (model(tx[ix]).squeeze(1)-ty[ix]).square().mean().backward()
            optimizer.step()
        if valid is not None:
            loss = float(np.mean((nn_predict(model, valid[0])-valid[1])**2))
            if loss < best: best, chosen = loss, epoch
            event(stage='nn_epoch', seed=seed, epoch=epoch, inner_mse=loss, chosen=chosen)
            if epoch-chosen >= cfg['nn_patience']: break
    # Selection returns a budget only; refitting always creates a fresh model.
    return model, chosen


def nn_predict(model, features):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(block))).squeeze(1).numpy()
                               for block in np.array_split(features, max(1, (len(features)+65535)//65536))])


def fit_fold(features, z, y, groups, grams, rhs, fold, cfg, family, output):
    """Every supervised operation excludes the outer scoring fold."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    outer = groups//2; training, held = outer != fold, outer == fold
    fit, valid = training & (groups%2 == 0), training & (groups%2 == 1)
    if not all(np.any(v) for v in (training, held, fit, valid)):
        raise ValueError('Empty outer/inner user partition')
    gi = np.arange(10); eligible = gi//2 != fold
    iw = ridge_weights(grams, rhs, eligible & (gi%2 == 0), cfg['ridge_lambda'])
    fw = ridge_weights(grams, rhs, eligible, cfg['ridge_lambda'])
    inner_base = np.clip(z@iw, 1.05, 4.95)
    base = np.clip(z@fw, 1.05, 4.95)
    np.savez(output/'ridge.npz', inner=iw, full=fw)
    inner_x = np.column_stack([features, inner_base]).astype(np.float32)
    full_x = np.column_stack([features, base]).astype(np.float32)
    if family == 'nn':
        inner_x, inner_scaler = scaled(inner_x, fit)
        full_x, full_scaler = scaled(full_x, training)
        np.savez(output/'scalers.npz', **{'inner_'+k:v for k,v in inner_scaler.items()},
                 **{'full_'+k:v for k,v in full_scaler.items()})
    correction = np.zeros(held.sum(), dtype=np.float64); chosen = []
    for seed in cfg['seeds']:
        actual_seed = seed+fold
        event(stage='select', family=family, fold=fold, seed=seed)
        if family == 'tree':
            import lightgbm as lgb
            params = dict(objective='regression', metric='l2', learning_rate=cfg['tree_lr'],
                num_leaves=cfg['tree_leaves'], min_data_in_leaf=cfg['tree_min_leaf'],
                feature_fraction=.9, bagging_fraction=.8, bagging_freq=5,
                seed=actual_seed, num_threads=cfg['threads'], verbosity=-1,
                deterministic=True, force_col_wise=True)
            select = lgb.train(params, lgb.Dataset(inner_x[fit], label=y[fit]-inner_base[fit]),
                num_boost_round=cfg['tree_rounds'],
                valid_sets=[lgb.Dataset(inner_x[valid], label=y[valid]-inner_base[valid])],
                callbacks=[lgb.early_stopping(cfg['tree_patience'], verbose=False)])
            count = select.best_iteration if select.best_score['valid_0']['l2'] < np.mean((y[valid]-inner_base[valid])**2) else 0
            del select
            if count:
                model = lgb.train(params, lgb.Dataset(full_x[training], label=y[training]-base[training]),
                                  num_boost_round=count)
                correction += model.predict(full_x[held], num_threads=cfg['threads'])/len(cfg['seeds'])
                model.save_model(str(output/f'tree-{seed}.txt'))
                del model
        elif family == 'nn':
            import torch
            _, count = train_nn(inner_x[fit], y[fit]-inner_base[fit], cfg, actual_seed,
                cfg['nn_epochs'], valid=(inner_x[valid], y[valid]-inner_base[valid]))
            if count:
                model, _ = train_nn(full_x[training], y[training]-base[training], cfg, actual_seed, count)
                correction += nn_predict(model, full_x[held])/len(cfg['seeds'])
                torch.save(dict(state=model.state_dict(), config=cfg, feature_width=full_x.shape[1],
                                seed=actual_seed, epochs=count), output/f'nn-{seed}.pt')
                del model
        else:
            raise ValueError('Unknown family')
        chosen.append(count)
        event(stage='refit_complete', family=family, fold=fold, seed=seed, iterations=count)
    result = np.clip(base[held]+correction, 1.05, 4.95).astype('<f4')
    result.tofile(output/'prediction.f32')
    return dict(family=family, fold=fold, chosen_iterations=chosen, rows=int(held.sum()),
                rmse=x.metric(result, y[held]), baseline_rmse=x.metric(base[held], y[held]))


def code_hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),
        u.ROOT/'scripts/upstream_snapshot_blend.py', u.ROOT/'scripts/upstream_exchange.py',
        u.ROOT/'scripts/upstream_expansion.py', u.ROOT/'scripts/upstream_repro.py']}


def versions():
    return {name:importlib.metadata.version(name) for name in ['numpy', 'torch', 'lightgbm']}


def verify_prepared():
    manifest = u.read(REPORT/'manifest.json')
    if manifest['config'] != CONFIG or manifest['code_sha256'] != code_hashes() or manifest['versions'] != versions():
        raise ValueError('Frozen recipe/code/environment changed; use a separate experiment')
    for name, digest in manifest['prepared_sha256'].items():
        if u.sha(RUN/name) != digest: raise ValueError('Prepared input changed: '+name)
    for name, digest in manifest['source_manifests_sha256'].items():
        if u.sha(u.ROOT/name) != digest: raise ValueError('Source manifest changed: '+name)
    return manifest


def prepare():
    if (REPORT/'manifest.json').exists(): return verify_prepared()
    if RUN.exists(): raise ValueError('Partial preparation exists; preserve and inspect before recovery')
    event(stage='verify_fresh_bank')
    u.validate_fresh(); labels, users, fresh = e.collect_expansion()
    frozen = u.read(b.REPORT/'manifest.json')
    current = b.freeze(labels, users, fresh)
    if any(current[k] != frozen[k] for k in current): raise ValueError('Snapshot bank changed')
    labels = labels.astype(np.float64)
    for name, digest in frozen['context_inputs'].items():
        if u.sha(u.ROOT/name) != digest: raise ValueError('Context input changed')
    context = b.context_features(*[np.load(u.ROOT/p, mmap_mode='r', allow_pickle=False)
                                  for p in frozen['context_inputs']])
    if x.array_sha(context) != frozen['context_content_sha256']: raise ValueError('Context changed')
    name = 'rbmx2-150__knnf'; receipt_path = u.BASE/'fresh-receipts'/(name+'.json')
    receipt = u.read(receipt_path)
    if receipt['source_commit'] != u.PIN or receipt['job'] != name: raise ValueError('Incorrect fresh replay receipt')
    for path, digest in receipt['outputs'].items():
        if u.sha(u.FRESH/path) != digest: raise ValueError('Fresh replay output changed')
    replay_path = u.ROOT/'experiments/upstream-missing-screen/replay/evaluation.json'
    replay = u.read(replay_path)
    if u.sha(receipt_path) != replay['native_receipt_sha256']: raise ValueError('Fresh replay provenance changed')
    rows = [r for r in frozen['entries'] if r['duplicate_of'] is None]
    arrays = [fresh[r['name']] for r in rows]+[np.load(u.FRESH/'preds_new'/(name+'.probex.npy'), allow_pickle=False)]
    clip = [r['clip'] for r in rows]+[True]
    if len(arrays) != 62: raise ValueError('Unexpected fresh bank width')
    RUN.mkdir(parents=True)
    raw = np.column_stack(arrays)
    raw[:, clip] = np.clip(raw[:, clip], 0, 6)
    mean = raw[:, clip].mean(axis=1, keepdims=True)
    nonlin = raw.copy(); nonlin[:, clip] -= mean
    np.save(RUN/'features.npy', np.column_stack([mean-3.6, nonlin, context]).astype(np.float32))
    z = np.lib.format.open_memmap(RUN/'design.npy', mode='w+', dtype=np.float64, shape=(len(labels), 138))
    for lo in range(0, len(labels), 8192):
        hi = min(lo+8192, len(labels))
        z[lo:hi] = b.design(raw[lo:hi], clip, context[lo:hi], tuple(range(14)))
    z.flush(); del raw, nonlin, mean
    folds = x.user_folds(users)
    inner = x.user_folds(users.astype(np.uint64)*np.uint64(7)+np.uint64(19)) == 0
    groups = folds*2+inner.astype(np.int8)
    grams, rhs = grouped_stats(z, labels, groups)
    np.savez(RUN/'stats.npz', gram=grams, rhs=rhs)
    baseline = np.empty(len(labels))
    for fold in range(5):
        w = ridge_weights(grams, rhs, np.arange(10)//2 != fold)
        held = folds == fold; baseline[held] = np.clip(z[held]@w, 1.05, 4.95)
    score = x.metric(baseline, labels)
    if abs(score-0.866286461848646) > 1e-9: raise ValueError('Latest fresh control does not reproduce: '+str(score))
    for name, value in [('labels', labels), ('users', users), ('groups', groups), ('baseline', baseline)]:
        np.save(RUN/(name+'.npy'), value)
    source_paths = [b.REPORT/'manifest.json', replay_path, receipt_path]
    manifest = dict(config=CONFIG, source_commit=u.PIN, versions=versions(), code_sha256=code_hashes(),
        source_manifests_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in source_paths},
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        baseline_rmse=score, rows=len(labels), prediction_columns=62, design_columns=138,
        isolation='Outer user labels excluded from ridge, inner stopping, scaling, nonlinear refits. Inner ridge/scaler fit inner-training users only. Final refit uses all outer-training users; no globally cross-fitted residual targets.',
        scope='Fresh-only exploratory ProbeX development comparison; historically selected base recipes. No downloaded prediction features or qualifying outcomes.',
        arms=['control', 'tree', 'nn', 'fixed-half-tree-nn'])
    u.write(REPORT/'manifest.json', manifest)
    event(stage='control_reproduced', rmse=score)
    return manifest


def worker(family):
    verify_prepared(); event(stage='worker_start', family=family)
    features = np.load(RUN/'features.npy', mmap_mode='r')
    z = np.load(RUN/'design.npy', mmap_mode='r')
    y, groups = [np.load(RUN/(n+'.npy'), allow_pickle=False) for n in ['labels', 'groups']]
    with np.load(RUN/'stats.npz') as stats: grams, rhs = stats['gram'], stats['rhs']
    for fold in range(5):
        root = RUN/family/str(fold); receipt_path = REPORT/family/(str(fold)+'.json')
        if receipt_path.exists():
            receipt = u.read(receipt_path)
            if receipt['manifest_sha256'] != u.sha(REPORT/'manifest.json'): raise ValueError('Stage recipe changed')
            for name, digest in receipt['outputs'].items():
                if u.sha(root/name) != digest: raise ValueError('Completed fold output changed')
            event(stage='verified_resume', family=family, fold=fold); continue
        start = time.monotonic()
        record = fit_fold(features, z, y, groups, grams, rhs, fold, CONFIG, family, root)
        record.update(seconds=time.monotonic()-start, manifest_sha256=u.sha(REPORT/'manifest.json'),
                      outputs={p.name:u.sha(p) for p in root.iterdir() if p.is_file()})
        u.write(receipt_path, record); event(stage='fold_complete', **record)


def summarize():
    manifest = verify_prepared()
    y, groups, baseline = [np.load(RUN/(n+'.npy')) for n in ['labels', 'groups', 'baseline']]
    folds = groups//2; predictions = dict(control=baseline)
    for family in ['tree', 'nn']:
        value = np.empty(len(y))
        for fold in range(5):
            receipt = u.read(REPORT/family/(str(fold)+'.json'))
            path = RUN/family/str(fold)/'prediction.f32'
            if receipt['manifest_sha256'] != u.sha(REPORT/'manifest.json') or u.sha(path) != receipt['outputs']['prediction.f32']:
                raise ValueError('Fold receipt changed')
            part = np.fromfile(path, dtype='<f4'); held = folds == fold
            if part.shape != (held.sum(),) or not np.isfinite(part).all(): raise ValueError('Malformed fold predictions')
            value[held] = part
        predictions[family] = value
    predictions['fixed-half-tree-nn'] = (predictions['tree']+predictions['nn'])/2
    records = []
    for arm, value in predictions.items():
        dest = RUN/(arm+'.f32'); value.astype('<f4').tofile(dest)
        stored = np.fromfile(dest, dtype='<f4')
        score = x.metric(stored, y)
        record = dict(arm=arm, rmse=score, delta_vs_control=score-manifest['baseline_rmse'],
            fold_rmse=[x.metric(stored[folds==f], y[folds==f]) for f in range(5)],
            fold_delta=[x.metric(stored[folds==f], y[folds==f])-x.metric(baseline[folds==f], y[folds==f]) for f in range(5)],
            prediction_sha256=u.sha(dest))
        records.append(record)
    report = dict(state='complete', manifest_sha256=u.sha(REPORT/'manifest.json'), records=records,
                  scope=manifest['scope'], isolation=manifest['isolation'])
    u.write(REPORT/'evaluation.json', report); event(stage='complete', records=records)
    return report


def run():
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(); children = []; streams = []
        try:
            for family in ['tree', 'nn']:
                log = (REPORT/(family+'.log')).open('a'); streams.append(log)
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', family],
                    cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
            while any(p.poll() is None for p in children):
                if any(p.poll() not in [None, 0] for p in children): raise RuntimeError('Worker failed; see family logs')
                u.write(REPORT/'status.json', dict(state='running', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    workers={family:dict(pid=p.pid, returncode=p.poll()) for family,p in zip(['tree','nn'], children)}))
                time.sleep(30)
            if any(p.returncode for p in children): raise RuntimeError('Worker failed; see family logs')
            summarize(); u.write(REPORT/'status.json', dict(state='complete'))
        except BaseException as error:
            for p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p in children:
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
            u.write(REPORT/'status.json', dict(state='failed', error=repr(error)))
            raise
        finally:
            for stream in streams: stream.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--worker', choices=['tree', 'nn'])
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    os.nice(10)
    if args.worker: worker(args.worker)
    elif args.prepare: prepare()
    elif args.summarize: summarize()
    else: run()
