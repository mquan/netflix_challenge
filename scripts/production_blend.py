#!/usr/bin/env python3
"""Fit the retained tree/NN blend on all development labels; persist and reload it.

No qualifying rows participate in fitting or budget selection. The caller must
provide ordered, hash-bound banks; missing production columns fail closed.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from gbdt_blend import load
from moonshot_nn_blend import design, ridge_fit
import round8_nn_blend as nn
from round9_latent import digest


def budgets(tree, neural):
    if tree['inputs'] != neural['inputs']:
        raise ValueError('tree and neural retained inputs differ')
    result = {}
    for name, report, key in (('tree', tree, 'chosen_rounds'), ('neural', neural, 'chosen_epochs')):
        folds = sorted(report['folds'], key=lambda f: f['fold'])
        if [f['fold'] for f in folds] != list(range(5)):
            raise ValueError('exactly five distinct fold records required')
        seeds = report['config']['seeds']
        values = np.asarray([f[key] if isinstance(f[key], list) else [f[key]] for f in folds])
        if values.shape != (5, len(seeds)) or np.any(values < 1) or not np.array_equal(values, values.astype(int)):
            raise ValueError('invalid selected budgets')
        result[name] = [dict(seed=int(seed), steps=int(step)) for seed, step in zip(seeds, np.median(values, axis=0))]
    return result


def read_bank(spec):
    """All matrices preserve declared column order, including duplicate aliases."""
    missing = [v['name'] for group in ('predictions', 'auxiliary', 'context') for v in spec[group] if v.get('pending')]
    if missing:
        raise ValueError(f'{len(missing)} production columns pending: {missing[:3]}')
    path = Path(spec['dataset']['path'])
    if digest(path) != spec['dataset']['sha256']:
        raise ValueError('dataset binding changed')
    d = load(path)
    rows = len(d['probe'])
    if rows != spec['rows']:
        raise ValueError('query count differs from bank')
    cache, hashes, matrices, schema = {}, {str(path): spec['dataset']['sha256']}, {}, {}
    for group in ('predictions', 'auxiliary', 'context'):
        cols = []
        schema[group] = [v['name'] for v in spec[group]]
        for entry in spec[group]:
            path = str(entry['path'])
            if path in hashes and hashes[path] != entry['sha256']:
                raise ValueError('conflicting digests for one file')
            if path not in cache:
                if digest(path) != entry['sha256']:
                    raise ValueError('column digest changed: '+path)
                cache[path] = (np.load(path, mmap_mode='r', allow_pickle=False) if path.endswith('.npy')
                               else np.memmap(path, mode='r', dtype='<f4'))
                hashes[path] = entry['sha256']
            values = cache[path]
            if 'column' in entry:
                index = entry['column']
                if values.ndim != 2 or not isinstance(index, int) or not 0 <= index < values.shape[1]:
                    raise ValueError('invalid matrix column')
                values = values[:, index]
            if values.shape != (rows,) or values.dtype != np.dtype('<f4') or not np.isfinite(values).all():
                raise ValueError('invalid float32 column: '+entry['name'])
            cols.append(values)
        if not cols:
            raise ValueError('empty bank group: '+group)
        matrices[group] = np.column_stack(cols).astype(np.float64 if group == 'auxiliary' else np.float32)
    return matrices, schema, d, hashes


def neural_features(predictions, auxiliary):
    # Keep float32 mean/deviation arithmetic before the float64 auxiliary stack,
    # matching the retained round8_nn_blend feature construction.
    mean = predictions.mean(1, keepdims=True)
    return np.column_stack([mean-3.6, predictions-mean, auxiliary])


def fit_normalizer(features):
    lo, hi = np.percentile(features, [1, 99], axis=0)
    clipped = np.clip(features, lo, hi)
    return dict(lo=lo, hi=hi, mean=clipped.mean(0), std=np.maximum(clipped.std(0), 1e-6))


def transform(features, normalizer):
    return ((np.clip(features, normalizer['lo'], normalizer['hi'])-normalizer['mean'])/normalizer['std']).astype(np.float32)


def check_versions(settings):
    if settings['lightgbm_version'] != importlib.metadata.version('lightgbm') or settings['torch_version'] != torch.__version__:
        raise ValueError('library versions differ from frozen recipe')


def tree_worker(action, matrices, settings, model_dir, *, residual=None, batch=65536):
    """One isolated process per tree stage avoids incompatible OpenMP runtimes."""
    with tempfile.TemporaryDirectory(prefix='netflix-blend-tree-') as temporary:
        root = Path(temporary)
        rows = len(matrices['predictions'])
        width = matrices['predictions'].shape[1]+matrices['auxiliary'].shape[1]
        x = np.lib.format.open_memmap(root / 'features.npy', mode='w+', dtype='<f4', shape=(rows, width))
        for start in range(0, rows, batch):
            stop = min(start+batch, rows)
            x[start:stop] = np.column_stack([matrices['predictions'][start:stop], matrices['auxiliary'][start:stop]])
        x.flush()
        del x
        (root / 'settings.json').write_text(json.dumps(settings))
        command = [sys.executable, str(Path(__file__).with_name('production_blend_tree.py')), action,
            '--features', str(root / 'features.npy'), '--settings', str(root / 'settings.json'),
            '--model-dir', str(model_dir), '--batch', str(batch)]
        if action == 'fit':
            np.save(root / 'residual.npy', residual, allow_pickle=False)
            command += ['--residual', str(root / 'residual.npy')]
        else:
            command += ['--output', str(root / 'correction.npy')]
        subprocess.run(command, check=True)
        if action == 'predict':
            return np.load(root / 'correction.npy', allow_pickle=False)


def train(matrices, labels, schema, settings, output):
    """Fit only development arrays. There is deliberately no query input argument."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    check_versions(settings)
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (len(matrices['predictions']),) or not np.all((labels >= 1) & (labels <= 5)):
        raise ValueError('full development labels required; unknown sentinels cannot train a blender')
    output.mkdir(parents=True)
    torch.set_num_threads(settings['neural_config']['threads'])
    p, aux, context = (matrices[k] for k in ('predictions', 'auxiliary', 'context'))
    rx = design(p, context)[0]
    weights = ridge_fit(rx, labels)
    base = 3.6 + rx @ weights
    np.save(output / 'ridge.npy', weights, allow_pickle=False)
    del rx
    residual = labels-base
    tree_worker('fit', matrices, settings, output, residual=residual)
    trees = [dict(file=f'tree-{i}.txt', **arm) for i, arm in enumerate(settings['budgets']['tree'])]
    features = neural_features(p, aux)
    normalizer = fit_normalizer(features)
    np.savez(output / 'normalizer.npz', **normalizer)
    x = transform(features, normalizer)
    del features
    networks = []
    args = SimpleNamespace(**settings['neural_config'])
    for i, arm in enumerate(settings['budgets']['neural']):
        model, _ = nn.train(x, residual, args, arm['seed'], arm['steps'])
        path = output / f'neural-{i}.pt'
        torch.save(model.state_dict(), path)
        networks.append(dict(file=path.name, **arm))
        print(json.dumps(dict(stage='neural', **arm)), flush=True)
    report = dict(state='complete', settings=settings, schema=schema, training_rows=len(labels),
        neural_width=x.shape[1], trees=trees, networks=networks,
        artifacts={path.name: digest(path) for path in output.iterdir() if path.is_file()},
        sources={str(Path(__file__).with_name(name).resolve()): digest(Path(__file__).with_name(name))
                 for name in ('production_blend.py', 'production_blend_tree.py', 'round8_nn_blend.py', 'moonshot_nn_blend.py', 'round4_boost.py', 'gbdt_blend.py')},
        aggregation='float32 tree prediction plus float64 neural prediction, averaged 50/50; clip final output to [1,5]',
        limitation='Full-probe fitted blender has no held-out training score. Requires independently retrained qualifying input bank; no qualifying labels or score are available.')
    (output / 'model.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def predict(matrices, schema, model_dir, batch=65536):
    model_dir = Path(model_dir)
    report = json.loads((model_dir / 'model.json').read_text())
    if report['state'] != 'complete' or report['schema'] != schema:
        raise ValueError('inference column order/schema differs from fitted model')
    if batch <= 0:
        raise ValueError('batch size must be positive')
    check_versions(report['settings'])
    for name, sha in report['artifacts'].items():
        if digest(model_dir / name) != sha:
            raise ValueError('fitted artifact changed: '+name)
    for path, sha in report['sources'].items():
        if digest(path) != sha:
            raise ValueError('inference source changed: '+path)
    torch.set_num_threads(report['settings']['neural_config']['threads'])
    weights = np.load(model_dir / 'ridge.npy', allow_pickle=False)
    with np.load(model_dir / 'normalizer.npz', allow_pickle=False) as z:
        normalizer = dict(z)
    tree_correction = tree_worker('predict', matrices, report['settings'], model_dir, batch=batch)
    cfg = report['settings']['neural_config']
    networks = []
    for arm in report['networks']:
        model = nn.net(report['neural_width'], cfg['hidden'], cfg['layers'], cfg['dropout'])
        model.load_state_dict(torch.load(model_dir / arm['file'], map_location='cpu', weights_only=True))
        model.eval()
        networks.append(model)
    rows = len(matrices['predictions'])
    out = {name: np.empty(rows, dtype='<f4') for name in ('tree', 'neural', 'average', 'predictions')}
    for start in range(0, rows, batch):
        stop = min(start+batch, rows)
        p, aux, context = (matrices[k][start:stop] for k in ('predictions', 'auxiliary', 'context'))
        base = 3.6 + design(p, context)[0] @ weights
        tree = (base+tree_correction[start:stop]).astype(np.float32)
        nx = transform(neural_features(p, aux), normalizer)
        correction = np.zeros(stop-start)
        for model in networks:
            correction += nn.predict(model, nx)/len(networks)
        neural = base+correction
        average = (tree.astype(np.float64)+neural)/2
        for name, values in (('tree', tree), ('neural', neural), ('average', average), ('predictions', np.clip(average, 1, 5))):
            out[name][start:stop] = values
    if any(not np.isfinite(values).all() for values in out.values()):
        raise ValueError('nonfinite blend output')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'fit', 'infer'])
    parser.add_argument('--bank', required=True)
    parser.add_argument('--model')
    parser.add_argument('--output')
    args = parser.parse_args()
    bank_path = Path(args.bank)
    bank_bytes = bank_path.read_bytes()
    bank = json.loads(bank_bytes)
    for path, sha in bank.get('sources', {}).items():
        if digest(path) != sha:
            raise ValueError('frozen bank source changed: '+path)
    split = 'qualifying' if args.action == 'infer' else 'development'
    matrices, schema, d, hashes = read_bank(bank[split])
    if schema != bank['schema']:
        raise ValueError('bank schema differs from declared order')
    if args.action == 'check':
        print(json.dumps(dict(rows=len(d['probe']), widths={k: v.shape[1] for k, v in matrices.items()},
            pending_qualifying=sum(bool(v.get('pending')) for k in ('predictions', 'auxiliary', 'context') for v in bank['qualifying'][k]))))
        return
    if not args.output:
        parser.error('--output required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.action == 'fit':
        train(matrices, d['probe']['rating'], schema, bank['settings'], output)
    else:
        if not args.model:
            parser.error('--model required for inference')
        model_dir = Path(args.model)
        binding = json.loads((model_dir / 'run.json').read_text())
        if binding['action'] != 'fit' or digest(model_dir / 'model.json') != binding['outputs']['model.json']:
            raise ValueError('model lacks a verified fit binding')
        result = predict(matrices, schema, args.model)
        output.mkdir(parents=True)
        for name, values in result.items():
            values.tofile(output / (name+'.f32'))
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('input changed during operation: '+path)
    if bank_path.read_bytes() != bank_bytes:
        raise ValueError('bank manifest changed during operation')
    binding = dict(action=args.action, bank=str(bank_path.resolve()), bank_sha256=digest(bank_path),
        input_hashes=hashes, rows=len(d['probe']),
        outputs={p.name: digest(p) for p in output.iterdir() if p.is_file()})
    (output / 'run.json').write_text(json.dumps(binding, indent=2)+'\n')


if __name__ == '__main__':
    main()
