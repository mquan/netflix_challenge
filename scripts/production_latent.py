#!/usr/bin/env python3
"""Replay retained latent columns, then export qualifying features in frozen coordinates.

The feature encoder stays bound to its original training dataset and checkpoint.
Production K1 rating predictions may use a retrained model; these blender features
must keep the original model AND projection. Query labels are never inspected.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import load
from round9_latent import digest, read_model


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'input digest mismatch: {path}')
    return actual


def same_context(original, query_data):
    """The production manifest binds dense IDs; these checks bind session indexing."""
    for name in ('users', 'items', 'min_day', 'max_day'):
        if original[name] != query_data[name]:
            raise ValueError(f'changed context: {name}')
    for name in ('implicit_offsets', 'implicit_items', 'implicit_days'):
        if not np.array_equal(original[name], query_data[name]):
            raise ValueError(f'changed context: {name}')


def features(d, model, basis, queries):
    """Original round9_latent arithmetic with a saved basis and explicit queries."""
    rank = model['item_factors'].shape[1]
    if (basis.ndim != 2 or basis.shape[0] != rank or not 0 < basis.shape[1] <= rank
            or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)):
        raise ValueError('invalid frozen projection')
    users = queries['user'].astype(np.int64)
    items = queries['item'].astype(np.int64) - 1
    days = queries['day'].astype(np.int64)
    if (np.any(users < 0) or np.any(users >= d['users']) or np.any(items < 0)
            or np.any(items >= d['items']) or np.any(days < d['min_day'])
            or np.any(days > d['max_day'])):
        raise ValueError('query outside original context')
    item = np.asarray(model['item_factors'], dtype=np.float64) @ basis
    effective = model['user_factors'] @ basis
    implicit = model['implicit_factors'] @ basis
    drift = model['drift'] @ basis
    order = np.argsort(users, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(users, minlength=d['users']))]
    sessions = np.empty(len(queries), dtype=np.int64)
    counts = np.empty(len(queries), dtype=np.int64)
    dev = np.empty(len(queries), dtype=np.float32)
    session_offset = 0
    for u in range(d['users']):
        lo, hi = map(int, d['implicit_offsets'][u:u+2])
        ids = d['implicit_items'][lo:hi].astype(np.int64) - 1
        history_days = d['implicit_days'][lo:hi]
        if len(ids):
            effective[u] += implicit[ids].sum(0) / np.sqrt(len(ids))
        unique, n = np.unique(history_days, return_counts=True)
        rows = order[offsets[u]:offsets[u+1]]
        if len(rows):
            # Require each complete user/movie/date triple, not only a known day.
            keys = history_days.astype(np.int64)*65536 + ids
            query_keys = days[rows]*65536 + items[rows]
            positions = np.searchsorted(keys, query_keys)
            if np.any(positions >= len(keys)) or not np.array_equal(keys[positions], query_keys):
                raise ValueError('query absent from original implicit interactions')
            pos = np.searchsorted(unique, days[rows])
            sessions[rows] = session_offset + pos
            counts[rows] = np.minimum(n[pos], 65535)
            delta = days[rows].astype(np.float32) - np.float32(history_days.mean())
            dev[rows] = np.copysign(np.abs(delta) ** np.float32(.4), delta)
        session_offset += len(unique)
    if session_offset != len(model['day_bias']) or session_offset != len(model['day_scale']):
        raise ValueError('checkpoint sessions differ from original context')
    freq = np.minimum(7, np.floor(np.log(np.maximum(counts, 1))/np.log(6.76)).astype(np.int64))
    timebin = np.minimum(29, (days-d['min_day'])*30//(d['max_day']-d['min_day']+1))
    out = {}
    for k in range(basis.shape[1]):
        out[f'user_pc{k}'] = effective[users, k] + drift[users, k]*dev
        out[f'item_pc{k}'] = item[items, k]
    out.update(user_bias=model['user_bias'][users], user_drift=model['user_alpha'][users]*dev,
               day_bias=model['day_bias'][sessions],
               scale=model['user_scale'][users]+model['day_scale'][sessions],
               item_bias=model['item_bias'][items]+model['item_bin_bias'][items*30+timebin],
               frequency_bias=model['item_frequency_bias'][items*8+freq])
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError('nonfinite features')
    return out


def export(reference_root, latent_manifest, data, model, production_manifest, output):
    root, manifest_path, data, model_path, production_path, output = map(
        Path, (reference_root, latent_manifest, data, model, production_manifest, output))
    if output.exists():
        raise FileExistsError(output)
    ref = json.loads(manifest_path.read_text())
    prod = json.loads(production_path.read_text())
    cfg = ref['config']
    basis_path = root / cfg['output'] / 'basis.npy'
    prod_data = production_path.parent / 'train-probe-qualifying.bin'
    inputs = {str(manifest_path): digest(manifest_path), str(production_path): digest(production_path),
              str(basis_path): digest(basis_path)}
    inputs[str(data)] = verify(data, ref['inputs'][cfg['data']])
    inputs[str(model_path)] = verify(model_path, ref['inputs'][cfg['model']])
    if prod['state'] != 'complete' or prod['source_sha256'] != inputs[str(data)]:
        raise ValueError('production manifest does not bind original dataset')
    inputs[str(prod_data)] = verify(prod_data, prod['outputs'][prod_data.name])
    d, q = load(data), load(prod_data)
    same_context(d, q)
    if len(q['probe']) != prod['qualifying_count']:
        raise ValueError('production query count differs from manifest')
    m, info = read_model(model_path, d['users'], d['items'])
    if info != ref['model']:
        raise ValueError('checkpoint header differs from retained encoder')
    basis = np.load(basis_path, allow_pickle=False)
    if basis.shape != (info['rank'], cfg['dimensions']):
        raise ValueError('projection dimensions differ from retained encoder')
    names = [Path(p).stem for p in ref['features']]
    output.mkdir(parents=True)
    stage_records = {}
    for stage, queries in (('development_replay', d['probe']), ('qualifying', q['probe'])):
        print(json.dumps(dict(stage=stage, rows=len(queries))), flush=True)
        cols = features(d, m, basis, queries)
        if list(cols) != names:
            raise ValueError('retained feature order differs')
        dest = output / stage
        dest.mkdir()
        outputs = {}
        for name, values in cols.items():
            path = dest / (name+'.f32')
            np.asarray(values, dtype='<f4').tofile(path)
            outputs[name] = dict(path=str(path.resolve()), sha256=digest(path), rows=len(values))
            if stage == 'development_replay':
                retained = ref['features'][names.index(name)]
                inputs[str(root / retained)] = verify(root / retained, ref['outputs'][retained])
                if outputs[name]['sha256'] != ref['outputs'][retained]:
                    raise ValueError(f'development feature replay differs: {name}')
        stage_records[stage] = outputs
        del cols
    for path, expected in inputs.items():
        verify(path, expected)
    report = dict(state='complete', encoder_policy='original_checkpoint_and_saved_basis',
        feature_order=names, model=info, inputs=inputs, stages=stage_records,
        sources={str(Path(__file__).resolve()): digest(__file__),
                 **{str(Path(__file__).with_name(f).resolve()): digest(Path(__file__).with_name(f))
                    for f in ('gbdt_blend.py', 'round9_latent.py')}},
        checks=['all retained development feature bytes reproduced',
                'production dataset hash and original source binding verified',
                'implicit arrays and session indexing unchanged',
                'every query triple present in original implicit context',
                'all qualifying feature rows finite and in input order',
                'frozen inputs rehashed after export'],
        limitation='Latent features only. No retrained base bank, fitted production blender, or score improvement is claimed.')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', required=True)
    p.add_argument('--latent-manifest', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--production-manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = export(**vars(a))
    print(json.dumps(dict(state=report['state'], features=len(report['feature_order']))), flush=True)


if __name__ == '__main__':
    main()
