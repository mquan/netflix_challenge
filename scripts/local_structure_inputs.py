"""Shared LC2 input verification and fit-only preprocessing, without neural runtimes."""
import json
from pathlib import Path

import numpy as np

from gbdt_blend import load
from local_structure_data import FIELDS
from train_holdout import digest


def read_inputs(spec_path, context_path):
    spec, context_path = json.loads(Path(spec_path).read_text()), Path(context_path)
    report = json.loads((context_path / 'manifest.json').read_text())
    if digest(spec_path) != report['spec_sha256'] or report['data_sha256'] != spec['data_sha256']:
        raise ValueError('context/spec mismatch')
    if digest(spec['data']) != spec['data_sha256']:
        raise ValueError('P2 dataset changed')
    for name, expected in report['outputs'].items():
        if digest(context_path / name) != expected:
            raise ValueError(f'context changed: {name}')
    arrays = {name: np.load(context_path / (name + '.npy'), mmap_mode='r') for name in FIELDS}
    queries = np.load(context_path / 'queries.npy')
    d = load(spec['data'])
    n = len(d['probe'])
    if len(queries) != report['rows'] or len(np.unique(queries)) != len(queries) or np.any((queries < 0) | (queries >= n)):
        raise ValueError('invalid query indices')
    columns, support = [], []
    for destination, sources in [(columns, spec['columns']), (support, spec.get('support', []))]:
        for item in sources:
            if digest(item['path']) != item['sha256']:
                raise ValueError(f"base column changed: {item['path']}")
            values = np.fromfile(item['path'], dtype='<f4')
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError('invalid base column')
            destination.append(values[queries])
    predictions = np.column_stack(columns).astype(np.float64)
    global_features = np.column_stack(columns + support + [arrays['pooled']]).astype(np.float32)
    for name, (_, shape) in FIELDS.items():
        if arrays[name].shape != (len(queries), *shape):
            raise ValueError(f'context shape mismatch: {name}')
    if not np.isfinite(global_features).all():
        raise ValueError('nonfinite global inputs')
    return arrays, predictions, global_features, d['probe']['rating'][queries].astype(float), d['probe']['user'][queries], report


def fit_preprocessing(predictions, global_features, y, rows):
    """All supervision and normalization are restricted to the supplied fit rows."""
    design = np.column_stack([np.ones(len(y)), predictions - 3.6])
    x = design[rows]
    penalty = np.eye(x.shape[1]) * len(rows) * 1e-5
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ (y[rows] - 3.6))
    mean = global_features[rows].mean(axis=0, dtype=np.float64)
    scale = np.maximum(global_features[rows].std(axis=0, dtype=np.float64), 1e-4)
    features = np.clip((global_features - mean) / scale, -8, 8).astype(np.float32)
    return features, (3.6 + design @ weights).astype(np.float32), dict(weights=weights, mean=mean, scale=scale)


