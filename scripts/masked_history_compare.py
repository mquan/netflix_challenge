#!/usr/bin/env python3
"""Add completed masked-history columns to the frozen R5 residual-tree bank.

Run separately from PyTorch: this environment's Torch and LightGBM wheels use
different OpenMP runtimes. Every arm retains the reference's nested user folds,
hyperparameters, seed, and base-column ordering.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

from experiment import capture_output
from gbdt_blend import blend_fold, load


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def paired(d, control, candidate):
    y = d['probe']['rating'].astype(np.float64)
    u = d['probe']['user']
    counts = np.bincount(u)
    counts = counts[counts > 0]
    delta = (candidate - y)**2 - (control - y)**2
    sums = np.bincount(u, delta)
    active = np.bincount(u) > 0
    centered = sums[active] - counts * delta.mean()
    se = np.sqrt(len(counts) / (len(counts) - 1) * np.sum(centered**2)) / len(y)
    score = lambda p, k: float(np.sqrt(np.mean((p[k] - y[k])**2)))
    all_rows = np.ones(len(y), dtype=bool)
    f = blend_fold(u)
    n = np.diff(d['train_offsets'])[u]
    slices = {}
    for name, mask in [('under20', n < 20), ('20to49', (n >= 20) & (n < 50)),
                       ('50to499', (n >= 50) & (n < 500)), ('500plus', n >= 500)]:
        if mask.any():
            slices[name] = dict(rows=int(mask.sum()), control_rmse=score(control, mask),
                                candidate_rmse=score(candidate, mask))
    return dict(control_rmse=score(control, all_rows), candidate_rmse=score(candidate, all_rows),
                delta_rmse=score(candidate, all_rows) - score(control, all_rows),
                fold_delta_rmse=[score(candidate, f == k) - score(control, f == k) for k in range(5)],
                delta_mse=float(delta.mean()),
                descriptive_user_cluster_95_mse=[float(delta.mean() - 1.96 * se), float(delta.mean() + 1.96 * se)],
                history_slices=slices)


def run(a):
    reference = json.loads(pathlib.Path(a.reference).read_text())
    config = reference['snapshot_config']
    root = pathlib.Path(config['bank_root'])
    for name, expected in reference['inputs'].items():
        if digest(root / name) != expected:
            raise ValueError(f'reference changed: {name}')
    control_path = root / 'control.f32'
    if digest(control_path) != reference['control_sha256']:
        raise ValueError('reference predictions changed')
    d = load(config['data'])
    control = np.clip(np.fromfile(control_path, dtype='<f4').astype(np.float64), 1, 5)
    additions = []
    for path in a.prediction:
        pred = np.fromfile(path, dtype='<f4')
        if len(pred) != len(control) or not np.isfinite(pred).all():
            raise ValueError(f'invalid candidate: {path}')
        additions.append(path)
    output = pathlib.Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = pathlib.Path(str(output) + '.comparison.json')
    if report_path.exists():
        raise FileExistsError('comparison exists; use a new output')
    command = [sys.executable, str(pathlib.Path(__file__).with_name('round4_boost.py')),
               '--data', config['data'], '--bank-root', str(root), '--output', str(output),
               '--rounds', str(config['rounds']), '--learning-rate', str(config['learning_rate']),
               '--leaves', str(config['leaves']), '--min-leaf', str(config['min_leaf']),
               '--processes', str(a.processes), '--seeds', *map(str, config['seeds']),
               '--extra', *config['extra'], *additions, '--extra-support', *config['extra_support']]
    if config['extra_meta']:
        command.append('--extra-meta')
    log = pathlib.Path('experiments') / (output.name + '.log')
    events = pathlib.Path('experiments') / (output.name + '.events.jsonl')
    if log.exists() or events.exists() or pathlib.Path(str(output) + '.json').exists():
        raise FileExistsError('blend experiment already exists')
    record = dict(reference=a.reference, reference_sha256=digest(a.reference), command=command,
                  candidate_inputs={p: digest(p) for p in additions}, data_sha256=digest(config['data']),
                  sources={p: digest(pathlib.Path(__file__).with_name(p))
                           for p in ('masked_history_compare.py', 'round4_boost.py', 'gbdt_blend.py',
                                     'moonshot_nn_blend.py')})
    pathlib.Path(str(output) + '.invocation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    code = capture_output(command, log, events)
    if code:
        raise subprocess.CalledProcessError(code, command)
    predictions_path = pathlib.Path(str(output) + '.refit.f32')
    candidate = np.clip(np.fromfile(predictions_path, dtype='<f4').astype(np.float64), 1, 5)
    record.update(predictions_sha256=digest(predictions_path), comparison=paired(d, control, candidate),
                  metric='Development CV; paired interval does not adjust for repeated model selection')
    report_path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record['comparison']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', default='experiments/MH-reference.json')
    p.add_argument('--prediction', action='append', default=[], help='omit to reproduce control')
    p.add_argument('--output', required=True)
    p.add_argument('--processes', type=int, default=2)
    run(p.parse_args())
