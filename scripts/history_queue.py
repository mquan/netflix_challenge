#!/usr/bin/env python3
"""Run the three authorized history experiments sequentially, with frozen control.

Launch from repository root with the project Python. Stops on the first failure.
Status is written atomically to experiments/history-queue-status.json.
"""
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from experiment import digest
from gbdt_blend import load, blend_fold

BANK = ['legacy.a', 'legacy.b', 'legacy.c', 'joint64', 'joint128-reg005',
        'joint128-reg02', 'joint128-day128', 'neighborhood64', 'joint256-reg01',
        'neighborhood128-interim1']
STATUS = pathlib.Path('experiments/history-queue-status.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(state):
    state['updated_utc'] = now()
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATUS)


def run(name, strategy, command):
    subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                    '--strategy', strategy, '--', *command], check=True)


def ensemble(name, paths):
    command = ['build/netflix', 'ensemble', '--data', 'data/processed/netflix.bin',
               '--contexts', '1', '--quadratic', '1', '--predictions', f'artifacts/history/{name}.f32',
               '--weights', f'artifacts/history/{name}.weights']
    for path in paths:
        command += ['--prediction', path]
    run(name, 'Frozen local B7 bank, original five user folds; development CV only', command)


def assess(name, data, control):
    path = f'artifacts/history/{name}-blend.f32'
    pred = np.fromfile(path, dtype='<f4').astype(np.float64)
    if pred.shape != control.shape or not np.isfinite(pred).all():
        raise ValueError('invalid completed ensemble output')
    y = data['probe']['rating'].astype(np.float64)
    user = data['probe']['user'].astype(np.int64)
    support = np.diff(data['train_offsets'])[user]
    errors, base = (np.clip(pred, 1, 5) - y) ** 2, (np.clip(control, 1, 5) - y) ** 2
    folds = blend_fold(user)
    masks = {'all': np.ones(len(y), bool), 'user_lt20': support < 20,
             'user_20_49': (support >= 20) & (support < 50), 'user_ge50': support >= 50}
    masks.update({f'fold{f}': folds == f for f in range(5)})
    slices = {}
    for key, mask in masks.items():
        slices[key] = dict(rows=int(mask.sum()),
                          control_rmse=float(np.sqrt(base[mask].mean())) if mask.any() else None,
                          candidate_rmse=float(np.sqrt(errors[mask].mean())) if mask.any() else None)
    # Paired user-cluster sandwich interval for descriptive MSE difference.
    delta = errors - base
    mean = float(delta.mean())
    counts = np.bincount(user, minlength=data['users'])
    sums = np.bincount(user, weights=delta, minlength=data['users'])
    active = counts > 0
    m = int(active.sum())
    se = np.sqrt(m / (m - 1) * np.sum((sums[active] - mean * counts[active]) ** 2)) / len(y)
    result = dict(name=name, slices=slices, mse_difference=mean,
                  descriptive_user_cluster_95_interval=[mean - 1.96 * se, mean + 1.96 * se],
                  predictions_sha256=digest(path),
                  limitation='Base/candidate epochs use probe. Interval does not adjust for model selection. This is not R5.')
    pathlib.Path(f'experiments/{name}-assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    if STATUS.exists():
        raise SystemExit('Queue status already exists; inspect it instead of launching twice.')
    paths = [f'artifacts/{name}.f32' for name in BANK]
    state = dict(pid=os.getpid(), started_utc=now(), status='running', stage='control', completed=[],
                 order=['HN1-history', 'HN2-sparse', 'HN3-session'],
                 bank={path: digest(path) for path in paths},
                 data_sha256=digest('data/processed/netflix.bin'))
    write_status(state)
    try:
        pathlib.Path('artifacts/history').mkdir(parents=True, exist_ok=True)
        ensemble('HN-control', paths)
        d = load('data/processed/netflix.bin')
        control = np.fromfile('artifacts/history/HN-control.f32', dtype='<f4').astype(np.float64)
        expected = np.fromfile('artifacts/blend-b7.f32', dtype='<f4').astype(np.float64)
        if control.shape != expected.shape or not np.allclose(control, expected, atol=2e-6, rtol=0):
            raise ValueError('Frozen B7 control failed reproduction; investigate before training')
        state['control_verified'] = True
        for name, arm in zip(state['order'], ['history', 'sparse', 'session']):
            # Detect accidental mutation of shared inputs before starting each run.
            if any(digest(path) != checksum for path, checksum in state['bank'].items()):
                raise ValueError('Frozen predictor bank changed')
            state['stage'] = name + ':training'; write_status(state)
            run(name, f'Masked ordinal rating-history pilot, arm={arm}; 20 epoch cap, patience 4; original probe development selection',
                [sys.executable, 'scripts/history_model.py', '--data', 'data/processed/netflix.bin',
                 '--model', f'artifacts/history/{name}.pt', '--predictions', f'artifacts/history/{name}.f32',
                 '--arm', arm, '--epochs', '20', '--patience', '4', '--device', 'mps'])
            state['stage'] = name + ':ensemble'; write_status(state)
            ensemble(name + '-blend', paths + [f'artifacts/history/{name}.f32'])
            report = assess(name, d, control)
            state['completed'].append(dict(name=name, summary=report['slices']['all']))
            write_status(state)
        state.update(status='complete', stage='done', finished_utc=now())
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_utc=now())
        write_status(state)
        raise
    write_status(state)


if __name__ == '__main__':
    main()
