#!/usr/bin/env python3
"""LC6: fixed A2h epochs 4/8 added to the promoted bank containing epoch 12."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from gbdt_blend import load
from local_context_chain import paired, resolve, write
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
R9 = Path('/Users/quan/workspace/netflix_round9')
BASE = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-s3.json'
CONTROL = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-NN2.json'
OUTPUT = ROOT / 'artifacts/local-context/LC6'


def read(path):
    return json.loads(Path(path).read_text())


def command(script, config):
    result = [sys.executable, script]
    for key, value in config.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            result.extend([flag, *map(str, value)])
        else:
            result.extend([flag, str(value)])
    return result


def plan():
    base, control = read(BASE), read(CONTROL)
    assert Path(control['config']['like']).name == BASE.name
    assert importlib.metadata.version('lightgbm') == base['lightgbm_version']
    assert importlib.metadata.version('torch') == control['torch_version']
    frozen = {}
    for name, expected in base['inputs'].items():
        path = resolve(name, base['config'], R9)
        if digest(path) != expected:
            raise ValueError(f'changed base input: {name}')
        frozen[str(path)] = expected
    data = ROOT / base['config']['data']
    original_data = R9 / base['config']['data']
    assert digest(data) == digest(original_data)
    assert digest(ROOT / 'artifacts/r4/meta-context.npz') == digest(R9 / 'artifacts/r4/meta-context.npz')
    refit_path = ROOT / 'artifacts/masked/A2h-refit.json'
    refit = read(refit_path)
    assert refit['data_sha256'] == digest(data)
    assert refit['selected_epoch'] == 12
    snapshots = [ROOT / f'artifacts/masked/A2h-refit.refit.e{epoch}.f32' for epoch in (4, 8)]
    d = load(data)
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    reference_path = CONTROL.with_suffix('.avg.f32')
    reference = np.fromfile(reference_path, dtype='<f4').astype(float)
    score = paired(y, reference, reference, users)['control_rmse']
    assert abs(score - control['avg_rmse']) < 1e-8
    for path in snapshots:
        assert str(path) not in frozen
        values = np.fromfile(path, dtype='<f4')
        assert values.shape == y.shape and np.isfinite(values).all()
    extra_paths = [BASE, CONTROL, data, refit_path, reference_path,
                   ROOT / 'artifacts/r4/meta-context.npz', *snapshots]
    for relative in ('experiments/UNION-round-manifest.json', 'experiments/BOOST-support-inputs.json'):
        extra_paths.append(R9 / base['config']['bank_root'] / relative)
    for name in ('ensemble_trajectory.py', 'round4_boost.py', 'round8_nn_blend.py',
                 'gbdt_blend.py', 'moonshot_nn_blend.py', 'local_context_chain.py'):
        extra_paths.append(ROOT / 'scripts' / name)
    frozen.update({str(path): digest(path) for path in extra_paths})
    cfg = dict(base['config'])
    cfg.update(data=str(data), bank_root=str(R9 / cfg['bank_root']),
               output=str(OUTPUT / 'LC6-trajectory-s3'), processes=3,
               extra=[str(resolve(p, base['config'], R9)) for p in cfg['extra']] + list(map(str, snapshots)),
               extra_support=[str(resolve(p, base['config'], R9)) for p in cfg['extra_support']])
    neural = dict(control['config'])
    neural.update(data=str(data), like=cfg['output'] + '.json', output=str(OUTPUT / 'LC6-trajectory-NN2'))
    for key in set(base['config']) - {'data', 'bank_root', 'output', 'processes', 'extra', 'extra_support'}:
        assert cfg[key] == base['config'][key]
    for key in set(control['config']) - {'data', 'like', 'output'}:
        assert neural[key] == control['config'][key]
    stages = [('LC6-trajectory-s3', command('scripts/round4_boost.py', cfg)),
              ('LC6-trajectory-NN2', command('scripts/round8_nn_blend.py', neural))]
    for name, _ in stages:
        if list((ROOT / 'experiments').glob(name + '.*')) or list(OUTPUT.glob(name + '.*')):
            raise FileExistsError(name)
    return dict(frozen=frozen, snapshot_epochs=[4, 8], snapshots=list(map(str, snapshots)),
                reference_rmse=score, base_report=str(BASE), control_report=str(CONTROL),
                data=str(data), stages=stages, changed_settings='Two added snapshot columns; three independent fold workers.'), y, users, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    a = parser.parse_args()
    os.chdir(ROOT)
    audit, y, users, reference = plan()
    print(json.dumps(dict(phase='preflight', frozen_inputs=len(audit['frozen']),
                         reference_rmse=audit['reference_rmse'], snapshot_epochs=audit['snapshot_epochs'])), flush=True)
    if a.check:
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT / 'experiments/LC6-plan.json'
    if plan_path.exists():
        raise FileExistsError(plan_path)
    write(plan_path, audit)
    status_path = ROOT / 'experiments/LC6-trajectory.status.json'
    completed = []
    def status(**values):
        write(status_path, dict(pid=os.getpid(), completed=completed,
                               updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values))
    try:
        for name, cmd in audit['stages']:
            status(state='running', stage=name)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                '--strategy', 'Fixed A2h epochs 4/8 plus promoted epoch 12; unchanged full-bank blending settings',
                '--', *cmd], check=True)
            completed.append(name)
        for path, expected in audit['frozen'].items():
            if digest(path) != expected:
                raise ValueError(f'input changed during run: {path}')
        nn = read(OUTPUT / 'LC6-trajectory-NN2.json')
        values = np.fromfile(OUTPUT / 'LC6-trajectory-NN2.avg.f32', dtype='<f4').astype(float)
        comparison = paired(y, reference, values, users)
        assert abs(comparison['treatment_rmse'] - nn['avg_rmse']) < 1e-8
        report = dict(comparison=comparison, promoted=comparison['all_folds_improve'],
            substantial_investment_gate=comparison['all_folds_improve'] and comparison['delta_rmse'] <= -.0003,
            plan_sha256=digest(plan_path), prediction_sha256=digest(OUTPUT / 'LC6-trajectory-NN2.avg.f32'),
            metric='Original-probe development user-fold CV; no independent hidden-test score.')
        write(ROOT / 'experiments/LC6-comparison.json', report)
        status(state='complete', result=report)
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
