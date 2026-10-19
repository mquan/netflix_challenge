#!/usr/bin/env python3
"""Replay the six retained GP/local columns from recovered historical sources."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
NAME = 'PR12-kernel-bank'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    os.chdir(ROOT)
    destination = ROOT / 'artifacts/production-replay' / NAME
    plan_path = ROOT / 'experiments' / (NAME + '-plan.json')
    status_path = ROOT / 'experiments' / (NAME + '.status.json')
    verification_path = ROOT / 'experiments' / (NAME + '-verification.json')
    if destination.exists() or any(p.exists() for p in (plan_path, status_path, verification_path)):
        raise FileExistsError('recovery output already exists; inspect it before manual recovery')
    inventory_path = ROOT / 'experiments/production-bank-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    columns = {Path(c['path']).name: c for c in inventory['columns']}
    source_path = ROOT / 'experiments/production-kernel-sources.json'
    source = json.loads(source_path.read_text())
    archive = ROOT / 'experiments/production-kernel-sources.zip'
    if digest(archive) != source['archive_sha256']:
        raise ValueError('source archive changed')
    binary_root = ROOT / 'artifacts/production-replay/PR12-source'
    for group in source['groups'].values():
        for info in group.values():
            if digest(binary_root / info['archive_member']) != info['sha256']:
                raise ValueError('recovered source changed')
    data, model = ROOT / 'data/processed/netflix.bin', ROOT / 'artifacts/joint128-reg005.bin'
    original = json.loads((ROOT / 'experiments/GP2-persistent-full.json').read_text())
    if digest(model) != original['model_artifact']['sha256']:
        raise ValueError('J3 differs from original kernel dependency')
    recipes = [
        ('gp-full', 'moonshot_gp_initial', ['--neighbors', '32'], ['.0.f32', '.1.f32', '.2.f32']),
        ('gp2-persistent-full', 'moonshot_gp_persistent', ['--neighbors', '32', '--time-scale', '30', '--persistent', '.5'], ['.0.f32']),
        ('local-global', 'moonshot_local', ['--groups', '1', '--rank', '16', '--epochs', '6'], ['.epoch6.f32']),
        ('local-local', 'moonshot_local', ['--groups', '8', '--rank', '8', '--epochs', '6'], ['.epoch6.f32']),
    ]
    frozen = {str(p): digest(p) for p in (Path(__file__), data, model, inventory_path, source_path, archive)}
    stages = []
    for name, binary, options, suffixes in recipes:
        binary_path = binary_root / binary
        frozen[str(binary_path)] = digest(binary_path)
        expected = []
        for suffix in suffixes:
            column = columns[name + suffix]
            if digest(column['path']) != column['sha256']:
                raise ValueError('retained prediction changed: ' + column['path'])
            frozen[column['path']] = column['sha256']
            expected.append(dict(reference=column['path'], sha256=column['sha256'], output=str(destination / (name + suffix))))
        command = [str(binary_path), '--data', str(data), '--model', str(model),
                   '--output', str(destination / name), *options]
        experiment = NAME + '-' + name
        if list((ROOT / 'experiments').glob(experiment + '.*')):
            raise FileExistsError(experiment)
        stages.append(dict(name=name, experiment=experiment, command=command, expected=expected))
    destination.mkdir(parents=True)
    plan = dict(frozen=frozen, stages=stages,
                scope='Fixed original-data recipe replay. No tuning, production-data training or promotion.',
                compiler=subprocess.check_output(['clang++', '--version'], text=True))
    write(plan_path, plan)
    results = []

    def status(**values):
        write(status_path, dict(pid=os.getpid(), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                completed=len(results), **values))

    try:
        for stage in stages:
            for path in (str(Path(__file__)), stage['command'][0], str(model), str(data)):
                if digest(path) != frozen[path]:
                    raise ValueError('frozen input changed: ' + path)
            status(state='running', stage=stage['name'])
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', stage['experiment'],
                            '--strategy', 'Recover retained GP/local columns with frozen historical source and fixed budgets; no score search',
                            '--', *stage['command']], check=True)
            comparisons = []
            for expected in stage['expected']:
                prediction = np.fromfile(expected['output'], dtype='<f4')
                if prediction.shape != (1408395,) or not np.isfinite(prediction).all():
                    raise ValueError('invalid prediction output')
                actual = digest(expected['output'])
                delta = prediction.astype(float) - np.fromfile(expected['reference'], dtype='<f4').astype(float)
                comparisons.append(dict(**expected, actual_sha256=actual, byte_identical=actual == expected['sha256'],
                                        max_absolute_difference=float(np.max(np.abs(delta)))))
            results.append(dict(stage=stage['name'], comparisons=comparisons))
            print(json.dumps(results[-1]), flush=True)
        for path, expected in frozen.items():
            if digest(path) != expected:
                raise ValueError('frozen artifact changed: ' + path)
        write(verification_path, dict(plan_sha256=digest(plan_path), results=results,
                                     scope=plan['scope']))
        status(state='complete', exact_columns=sum(c['byte_identical'] for r in results for c in r['comparisons']))
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
