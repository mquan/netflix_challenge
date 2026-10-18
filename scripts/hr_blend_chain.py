#!/usr/bin/env python3
"""Score new prediction columns on the frozen R9-LAT-A2h bank (round10_b5_chain recipe).

Stages: <name>-s3 (R9-LAT-A2h LightGBM config, three seeds, plus the columns),
then <name>-NN2 (NN2 settings, fixed 50/50 average), then a paired fold
comparison with R9-LAT-A2h-20261010-NN2. Promotion needs all five folds better.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path('/Users/quan/workspace/netflix_challenge/artifacts/residual-chains')
R9 = Path('/Users/quan/workspace/netflix_round9')
BASE = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-s3.json'
CONTROL = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-NN2.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(name, config):
    """Bank inputs are relative to the round-9 worktree or its bank_root."""
    for path in (R9 / name, R9 / config['bank_root'] / name):
        if path.exists():
            return path
    raise FileNotFoundError(name)


def plan(name, columns, processes):
    base, control = read(BASE), read(CONTROL)
    if Path(control['config']['like']).name != BASE.name:
        raise ValueError('control is not the neural blend of the base')
    config = dict(base['config'])
    for path, expected in base['inputs'].items():
        if sha(resolve(path, config)) != expected:
            raise ValueError(f'Control input changed: {path}')
    config['extra'] = [str(resolve(x, config)) for x in config['extra']] + [str(Path(c).resolve()) for c in columns]
    config['extra_support'] = [str(resolve(x, config)) for x in config['extra_support']]
    config['bank_root'] = str(R9 / config['bank_root'])
    config['data'] = str(ART.parents[1] / config['data'])  # this worktree has no data/
    config['output'] = str(ART / f'{name}-s3')
    config['processes'] = processes
    boost = [sys.executable, 'scripts/round4_boost.py']
    for key, value in config.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value: boost.append(flag)
        elif isinstance(value, list): boost += [flag, *map(str, value)]
        else: boost += [flag, str(value)]
    nn = [sys.executable, 'scripts/round8_nn_blend.py', '--data', config['data'],
          '--like', config['output'] + '.json', '--output', str(ART / f'{name}-NN2'),
          '--lr', '3e-4', '--dropout', '.2', '--seeds', '2026', '7', '13',
          '--epochs', '60', '--patience', '5', '--threads', '2']
    stages = [(f'{name}-s3', 'Frozen R9-LAT-A2h LightGBM config plus ' + ' '.join(columns), boost),
              (f'{name}-NN2', 'NN2 on the s3 run; three seeds and predeclared 50/50 LightGBM average', nn)]
    for stage, _, _ in stages:
        if list((ROOT / 'experiments').glob(stage + '.*')) or list(ART.glob(stage + '.*')):
            raise FileExistsError(stage)
    return stages, control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--columns', nargs='+', required=True)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    stages, control = plan(args.name, args.columns, args.processes)
    print(json.dumps([(n, s, ' '.join(c)) for n, s, c in stages], indent=2), flush=True)
    if args.check:
        return
    status_path = ROOT / 'experiments' / f'{args.name}.status.json'

    def status(**values):
        values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid())
        status_path.with_suffix('.tmp').write_text(json.dumps(values, indent=2) + '\n')
        status_path.with_suffix('.tmp').replace(status_path)

    completed = []
    try:
        for stage, strategy, command in stages:
            status(state='running', stage=stage, completed=completed)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', stage,
                            '--strategy', strategy, '--', *command], check=True)
            completed.append(stage)
        result = read(ART / f'{args.name}-NN2.json')
        deltas = [a - b for a, b in zip(result['avg_fold_rmse'], control['avg_fold_rmse'])]
        status(state='complete', completed=completed, avg_rmse=result['avg_rmse'],
               control='R9-LAT-A2h-20261010-NN2', control_rmse=control['avg_rmse'],
               rmse_delta=result['avg_rmse'] - control['avg_rmse'], fold_deltas=deltas,
               passes_promotion_rule=all(d < 0 for d in deltas),
               metric='Development user-fold CV; not a final test score')
    except BaseException as error:
        status(state='failed', completed=completed, error=repr(error))
        raise


if __name__ == '__main__':
    main()
