#!/usr/bin/env python3
"""Add blend features to the frozen R9-LAT-A2h config, then both blenders.

Stages, each through experiment.py and stopping on failure:
1. <run-id>-s3: the R9-LAT-A2h LightGBM config plus --features as extra support.
2. <run-id>-NN2: the same NN2 settings and fixed 50/50 average.
Then compares against R9-LAT-A2h-NN2 per fold (promotion: all five folds better).
Use --check to validate inputs and print the plan without starting anything.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from round10_b5_chain import BASE, CONTROL, ROOT, read, resolve, sha


def plan(run_id, features):
    base, control = read(BASE), read(CONTROL)
    if Path(control['config']['like']).name != BASE.name:
        raise ValueError('control is not the neural blend of the base')
    config = dict(base['config'])
    for name, expected in base['inputs'].items():
        if sha(resolve(name, config)) != expected:
            raise ValueError(f'Control input changed: {name}')
    for path in features:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    config['extra'] = [str(resolve(x, config)) for x in config['extra']]
    config['extra_support'] = [str(resolve(x, config)) for x in config['extra_support']] + [str(ROOT / x) for x in features]
    config['bank_root'] = str(Path(BASE).parents[2] / config['bank_root'])
    config['output'] = f'artifacts/r10/{run_id}-s3'
    config['processes'] = 3
    boost = [sys.executable, 'scripts/round4_boost.py']
    for name, value in config.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value: boost.append(flag)
        elif isinstance(value, list): boost += [flag, *map(str, value)]
        else: boost += [flag, str(value)]
    nn = [sys.executable, 'scripts/round8_nn_blend.py', '--data', config['data'],
          '--like', config['output'] + '.json', '--output', f'artifacts/r10/{run_id}-NN2',
          '--lr', '3e-4', '--dropout', '.2', '--seeds', '2026', '7', '13',
          '--epochs', '60', '--patience', '5', '--threads', '2']
    stages = [
        (f'{run_id}-s3', f'Frozen R9-LAT-A2h LightGBM config plus {len(features)} features; three seeds', boost),
        (f'{run_id}-NN2', f'NN2 on {run_id}-s3; three seeds and predeclared 50/50 LightGBM average', nn),
    ]
    for name, _, cmd in stages:
        prefix = Path(cmd[cmd.index('--output') + 1])
        if list(Path('experiments').glob(name + '.*')) or list(prefix.parent.glob(prefix.name + '.*')):
            raise FileExistsError(name)
    return stages, control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    stages, control = plan(args.run_id, args.features)
    print(json.dumps([(n, s, ' '.join(c)[-600:]) for n, s, c in stages], indent=2), flush=True)
    if args.check:
        return
    status_path = Path(f'experiments/{args.run_id}-chain.status.json')

    def status(**values):
        values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid())
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2) + '\n')
        temporary.replace(status_path)

    with status_path.open('x') as stream:  # refuses a second launch
        json.dump(dict(state='starting', pid=os.getpid()), stream)
    completed = []
    try:
        for name, strategy, command in stages:
            status(state='running', stage=name, completed=completed)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                            '--strategy', strategy, '--', *command], check=True)
            completed.append(name)
        result = read(f'artifacts/r10/{args.run_id}-NN2.json')
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
