#!/usr/bin/env python3
"""B5: B4's Bayesian timeSVD++ plus the item x rating-frequency group, then both blenders.

Stages, each through experiment.py and stopping on failure:
1. B5-full-r128: B4's exact command (rank 128, item-bin/drift factors, seed 13,
   160 sweeps) plus --item-freq 1, using the round-9 sampler (NFPBAYS2).
2. R10-B5-s3: the frozen R9-LAT-A2h LightGBM config plus the B5 sweep-160 column.
3. R10-B5-NN2: the same NN2 settings and fixed 50/50 average.
Then compares against R9-LAT-A2h-NN2 per fold (promotion: all five folds better).
Use --check to validate inputs and print the plan without starting anything.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
R9 = Path('/Users/quan/workspace/netflix_round9')
BASE = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-s3.json'
CONTROL = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-NN2.json'
BINARY = 'artifacts/r10/bin/netflix-bayes-91fde00'
STATUS = Path('experiments/B5-chain.status.json')
SWEEP = 160


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


def plan(threads):
    b4 = read('experiments/B4-full-r128.json')['command'].copy()
    b4[0] = BINARY
    b4[b4.index('--model') + 1] = 'artifacts/bayes/B5-full-r128.bin'
    b4[b4.index('--predictions') + 1] = 'artifacts/bayes/B5-full-r128.f32'
    b4[b4.index('--threads') + 1] = str(threads)
    bayes = [*b4, '--item-freq', '1']  # whole chain runs under nice 10
    column = f'artifacts/bayes/B5-full-r128.s{SWEEP:04d}.f32'

    base, control = read(BASE), read(CONTROL)
    if Path(control['config']['like']).name != BASE.name:
        raise ValueError('control is not the neural blend of the base')
    config = dict(base['config'])
    # Verify every frozen input before spending a day on sampling.
    for name, expected in base['inputs'].items():
        if sha(resolve(name, config)) != expected:
            raise ValueError(f'Control input changed: {name}')
    # Absolute paths so this checkout reads the round-9 bank and latent features.
    config['extra'] = [str(resolve(x, config)) for x in config['extra']] + [str(ROOT / column)]
    config['extra_support'] = [str(resolve(x, config)) for x in config['extra_support']]
    config['bank_root'] = str(R9 / config['bank_root'])
    config['output'] = 'artifacts/r10/R10-B5-s3'
    config['processes'] = 3
    boost = [sys.executable, 'scripts/round4_boost.py']
    for name, value in config.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value: boost.append(flag)
        elif isinstance(value, list): boost += [flag, *map(str, value)]
        else: boost += [flag, str(value)]
    nn = [sys.executable, 'scripts/round8_nn_blend.py', '--data', config['data'],
          '--like', config['output'] + '.json', '--output', 'artifacts/r10/R10-B5-NN2',
          '--lr', '3e-4', '--dropout', '.2', '--seeds', '2026', '7', '13',
          '--epochs', '60', '--patience', '5', '--threads', '2']
    stages = [
        ('B5-full-r128', 'B4 command plus --item-freq 1 (round-9 sampler 91fde00); sweep 160 predeclared', bayes),
        ('R10-B5-s3', 'Frozen R9-LAT-A2h LightGBM config plus the B5 sweep-160 column; three seeds', boost),
        ('R10-B5-NN2', 'NN2 on R10-B5-s3; three seeds and predeclared 50/50 LightGBM average', nn),
    ]
    for name, _, cmd in stages:
        if list(Path('experiments').glob(name + '.*')):
            raise FileExistsError(name)
        prefix = Path(cmd[cmd.index('--output' if '--output' in cmd else '--model') + 1])
        if list(prefix.parent.glob(prefix.stem.split('.')[0] + '.*')):
            raise FileExistsError(prefix)
    return stages, column, control


def status(**values):
    values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid())
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--threads', type=int, default=5)
    args = parser.parse_args()
    os.chdir(ROOT)
    stages, column, control = plan(args.threads)
    print(json.dumps([(n, s, ' '.join(c)) for n, s, c in stages], indent=2), flush=True)
    print('binary sha256', sha(BINARY), flush=True)
    if args.check:
        return
    Path('artifacts/r10').mkdir(exist_ok=True)
    with STATUS.open('x') as stream:  # refuses a second launch
        json.dump(dict(state='starting', pid=os.getpid()), stream)
    completed = []
    try:
        for name, strategy, command in stages:
            status(state='running', stage=name, completed=completed)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                            '--strategy', strategy, '--', *command], check=True)
            completed.append(name)
            if name == 'B5-full-r128' and not Path(column).exists():
                raise FileNotFoundError(column)
        result = read('artifacts/r10/R10-B5-NN2.json')
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
