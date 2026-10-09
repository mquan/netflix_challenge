#!/usr/bin/env python3
"""Run A2h's holdout-selected refit, then both three-seed ensemble evaluations.

Use --check to validate inputs and print the plan without starting training.
Each stage uses experiment.py; an existing experiment is never overwritten.
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
STATUS = Path('experiments/A2h-chain.status.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replace(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def plan():
    selection = read('artifacts/masked/A2h-select.json')
    assert read('experiments/A2h-select.json')['exit_code'] == 0
    epoch = selection['selected_epoch']
    assert epoch == min(selection['selection'], key=lambda row: row['validation_rmse'])['epoch']
    command = read('experiments/A2h-select.json')['command'].copy()
    script = Path(command[1])
    for path, key in [(script, 'source_sha256'),
                      (script.with_name('masked_history.py'), 'base_source_sha256'),
                      (script.with_name('gbdt_blend.py'), 'loader_sha256'),
                      (Path(selection['arguments']['data']), 'data_sha256'),
                      (Path(selection['arguments']['selection_data']), 'selection_sha256')]:
        if sha(path) != selection[key]:
            raise ValueError(f'Selection input changed: {path}')
    command.remove('--selection-only')
    index = command.index('--selection-data')
    del command[index:index + 2]
    replace(command, '--epochs', epoch)
    replace(command, '--output', 'artifacts/masked/A2h-refit')
    command.append('--save-epochs')
    boost = read('experiments/R8-A2f-s3.json')['command'].copy()
    replace(boost, '--output', 'artifacts/r8/R8-A2h-s3')
    boost.insert(boost.index('--extra-support'), f'artifacts/masked/A2h-refit.refit.e{epoch}.f32')
    nn = read('experiments/R8-A2f-NN2.json')['command'].copy()
    replace(nn, '--like', 'artifacts/r8/R8-A2h-s3.json')
    replace(nn, '--output', 'artifacts/r8/R8-A2h-NN2')
    # Verify the frozen control bank before spending time on the refit.
    control = read('artifacts/r8/R8-A2f-s3.json')
    for name, expected in control['inputs'].items():
        path = Path(name)
        if name not in control['config']['extra'] + control['config']['extra_support']:
            path = Path(control['config']['bank_root']) / name
        if sha(path) != expected:
            raise ValueError(f'Control input changed: {path}')
    stages = [
        ('A2h-refit', f'A2h fixed refit at holdout-selected epoch {epoch}; no probe epoch selection', command),
        ('R8-A2h-s3', 'R8-A2f-s3 frozen bank plus A2h; same three LightGBM seeds', boost),
        ('R8-A2h-NN2', 'NN2 on R8-A2h-s3; three seeds and predeclared 50/50 LightGBM average', nn),
    ]
    for name, _, cmd in stages:
        if list(Path('experiments').glob(name + '.*')):
            raise FileExistsError(name)
        prefix = cmd[cmd.index('--output') + 1]
        if list(Path(prefix).parent.glob(Path(prefix).name + '.*')):
            raise FileExistsError(prefix)
    return stages


def status(**values):
    values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid())
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    stages = plan()
    print(json.dumps(stages, indent=2), flush=True)
    if args.check:
        return
    # Exclusive creation prevents accidentally launching this chain twice.
    with STATUS.open('x') as stream:
        json.dump(dict(state='starting', pid=os.getpid()), stream)
    completed = []
    try:
        for name, strategy, command in stages:
            status(state='running', stage=name, completed=completed)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                            '--strategy', strategy, '--', *command], check=True)
            completed.append(name)
        result = read('artifacts/r8/R8-A2h-NN2.json')
        comparisons = {}
        for name in ['R8-B4-A2e-NN2', 'R8-A2f-NN2']:
            control = read(f'artifacts/r8/{name}.json')
            deltas = [a - b for a, b in zip(result['avg_fold_rmse'], control['avg_fold_rmse'])]
            comparisons[name] = dict(rmse_delta=result['avg_rmse'] - control['avg_rmse'],
                                     fold_deltas=deltas, all_five_better=all(d < 0 for d in deltas))
        status(state='complete', completed=completed, avg_rmse=result['avg_rmse'], comparisons=comparisons,
               passes_promotion_rule=comparisons['R8-B4-A2e-NN2']['all_five_better'])
    except BaseException as error:
        status(state='failed', completed=completed, error=repr(error))
        raise


if __name__ == '__main__':
    main()
