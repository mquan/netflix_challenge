#!/usr/bin/env python3
"""Run A2i (A2h with --suffix-prob .5) selection, refit, and both ensemble evaluations.

A2i changes one setting from A2h: the suffix-masked objective of the original A2
instead of A2b's random masking, at A2h's 1536-step history and 16384 batch tokens.
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
STATUS = Path('experiments/A2i-chain.status.json')
NAMES = ['A2i-select', 'A2i-refit', 'R8-A2i-s3', 'R8-A2i-NN2']
OUTPUTS = ['artifacts/masked/A2i-select', 'artifacts/masked/A2i-refit',
           'artifacts/r8/R8-A2i-s3', 'artifacts/r8/R8-A2i-NN2']
CONTROLS = ['R8-A2h-NN2', 'R8-B4-A2e-NN2']


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


def select_stage():
    previous = read('artifacts/masked/A2h-select.json')
    command = read('experiments/A2h-select.json')['command'].copy()
    script = Path(command[1])
    for path, key in [(script, 'source_sha256'),
                      (script.with_name('masked_history.py'), 'base_source_sha256'),
                      (script.with_name('gbdt_blend.py'), 'loader_sha256'),
                      (Path(previous['arguments']['data']), 'data_sha256'),
                      (Path(previous['arguments']['selection_data']), 'selection_sha256')]:
        if sha(path) != previous[key]:
            raise ValueError(f'A2h selection input changed: {path}')
    replace(command, '--suffix-prob', .5)
    replace(command, '--output', OUTPUTS[0])
    return ('A2i-select', 'A2i: A2h (seq-len 1536, batch-tokens 16384) with --suffix-prob .5; '
            'suffix-masked counterpart of A2h', command)


def later_stages():
    selection = read(OUTPUTS[0] + '.json')
    assert read(f'experiments/{NAMES[0]}.json')['exit_code'] == 0
    epoch = selection['selected_epoch']
    assert epoch == min(selection['selection'], key=lambda row: row['validation_rmse'])['epoch']
    command = read(f'experiments/{NAMES[0]}.json')['command'].copy()
    command.remove('--selection-only')
    index = command.index('--selection-data')
    del command[index:index + 2]
    replace(command, '--epochs', epoch)
    replace(command, '--output', OUTPUTS[1])
    command.append('--save-epochs')
    boost = read('experiments/R8-A2h-s3.json')['command'].copy()
    replace(boost, '--output', OUTPUTS[2])
    boost.insert(boost.index('--extra-support'), f'{OUTPUTS[1]}.refit.e{epoch}.f32')
    nn = read('experiments/R8-A2h-NN2.json')['command'].copy()
    replace(nn, '--like', OUTPUTS[2] + '.json')
    replace(nn, '--output', OUTPUTS[3])
    return [
        (NAMES[1], f'A2i fixed refit at holdout-selected epoch {epoch}; no probe epoch selection', command),
        (NAMES[2], 'R8-A2h-s3 frozen bank plus A2i; same three LightGBM seeds', boost),
        (NAMES[3], 'NN2 on R8-A2i-s3; three seeds and predeclared 50/50 LightGBM average', nn),
    ]


def preflight():
    # Verify the frozen control bank before spending time on selection and refit.
    control = read('artifacts/r8/R8-A2h-s3.json')
    for name, expected in control['inputs'].items():
        path = Path(name)
        if name not in control['config']['extra'] + control['config']['extra_support']:
            path = Path(control['config']['bank_root']) / name
        if sha(path) != expected:
            raise ValueError(f'Control input changed: {path}')
    for name, prefix in zip(NAMES, OUTPUTS):
        if list(Path('experiments').glob(name + '.*')):
            raise FileExistsError(name)
        if list(Path(prefix).parent.glob(Path(prefix).name + '.*')):
            raise FileExistsError(prefix)


def status(**values):
    values.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid())
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(STATUS)


def run(stage, completed):
    name, strategy, command = stage
    status(state='running', stage=name, completed=completed)
    subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                    '--strategy', strategy, '--', *command], check=True)
    completed.append(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    preflight()
    first = select_stage()
    print(json.dumps(first, indent=2), flush=True)
    if args.check:
        return
    # Exclusive creation prevents accidentally launching this chain twice.
    with STATUS.open('x') as stream:
        json.dump(dict(state='starting', pid=os.getpid()), stream)
    completed = []
    try:
        run(first, completed)
        stages = later_stages()
        print(json.dumps(stages, indent=2), flush=True)
        for stage in stages:
            run(stage, completed)
        result = read(OUTPUTS[3] + '.json')
        comparisons = {}
        for name in CONTROLS:
            control = read(f'artifacts/r8/{name}.json')
            deltas = [a - b for a, b in zip(result['avg_fold_rmse'], control['avg_fold_rmse'])]
            comparisons[name] = dict(rmse_delta=result['avg_rmse'] - control['avg_rmse'],
                                     fold_deltas=deltas, all_five_better=all(d < 0 for d in deltas))
        status(state='complete', completed=completed, avg_rmse=result['avg_rmse'], comparisons=comparisons,
               passes_promotion_rule=comparisons[CONTROLS[0]]['all_five_better'])
    except BaseException as error:
        status(state='failed', completed=completed, error=repr(error))
        raise


if __name__ == '__main__':
    main()
