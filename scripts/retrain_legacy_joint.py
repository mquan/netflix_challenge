#!/usr/bin/env python3
"""Reproduce retained J3/J4/J2 recipes sequentially, without changing bank artifacts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from recover_rbm_recipes import argument, replace
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('J3-joint128-reg005', 'J4-joint128-day128', 'J2-joint128-reg02')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', default='PR6-legacy-joint')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    plan_path = ROOT / 'experiments' / (args.name + '.plan.json')
    status_path = ROOT / 'experiments' / (args.name + '.status.json')
    if output.exists() or plan_path.exists() or status_path.exists():
        raise FileExistsError('choose new run paths')
    registry_path = ROOT / 'experiments/production-joint-recipes.json'
    registry = json.loads(registry_path.read_text())
    stages = {s['name']: s for s in registry['stages']}
    binary, data = ROOT / 'build/netflix', ROOT / 'data/processed/netflix.bin'
    frozen = {str(p): digest(p) for p in [registry_path, binary, data, Path(__file__)]}
    commands, outputs = [], {}
    for name in ORDER:
        stage = stages[name]
        for path, expected in [(stage['record'], stage['record_sha256']),
                               (stage['model'], stage['model_sha256']),
                               (stage['original_prediction'], stage['prediction_sha256'])]:
            if digest(path) != expected:
                raise ValueError('reference changed: ' + path)
            frozen[path] = expected
        command = json.loads(Path(stage['record']).read_text())['command']
        command[0] = str(binary)
        replace(command, '--data', data)
        replace(command, '--model', output / (name + '.bin'))
        replace(command, '--predictions', output / (name + '.f32'))
        replace(command, '--iterations', stage['fixed_final_epoch'])
        if '--patience' in command:
            replace(command, '--patience', 0)
        else:
            command += ['--patience', '0']
        command += ['--fixed-epoch', '--threads', '1']
        if stage['dependency']:
            replace(command, '--init-model', outputs[stage['dependency']])
        outputs[name] = argument(command, '--model')
        experiment_name = args.name + '-' + name.split('-')[0]
        if list((ROOT / 'experiments').glob(experiment_name + '.*')):
            raise FileExistsError(experiment_name)
        commands.append(dict(stage=name, experiment=experiment_name, command=command))
    output.mkdir(parents=True)
    plan_path.write_text(json.dumps(dict(frozen=frozen, stages=commands,
        purpose='Recipe reproduction only. No score search, bank mutation or production training.'), indent=2) + '\n')
    completed = []
    def status(**values):
        payload = dict(pid=os.getpid(), completed=completed,
                       updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(status_path)
    try:
        for entry in commands:
            name, command = entry['stage'], entry['command']
            if digest(binary) != frozen[str(binary)]:
                raise ValueError('native binary changed')
            status(state='running', stage=name)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', entry['experiment'],
                '--strategy', 'Fixed-budget reproduction of retained joint checkpoint; no score search',
                '--', *command], check=True)
            model, prediction = argument(command, '--model'), argument(command, '--predictions')
            expected = stages[name]
            actual_model, actual_prediction = digest(model), digest(prediction)
            a = np.fromfile(expected['original_prediction'], dtype='<f4').astype(float)
            b = np.fromfile(prediction, dtype='<f4').astype(float)
            if a.shape != b.shape or not np.isfinite(b).all():
                raise ValueError('invalid predictions')
            evidence = dict(stage=name, model_sha256=actual_model, prediction_sha256=actual_prediction,
                            expected_model_sha256=expected['model_sha256'],
                            expected_prediction_sha256=expected['prediction_sha256'],
                            model_byte_identical=actual_model == expected['model_sha256'],
                            prediction_byte_identical=actual_prediction == expected['prediction_sha256'],
                            max_abs_prediction_difference=float(np.max(np.abs(a - b))))
            evidence_path = ROOT / 'experiments' / (entry['experiment'] + '.verification.json')
            evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')
            if not (evidence['model_byte_identical'] and evidence['prediction_byte_identical']):
                raise ValueError('retraining does not reproduce the retained artifact: ' + name)
            completed.append(name)
        for path, expected in frozen.items():
            if digest(path) != expected:
                raise ValueError('frozen reference changed: ' + path)
        status(state='complete')
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
