"""Audit item-kNN inference against newly audited production models and graphs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from production_rbm_audit import digest, load_plan, verify, write_new


def argument(command, flag):
    return command[command.index(flag) + 1]


def dependencies(plan, stage):
    bound = {}
    for role in ('base', 'graph'):
        spec = stage[role + '_dependency']
        phase_name = spec['phase'] if role == 'base' else 'dependencies'
        phase = plan['source_phases'][phase_name]
        source_path = Path(phase['plan'])
        if digest(source_path) != phase['plan_sha256']:
            raise ValueError('Source phase changed')
        source = json.loads(source_path.read_text())
        if source['data'] != plan['data'] or source['dataset'] != plan['dataset']:
            raise ValueError('kNN source phase uses different production data')
        index = spec['stage']
        expected_stage = source['stages'][index]
        root = Path(expected_stage['output_dir'])
        receipt_path = root / 'audit.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt['state'] != 'verified_complete' or receipt['stage'] != index or
                receipt['plan_sha256'] != phase['plan_sha256'] or receipt['metric'] is not None):
            raise ValueError('kNN dependency is not audited')
        if role == 'base':
            target = root / 'model.bin'
            if phase_name == 'joint':
                if receipt['final_epoch'] != expected_stage['fixed_final_epoch']:
                    raise ValueError('Joint dependency budget differs')
            elif expected_stage['kind'] != 'baseline' or receipt['kind'] != 'baseline':
                raise ValueError('Expected production temporal baseline')
        else:
            kind = expected_stage['kind']
            if kind not in ('graph64', 'graph256') or receipt['kind'] != kind:
                raise ValueError('Expected production neighbor graph')
            target = root / ('derived/neighbors64.bin' if kind == 'graph64' else 'neighbors256.bin')
        if spec['model' if role == 'base' else 'graph'] != str(target):
            raise ValueError('kNN dependency path is not the new phase output')
        expected = receipt['outputs'].get(str(target))
        if expected is None or digest(target) != expected:
            raise ValueError('Consumed kNN dependency is missing, changed or unbound')
        bound[str(receipt_path)] = digest(receipt_path)
        bound[str(target)] = expected
    return bound


def started_record(plan, plan_sha, index, stage):
    return dict(state='prepared', stage=index, plan_sha256=plan_sha, command=stage['command'],
                data_sha256=plan['frozen'][plan['data']], binary_sha256=plan['frozen'][stage['command'][0]],
                dependencies=dependencies(plan, stage))


def prepare(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, plan_sha, index, stage)
    if digest(path) != plan_sha:
        raise ValueError('kNN plan changed')
    root = Path(stage['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def check_log(log, command):
    rows = re.findall(r'^knn base=(\S+) neighbors=(\d+) power=(\S+) shrink=(\S+) time_decay=(\S+) probe_rmse=\S+$', log, re.M)
    if len(rows) != 1:
        raise ValueError('Missing or repeated complete kNN inference record')
    base, neighbors, power, shrink, decay = rows[0]
    expected_base = argument(command, '--joint-model') if '--joint-model' in command else 'time'
    if base != expected_base or int(neighbors) != int(argument(command, '--neighbors')):
        raise ValueError('kNN log base or neighbor count differs')
    for actual, flag in [(power, '--power'), (shrink, '--shrink'), (decay, '--time-decay')]:
        expected = float(np.float32(argument(command, flag)))
        if not np.isfinite(float(actual)) or abs(float(actual) - expected) > 1e-8 * max(1, abs(expected)):
            raise ValueError('kNN log hyperparameters differ')


def audit(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    root = Path(stage['output_dir'])
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    started = json.loads((root / 'started.json').read_text())
    if started != started_record(plan, plan_sha, index, stage):
        raise ValueError('kNN command or consumed dependencies changed')
    files = ['started.json', 'model.f32', 'support.f32', 'replay.f32', 'replay-support.f32', 'run.log', 'replay.log']
    bound = {str(root / name): digest(root / name) for name in files}
    arrays = {}
    for name in ('model.f32', 'support.f32'):
        file = root / name
        if file.stat().st_size != plan['dataset']['query_count'] * 4:
            raise ValueError('Wrong qualifying prediction/support coverage')
        values = np.fromfile(file, '<f4')
        if not np.isfinite(values).all() or (name == 'support.f32' and np.any(values < 0)):
            raise ValueError('Nonfinite predictions or invalid support weights')
        arrays[name] = values
    if (bound[str(root / 'model.f32')] != bound[str(root / 'replay.f32')] or
            bound[str(root / 'support.f32')] != bound[str(root / 'replay-support.f32')]):
        raise ValueError('Repeated native inference differs')
    for name in ('run.log', 'replay.log'):
        check_log((root / name).read_text(), stage['command'])
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != plan_sha:
        raise ValueError('kNN plan changed')
    result = dict(state='verified_complete', stage=index, plan_sha256=plan_sha,
                  retained_columns=stage['retained_columns'], qualifying_count=plan['dataset']['query_count'],
                  support_min=float(arrays['support.f32'].min()), support_max=float(arrays['support.f32'].max()),
                  outputs=bound, dependencies=started['dependencies'], metric=None,
                  scope='Raw ordered qualifying predictions and nonnegative support weights; no qualifying score or promotion.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
