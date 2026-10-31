"""Audit GP inference and fixed six-epoch local residual production exports."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from production_rbm_audit import digest, load_plan, verify, write_new


def dependencies(plan):
    spec = plan['source_phase']
    path = Path(spec['plan'])
    if digest(path) != spec['plan_sha256']:
        raise ValueError('Kernel J3 source phase changed')
    source = json.loads(path.read_text())
    if source['data'] != plan['data'] or source['dataset'] != plan['dataset']:
        raise ValueError('Kernel J3 uses different production data')
    stage = source['stages'][spec['stage']]
    if (stage['family'], stage['rank'], stage['fixed_final_epoch'], stage['day_rank']) != ('joint', 128, 9, 0):
        raise ValueError('Kernel requires production J3 at epoch nine')
    root = Path(stage['output_dir'])
    receipt_path = root / 'audit.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['state'] != 'verified_complete' or receipt['stage'] != spec['stage'] or
            receipt['plan_sha256'] != spec['plan_sha256'] or receipt['final_epoch'] != 9 or
            receipt['metric'] is not None):
        raise ValueError('Kernel J3 is not audited')
    bound = {str(receipt_path): digest(receipt_path)}
    for name in ('model.bin', 'model.f32'):
        target = str(root / name)
        if target not in receipt['outputs']:
            raise ValueError('J3 dependency is not bound')
        bound[target] = receipt['outputs'][target]
    if spec['model'] != str(root / 'model.bin'):
        raise ValueError('Kernel model is not the new J3 output')
    verify(bound)
    return bound


def started_record(plan, sha, index, stage):
    if stage['command'][stage['command'].index('--model') + 1] != plan['source_phase']['model']:
        raise ValueError('Kernel command uses a different base')
    return dict(state='prepared', stage=index, plan_sha256=sha, command=stage['command'],
                data_sha256=plan['frozen'][plan['data']], binary_sha256=plan['frozen'][stage['command'][0]],
                dependencies=dependencies(plan))


def prepare(plan_path, index):
    path, sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, sha, index, stage)
    if digest(path) != sha:
        raise ValueError('Kernel plan changed')
    root = Path(stage['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def suffixes(stage):
    if stage['family'] == 'gp':
        return [f'.{i}.f32' for i in range(4)] + ['.base.f32', '.mask.f32']
    if stage['family'] == 'local_residual':
        return [f'.epoch{i}.f32' for i in range(1, 7)]
    raise ValueError('Unsupported kernel family')


def check_log(log, stage, dataset):
    if stage['family'] == 'gp':
        headers = re.findall(r'^kernel_ready (.+)$', log, re.M)
        if len(headers) != 1:
            raise ValueError('Missing or repeated GP kernel header')
        fields = dict(re.findall(r'(\w+)=([^ ]+)', headers[0]))
        expected = dict(items=dataset['items'], gamma=4, time_scale=stage['parameters']['time_scale'],
                        neighbors=32, user_modulo=1)
        if 'persistent=' in headers[0]:
            expected.update(persistent=stage['parameters']['persistent'], implicit_mix=0)
        elif stage['parameters']['persistent'] != 0:
            raise ValueError('Persistent GP header missing')
        if any(key not in fields or float(fields[key]) != value for key, value in expected.items()):
            raise ValueError('GP kernel parameters differ')
        variants = re.findall(r'^variant=(\d+) sampled_rmse=\S+ count=(\d+)$', log, re.M)
        if variants != [(str(i), str(dataset['query_count'])) for i in range(4)]:
            raise ValueError('GP variants or query coverage differ')
        if len(re.findall(r'^base_sampled_rmse=\S+ seconds=\S+$', log, re.M)) != 1:
            raise ValueError('Missing GP completion record')
    else:
        rows = re.findall(r'^local groups=(\d+) rank=(\d+) active=(\d+) regularization=(\S+) seed=(\d+)$', log, re.M)
        p = stage['parameters']
        if (len(rows) != 1 or tuple(map(int, rows[0][:3])) !=
                (p['groups'], p['rank'], 1 if p['groups'] == 1 else 2) or
                float(rows[0][3]) != .05 or int(rows[0][4]) != 20260928):
            raise ValueError('Local model parameters differ')
        if [int(i) for i in re.findall(r'^epoch=(\d+) ', log, re.M)] != list(range(1, 7)):
            raise ValueError('Local fixed six-epoch budget incomplete or repeated')


def audit(plan_path, index):
    path, sha, plan, stage = load_plan(plan_path, index)
    root = Path(stage['output_dir'])
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    started = json.loads((root / 'started.json').read_text())
    if started != started_record(plan, sha, index, stage):
        raise ValueError('Kernel recipe or J3 dependency changed')
    names = ['started.json', 'run.log', 'replay.log']
    names += [prefix + suffix for prefix in ('model', 'replay') for suffix in suffixes(stage)]
    bound = {str(root / name): digest(root / name) for name in names}
    for name in ('run.log', 'replay.log'):
        check_log((root / name).read_text(), stage, plan['dataset'])
    for suffix in suffixes(stage):
        target = root / ('model' + suffix)
        if target.stat().st_size != plan['dataset']['query_count'] * 4:
            raise ValueError('Kernel query count differs')
        values = np.fromfile(target, '<f4')
        if not np.isfinite(values).all() or digest(target) != bound[str(root / ('replay' + suffix))]:
            raise ValueError('Kernel predictions are nonfinite or replay differs')
        if suffix == '.mask.f32' and not np.all(values == 1):
            raise ValueError('GP did not cover every qualifying row')
    if stage['family'] == 'gp':
        model = Path(plan['source_phase']['model']).with_suffix('.f32')
        if bound[str(root / 'model.base.f32')] != started['dependencies'][str(model)]:
            raise ValueError('GP base differs from audited J3 predictions')
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != sha:
        raise ValueError('Kernel plan changed')
    result = dict(state='verified_complete', stage=index, name=stage['name'], plan_sha256=sha,
                  family=stage['family'], fixed_epochs=6 if stage['family'] == 'local_residual' else 0,
                  qualifying_count=plan['dataset']['query_count'], outputs=bound,
                  dependencies=started['dependencies'], metric=None,
                  scope='Ordered GP/local prediction exports. Local factors are not persisted; '
                        'the complete fixed six-epoch run is repeated exactly. No qualifying score or promotion.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
