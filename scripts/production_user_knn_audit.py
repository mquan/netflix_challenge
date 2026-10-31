"""Audit the new fixed-budget K1 and retained user-kNN prediction/support outputs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from production_joint_audit import audit as audit_joint, prepare as prepare_joint
from production_rbm_audit import digest, load_plan, verify, write_new


def dependencies(plan, sha):
    parent = plan['stages'][0]
    if (parent['family'], parent['rank'], parent['fixed_final_epoch'], parent['day_rank']) != ('joint', 256, 41, 0):
        raise ValueError('User-kNN requires production K1 rank256 epoch41')
    root = Path(parent['output_dir'])
    path = root / 'audit.json'
    receipt = json.loads(path.read_text())
    model = str(root / 'model.bin')
    if (receipt['state'] != 'verified_complete' or receipt['stage'] != 0 or
            receipt['plan_sha256'] != sha or receipt['final_epoch'] != 41 or
            receipt['metric'] is not None or model not in receipt['outputs']):
        raise ValueError('Production K1 is not audited')
    verify(receipt['outputs'])
    command = plan['stages'][1]['command']
    if command[command.index('--joint-model') + 1] != model:
        raise ValueError('User-kNN command does not consume new production K1')
    return {str(path): digest(path), model: receipt['outputs'][model]}


def started_record(plan, sha, stage):
    return dict(state='prepared', stage=1, plan_sha256=sha, command=stage['command'],
                data_sha256=plan['frozen'][plan['data']], binary_sha256=plan['frozen'][stage['command'][0]],
                dependencies=dependencies(plan, sha))


def prepare(plan_path, index):
    if index == 0:
        return prepare_joint(plan_path, 0)
    path, sha, plan, stage = load_plan(plan_path, index)
    if index != 1 or stage['family'] != 'user_knn':
        raise ValueError('Invalid user-kNN stage')
    record = started_record(plan, sha, stage)
    if digest(path) != sha:
        raise ValueError('User-kNN phase changed')
    root = Path(stage['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def check_log(log, stage, dataset, model):
    bases = re.findall(r'^user_knn base=(\S+) train_rmse=\S+ residual_seconds=\S+$', log, re.M)
    if bases != [model]:
        raise ValueError('User-kNN log base differs')
    rows = re.findall(r'^user_knn queries=(\d+) items=(\d+) (.+)$', log, re.M)
    if len(rows) != 1 or int(rows[0][0]) != dataset['query_count'] or not 1 <= int(rows[0][1]) <= dataset['items']:
        raise ValueError('User-kNN query coverage differs')
    fields = dict(re.findall(r'(\w+)=([^ ]+)', rows[0][2]))
    expected = dict(screen_rank=256, candidates=300, neighbors=100, max_raters=0,
                    alpha=50, power=2, shrink=2, time_decay=0, threads=stage['threads'])
    if any(key not in fields or float(fields[key]) != value for key, value in expected.items()):
        raise ValueError('User-kNN inference parameters differ')
    finals = re.findall(r'^user_knn_result queries=(\d+) base_rmse=\S+ probe_rmse=\S+ factor_rmse=\S+ mean_neighbors=\S+$', log, re.M)
    if finals != [str(dataset['query_count'])]:
        raise ValueError('User-kNN completion record differs')


def audit(plan_path, index):
    if index == 0:
        return audit_joint(plan_path, 0)
    path, sha, plan, stage = load_plan(plan_path, index)
    if index != 1 or stage['family'] != 'user_knn':
        raise ValueError('Invalid user-kNN stage')
    root = Path(stage['output_dir'])
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    started = json.loads((root / 'started.json').read_text())
    if started != started_record(plan, sha, stage):
        raise ValueError('User-kNN recipe or K1 dependency changed')
    names = ['started.json', 'run.log', 'replay.log']
    names += [prefix + suffix for prefix in ('model', 'replay') for suffix in ('.f32', '.support.f32', '.count.f32')]
    bound = {str(root / name): digest(root / name) for name in names}
    for suffix in ('.f32', '.support.f32', '.count.f32'):
        file = root / ('model' + suffix)
        if file.stat().st_size != plan['dataset']['query_count'] * 4:
            raise ValueError('User-kNN output count differs')
        values = np.fromfile(file, '<f4')
        if not np.isfinite(values).all() or bound[str(file)] != bound[str(root / ('replay' + suffix))]:
            raise ValueError('User-kNN output is nonfinite or replay differs')
        if suffix != '.f32' and np.any(values < 0):
            raise ValueError('Negative user-kNN support or count')
        if suffix == '.count.f32' and (np.any(values > min(100, plan['dataset']['users'] - 1)) or
                                      np.any(values != np.floor(values))):
            raise ValueError('Invalid user-kNN neighbor counts')
    model = str(Path(plan['stages'][0]['output_dir']) / 'model.bin')
    for name in ('run.log', 'replay.log'):
        check_log((root / name).read_text(), stage, plan['dataset'], model)
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != sha:
        raise ValueError('User-kNN plan changed')
    result = dict(state='verified_complete', stage=1, name=stage['name'], plan_sha256=sha,
                  qualifying_count=plan['dataset']['query_count'], outputs=bound,
                  dependencies=started['dependencies'], metric=None,
                  scope='Ordered user-kNN qualifying predictions, support weights and neighbor counts from new '
                        'production K1; separate frozen latent-feature basis remains unchanged. No qualifying score.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
