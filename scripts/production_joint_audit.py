"""Audit fixed-budget production joint/integrated checkpoints and resume state."""
import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np

from production_dependency_audit import graph, graph_fingerprint
from production_rbm_audit import digest, load_plan, verify, write_new


def read_values(stream, count):
    while count:
        n = min(count, 1024 * 1024)
        block = stream.read(n * 4)
        if len(block) != n * 4 or not np.isfinite(np.frombuffer(block, '<f4')).all():
            raise ValueError('Truncated or nonfinite joint parameters')
        count -= n


def read_size(stream):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError('Truncated joint vector size')
    return struct.unpack('<Q', raw)[0]


def checkpoint(path, dataset, stage):
    users, items, sessions, rank = (dataset['users'], dataset['items'], dataset['sessions'], stage['rank'])
    counts = [users, users, users, items, items * 30, items * 8, sessions, sessions,
              users * rank, items * rank, items * rank, users * rank]
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or struct.unpack('<QQII', header) != (
                0x4E46504A4F494E30 + stage['model_format'], dataset['fingerprint'], rank, stage['fixed_final_epoch']):
            raise ValueError('Joint format, dataset, rank or final epoch differs')
        if stage['model_format'] not in (2, 3):
            raise ValueError('Unsupported production joint format')
        for count in counts:
            if read_size(stream) != count:
                raise ValueError('Joint vector dimensions differ')
            read_values(stream, count)
        raw_day = stream.read(4)
        if len(raw_day) != 4 or struct.unpack('<I', raw_day)[0] != stage['day_rank']:
            raise ValueError('Joint day rank differs')
        expected = sessions * stage['day_rank']
        if read_size(stream) != expected:
            raise ValueError('Joint day-factor dimensions differ')
        read_values(stream, expected)
        if stage['model_format'] == 3:
            # None of the retained J1-J4 recipes enables frequency factors.
            if read_size(stream) != 0:
                raise ValueError('Unexpected joint frequency factors')
        if stream.read(1):
            raise ValueError('Trailing joint checkpoint data')


def sidecar(path, body):
    _, _, items, neighbors = struct.unpack('<QQII', body[:24])
    with Path(path).open('rb') as stream:
        header = stream.read(16)
        if len(header) != 16 or struct.unpack('<QQ', header) != (0x4E46494E54474E31, graph_fingerprint(body)):
            raise ValueError('Integrated sidecar graph fingerprint differs')
        read_values(stream, items * neighbors * 2)
        if stream.read(1):
            raise ValueError('Trailing integrated sidecar data')


def dependencies(plan, plan_sha, stage):
    bound = {}
    index = stage['dependency_index']
    if index is not None:
        parent = plan['stages'][index]
        path = Path(parent['output_dir']) / 'audit.json'
        receipt = json.loads(path.read_text())
        if (receipt['state'] != 'verified_complete' or receipt['stage'] != index or
                receipt['plan_sha256'] != plan_sha or receipt['metric'] is not None or
                receipt['final_epoch'] != parent['fixed_final_epoch']):
            raise ValueError('Unaudited production joint parent')
        required = [str(Path(parent['output_dir']) / 'model.bin')]
        if stage['family'] == 'integrated':
            if parent['family'] != 'integrated':
                raise ValueError('Integrated resume requires an integrated parent')
            required.append(str(Path(parent['output_dir']) / 'model.bin.nbr'))
        if not set(required) <= set(receipt['outputs']):
            raise ValueError('Parent checkpoint or mandatory neighbor sidecar is not audited')
        verify(receipt['outputs'])
        bound[str(path)] = digest(path)
        bound.update(receipt['outputs'])
    if stage['family'] == 'integrated':
        spec = plan['graph_dependency']
        external = json.loads(Path(spec['plan']).read_text())
        if (digest(spec['plan']) != spec['plan_sha256'] or external['data'] != plan['data'] or
                external['dataset'] != plan['dataset']):
            raise ValueError('Graph phase refers to different production data')
        path = Path(spec['audit'])
        receipt = json.loads(path.read_text())
        if (receipt['state'] != 'verified_complete' or receipt['plan_sha256'] != spec['plan_sha256'] or
                receipt['stage'] != spec['stage'] or receipt['kind'] != 'graph64' or receipt['metric'] is not None or
                spec['graph'] not in receipt['outputs']):
            raise ValueError('Production graph64 is not audited')
        verify(receipt['outputs'])
        bound[str(path)] = digest(path)
        bound.update(receipt['outputs'])
    return bound


def started_record(plan, plan_sha, index, stage):
    return dict(state='prepared', stage=index, plan_sha256=plan_sha, command=stage['command'],
                data_sha256=plan['frozen'][plan['data']], binary_sha256=plan['frozen'][stage['command'][0]],
                dependencies=dependencies(plan, plan_sha, stage))


def prepare(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, plan_sha, index, stage)
    if digest(path) != plan_sha:
        raise ValueError('Joint phase changed')
    root = Path(stage['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def audit(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    root = Path(stage['output_dir'])
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    started = json.loads((root / 'started.json').read_text())
    if started != started_record(plan, plan_sha, index, stage):
        raise ValueError('Joint recipe or dependencies changed')
    files = ['started.json', 'model.bin', 'model.f32', 'reload.f32', 'train.log', 'reload.log']
    if stage['family'] == 'integrated':
        files.append('model.bin.nbr')
    bound = {str(root / name): digest(root / name) for name in files}
    log = (root / 'train.log').read_text()
    if ([int(n) for n in re.findall(r'^epoch=(\d+) ', log, re.M)] !=
            list(range(stage['start_epoch'], stage['fixed_final_epoch'] + 1)) or 'early_stop' in log):
        raise ValueError('Incomplete or repeated joint epoch budget')
    first = log.splitlines()[0] if log else ''
    fields = {k: int(v) for k, v in re.findall(r'(factors|seed|fingerprint|k)=(\d+)', first)}
    if not first.startswith(stage['family'] + ' ') or fields.get('factors') != stage['rank'] or fields.get('seed') != stage['seed']:
        raise ValueError('Joint training header differs')
    if stage['family'] == 'joint':
        if fields.get('fingerprint') != plan['dataset']['fingerprint']:
            raise ValueError('Joint log dataset differs')
        day = re.search(r'^day_factors=(\d+) ', log, re.M)
        if day is None or int(day.group(1)) != stage['day_rank']:
            raise ValueError('Joint log day rank differs')
    elif fields.get('k') != plan['graph_dependency']['neighbors']:
        raise ValueError('Integrated log graph size differs')
    checkpoint(root / 'model.bin', plan['dataset'], stage)
    if stage['family'] == 'integrated':
        spec = plan['graph_dependency']
        body = graph(spec['graph'], plan['dataset'], spec['neighbors'])
        sidecar(root / 'model.bin.nbr', body)
    if (root / 'model.f32').stat().st_size != plan['dataset']['query_count'] * 4:
        raise ValueError('Wrong qualifying prediction count')
    values = np.fromfile(root / 'model.f32', '<f4')
    if not np.isfinite(values).all() or bound[str(root / 'model.f32')] != bound[str(root / 'reload.f32')]:
        raise ValueError('Nonfinite predictions or differing reload')
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != plan_sha:
        raise ValueError('Joint phase changed')
    result = dict(state='verified_complete', stage=index, plan_sha256=plan_sha, name=stage['name'],
                  final_epoch=stage['fixed_final_epoch'], outputs=bound, dependencies=started['dependencies'],
                  qualifying_count=len(values), metric=None,
                  scope='Raw ordered qualifying joint column and required resume state; no qualifying score or promotion.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
