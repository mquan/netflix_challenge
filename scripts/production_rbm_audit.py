"""Prepare and audit fixed-budget native production RBM stages."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError('Frozen input or output changed: ' + str(path))


def load_plan(path, index):
    path = Path(path)
    plan_sha = digest(path)
    plan = json.loads(path.read_text())
    verify(plan['frozen'])
    if not 0 <= index < len(plan['stages']):
        raise ValueError('Invalid RBM stage index')
    return path, plan_sha, plan, plan['stages'][index]


def dependencies(plan, plan_sha, stage):
    index = stage['dependency_index']
    if index is None:
        return {}
    parent = plan['stages'][index]
    path = Path(parent['output_dir']) / 'audit.json'
    receipt = json.loads(path.read_text())
    model = str(Path(parent['output_dir']) / 'model.bin')
    if (receipt['state'] != 'verified_complete' or receipt['plan_sha256'] != plan_sha or
            receipt['stage'] != index or receipt['metric'] is not None or
            receipt['final_epoch'] != parent['fixed_final_epoch']):
        raise ValueError('Resume dependency is not an audited production parent')
    verify(receipt['outputs'])
    return {str(path): digest(path), model: receipt['outputs'][model]}


def started_record(plan, plan_sha, index, stage):
    return dict(state='prepared', plan_sha256=plan_sha, stage=index,
                command=stage['command'], data_sha256=plan['frozen'][plan['data']],
                binary_sha256=plan['frozen'][stage['command'][0]],
                dependencies=dependencies(plan, plan_sha, stage))


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, plan_sha, index, stage)
    root = Path(stage['output_dir'])
    if digest(path) != plan_sha:
        raise ValueError('Plan changed during stage preparation')
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def checkpoint(path, dataset, stage):
    """Check the exact V1/V2 native layout in bounded memory."""
    items, hidden = dataset['items'], stage['hidden']
    counts = [items * 5 * hidden, items * 5, hidden, items * hidden]
    if stage['conditional']:
        counts += [dataset['users'] * 5, dataset['sessions'] * 5, items * 8 * 5]
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24:
            raise ValueError('Truncated RBM header')
        magic, fingerprint, rank, epoch = struct.unpack('<QQII', header)
        expected_magic = 0x4E4650524253324D if stage['conditional'] else 0x4E4650524253314D
        if (magic, fingerprint, rank, epoch) != (
                expected_magic, dataset['fingerprint'], hidden, stage['fixed_final_epoch']):
            raise ValueError('Checkpoint format, dataset, hidden size or final epoch differs')
        for index, count in enumerate(counts):
            if index >= 4:
                size = stream.read(8)
                if len(size) != 8 or struct.unpack('<Q', size)[0] != count:
                    raise ValueError('Conditional-bias dimensions differ')
            remaining = count
            while remaining:
                n = min(remaining, 1024 * 1024)
                block = stream.read(n * 4)
                if len(block) != n * 4 or not np.isfinite(np.frombuffer(block, '<f4')).all():
                    raise ValueError('Truncated or nonfinite checkpoint parameters')
                remaining -= n
        if stream.read(1):
            raise ValueError('Trailing checkpoint data')


def audit(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    root = Path(stage['output_dir'])
    destination = root / 'audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    started_path = root / 'started.json'
    started = json.loads(started_path.read_text())
    if started != started_record(plan, plan_sha, index, stage):
        raise ValueError('Started recipe or resume dependencies changed')
    output_paths = [started_path, root / 'model.bin', root / 'model.f32',
                    root / 'reload.f32', root / 'train.log', root / 'reload.log']
    bound = {str(p): digest(p) for p in output_paths}
    log = (root / 'train.log').read_text()
    records = re.findall(r'^epoch=(\d+)\s+.*$', log, re.M)
    if [int(epoch) for epoch in records] != list(range(stage['start_epoch'], stage['fixed_final_epoch'] + 1)):
        raise ValueError('Incomplete or repeated fixed-budget epochs')
    first = log.splitlines()[0] if log else ''
    fields = {k: int(v) for k, v in re.findall(r'(hidden|threads|seed|fingerprint|conditional_biases)=(\d+)', first)}
    expected = dict(hidden=stage['hidden'], threads=stage['threads'], seed=stage['seed'],
                    fingerprint=plan['dataset']['fingerprint'])
    if any(fields.get(k) != v for k, v in expected.items()) or fields.get('conditional_biases', 0) != int(stage['conditional']):
        raise ValueError('Training header differs from frozen recipe')
    if 'early_stop' in log:
        raise ValueError('Fixed-budget training stopped early')
    checkpoint(root / 'model.bin', plan['dataset'], stage)
    if (root / 'model.f32').stat().st_size != plan['dataset']['query_count'] * 4:
        raise ValueError('Wrong qualifying prediction count')
    values = np.fromfile(root / 'model.f32', '<f4')
    if not np.isfinite(values).all() or np.any(values < 1 - 1e-5) or np.any(values > 5 + 1e-5):
        raise ValueError('Invalid softmax expected-rating predictions')
    if bound[str(root / 'model.f32')] != bound[str(root / 'reload.f32')]:
        raise ValueError('Reloaded predictions differ')
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != plan_sha:
        raise ValueError('Production RBM plan changed')
    result = dict(state='verified_complete', plan_sha256=plan_sha, stage=index,
                  name=stage['name'], final_epoch=stage['fixed_final_epoch'],
                  qualifying_count=len(values), prediction_min=float(values.min()),
                  prediction_max=float(values.max()), outputs=bound,
                  dependencies=started['dependencies'], metric=None,
                  scope='Ordered production qualifying column, not a qualifying score or bank promotion.')
    write_new(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
