"""Audit newly trained production baseline, neighbor graphs and neighborhood."""
import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np

from production_graph import prefix_graph
from production_rbm_audit import digest, load_plan, verify, write_new


def finite_floats(stream, count):
    while count:
        n = min(count, 1024 * 1024)
        block = stream.read(n * 4)
        if len(block) != n * 4 or not np.isfinite(np.frombuffer(block, '<f4')).all():
            raise ValueError('Truncated or nonfinite dependency parameters')
        count -= n
    if stream.read(1):
        raise ValueError('Trailing dependency parameters')


def baseline(path, dataset, iterations):
    with Path(path).open('rb') as stream:
        header = stream.read(40)
        if len(header) != 40:
            raise ValueError('Truncated temporal-baseline header')
        actual = struct.unpack('<8sIIQI4xd', header)
        expected = (b'NFPTIM1\0', dataset['users'], dataset['items'],
                    dataset['sessions'], iterations, dataset['mean'])
        if actual != expected:
            raise ValueError('Temporal baseline dimensions, budget or mean differ')
        count = dataset['users'] * 3 + dataset['items'] * (1 + 30 + 8) + dataset['sessions'] * 2
        finite_floats(stream, count)


def graph(path, dataset, neighbors):
    body = Path(path).read_bytes()
    prefix_graph(body, neighbors)
    if struct.unpack('<QQII', body[:24]) != (
            0x4E46475241504831, dataset['fingerprint'], dataset['items'], neighbors):
        raise ValueError('Graph dataset fingerprint or dimensions differ')
    return body


def graph_fingerprint(body):
    value = 14695981039346656037
    for byte in body[24:]:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def neighborhood(path, dataset, graph_body, epochs):
    _, _, items, count = struct.unpack('<QQII', graph_body[:24])
    with Path(path).open('rb') as stream:
        header = stream.read(40)
        if len(header) != 40 or struct.unpack('<QQQIIIf', header) != (
                0x4E464E42484F4F32, dataset['fingerprint'], graph_fingerprint(graph_body),
                epochs, items, count, 0.0):
            raise ValueError('Neighborhood checkpoint dataset, graph, budget or decay differs')
        finite_floats(stream, items * count * 2)


def dependencies(plan, plan_sha, stage):
    bound = {}
    for index in stage['dependency_indices']:
        parent = plan['stages'][index]
        path = Path(parent['output_dir']) / 'audit.json'
        receipt = json.loads(path.read_text())
        if (receipt['state'] != 'verified_complete' or receipt['plan_sha256'] != plan_sha or
                receipt['stage'] != index or receipt['metric'] is not None):
            raise ValueError('Unaudited production dependency')
        verify(receipt['outputs'])
        bound[str(path)] = digest(path)
        bound.update(receipt['outputs'])
    return bound


def started_record(plan, plan_sha, index, stage):
    return dict(state='prepared', stage=index, plan_sha256=plan_sha,
                commands=stage['commands'], data_sha256=plan['frozen'][plan['data']],
                dependencies=dependencies(plan, plan_sha, stage))


def prepare(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, plan_sha, index, stage)
    if digest(path) != plan_sha:
        raise ValueError('Dependency plan changed')
    root = Path(stage['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def audit(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    root, dataset = Path(stage['output_dir']), plan['dataset']
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    started = json.loads((root / 'started.json').read_text())
    if started != started_record(plan, plan_sha, index, stage):
        raise ValueError('Started recipe or dependencies changed')
    files = ['started.json', *stage['files']]
    bound = {str(root / name): digest(root / name) for name in files}
    kind = stage['kind']
    log = (root / 'train.log').read_text()
    if kind == 'baseline':
        baseline(root / 'model.bin', dataset, 40)
        if [int(n) for n in re.findall(r'^iteration=(\d+) ', log, re.M)] != list(range(1, 41)):
            raise ValueError('Incomplete temporal-baseline budget')
    elif kind == 'graph256':
        graph(root / 'neighbors256.bin', dataset, 256)
        if f"graph_complete items={dataset['items']} neighbors=256 " not in log:
            raise ValueError('Incomplete graph construction')
    elif kind == 'graph64':
        source = Path(stage['source_graph'])
        body = graph(source, dataset, 256)
        expected, _ = prefix_graph(body, 64)
        actual = graph(root / 'derived/neighbors64.bin', dataset, 64)
        if actual != expected:
            raise ValueError('Graph64 is not the exact sorted graph256 prefix')
        manifest = json.loads((root / 'derived/manifest.json').read_text())
        if (manifest['source_sha256'] != digest(source) or
                manifest['output_sha256'] != digest(root / 'derived/neighbors64.bin') or
                manifest['source'] != str(source.resolve()) or
                manifest['output'] != str((root / 'derived/neighbors64.bin').resolve())):
            raise ValueError('Graph-prefix manifest differs')
    elif kind == 'neighborhood':
        body = graph(stage['source_graph'], dataset, 64)
        neighborhood(root / 'model.bin', dataset, body, 4)
        if [int(n) for n in re.findall(r'^epoch=(\d+) ', log, re.M)] != list(range(1, 5)):
            raise ValueError('Incomplete neighborhood budget')
        if (root / 'model.f32').stat().st_size != dataset['query_count'] * 4:
            raise ValueError('Invalid qualifying prediction byte count')
        values = np.fromfile(root / 'model.f32', '<f4')
        if len(values) != dataset['query_count'] or not np.isfinite(values).all():
            raise ValueError('Invalid qualifying prediction coverage or values')
        if bound[str(root / 'model.f32')] != bound[str(root / 'reload.f32')]:
            raise ValueError('Neighborhood reload differs')
    else:
        raise ValueError('Unknown dependency stage kind')
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != plan_sha:
        raise ValueError('Dependency plan changed')
    result = dict(state='verified_complete', stage=index, kind=kind,
                  plan_sha256=plan_sha, outputs=bound, dependencies=started['dependencies'],
                  metric=None, scope='Production dependencies or raw qualifying column; no qualifying score or promotion.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
