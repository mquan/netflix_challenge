"""Audit fixed-sweep NFPBAYS1 production stages and posterior resume state."""
import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np

from production_rbm_audit import digest, load_plan, verify, write_new
from recover_bayes_recipes import GROUPS, posterior


def checkpoint(path, dataset, stage):
    info, mean = posterior(path)
    expected = dict(rank=stage['rank'], sweep=stage['final_sweep'],
                    kept=stage['final_sweep'] - stage['burn_in'],
                    fingerprint=dataset['fingerprint'], groups=stage['groups'],
                    prediction_count=dataset['query_count'])
    if any(info[key] != value for key, value in expected.items()):
        raise ValueError('Bayes checkpoint dataset, dimensions or posterior window differs')
    if (info['bag_power'] != 0.5 or not np.isfinite(info['drift_scale']) or
            not np.isfinite(info['intercept']) or not np.isfinite(info['precision']) or
            info['precision'] <= 0):
        raise ValueError('Invalid Bayes checkpoint scalar')
    # The archival posterior reader checks dimensions but skips parameter values.
    # Scan those values in bounded memory before allowing this checkpoint as a parent.
    with Path(path).open('rb') as stream:
        stream.seek(28 + len(GROUPS) * 12 + 24)
        for group in GROUPS:
            raw = stream.read(16)
            if len(raw) != 16 or not np.isfinite(struct.unpack('<dd', raw)).all():
                raise ValueError('Nonfinite Bayes prior')
            for dtype in ('<f4', '<f4', '<f8', '<f8'):
                count, = struct.unpack('<Q', stream.read(8))
                while count:
                    n = min(count, 1024 * 1024)
                    width = np.dtype(dtype).itemsize
                    raw = stream.read(n * width)
                    if len(raw) != n * width or not np.isfinite(np.frombuffer(raw, dtype)).all():
                        raise ValueError('Truncated or nonfinite Bayes parameter')
                    count -= n
    return info, mean


def dependencies(plan, plan_sha, stage):
    index = stage['dependency_index']
    if index is None:
        if stage['start_sweep'] != 0:
            raise ValueError('Fresh Bayes stage must start at sweep zero')
        return {}
    parent = plan['stages'][index]
    path = Path(parent['output_dir']) / 'audit.json'
    receipt = json.loads(path.read_text())
    model = str(Path(parent['output_dir']) / 'model.bin')
    if (receipt['state'] != 'verified_complete' or receipt['plan_sha256'] != plan_sha or
            receipt['stage'] != index or receipt['metric'] is not None or
            receipt['final_sweep'] != stage['start_sweep'] or model not in receipt['outputs'] or
            parent['rank'] != stage['rank'] or parent['groups'] != stage['groups'] or
            parent['burn_in'] != stage['burn_in']):
        raise ValueError('Bayes resume requires an audited matching production parent')
    verify(receipt['outputs'])
    checkpoint(model, plan['dataset'], parent)
    return {str(path): digest(path), model: receipt['outputs'][model]}


def started_record(plan, plan_sha, index, stage):
    return dict(state='prepared', plan_sha256=plan_sha, stage=index,
                command=stage['command'], data_sha256=plan['frozen'][plan['data']],
                binary_sha256=plan['frozen'][stage['command'][0]],
                dependencies=dependencies(plan, plan_sha, stage))


def prepare(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, plan_sha, index, stage)
    if digest(path) != plan_sha:
        raise ValueError('Bayes plan changed')
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
        raise ValueError('Bayes recipe or parent changed')
    names = ('started.json', 'model.bin', 'model.f32', 'reload.f32', 'train.log', 'reload.log')
    bound = {str(root / name): digest(root / name) for name in names}
    log = (root / 'train.log').read_text()
    sweeps = [int(n) for n in re.findall(r'^sweep=(\d+) ', log, re.M)]
    if sweeps != list(range(stage['start_sweep'] + 1, stage['final_sweep'] + 1)):
        raise ValueError('Incomplete, extra or repeated Bayes sweeps')
    header = log.splitlines()[0] if log else ''
    fields = dict(re.findall(r'(\w+)=([^ ]+)', header))
    expected = dict(factors=stage['rank'], threads=stage['threads'], samples=stage['final_sweep'],
                    burn_in=stage['burn_in'], fingerprint=plan['dataset']['fingerprint'],
                    resume_sweep=stage['start_sweep'])
    if not header.startswith('bayes ') or any(fields.get(k) != str(v) for k, v in expected.items()):
        raise ValueError('Bayes training header differs')
    # Native logging places the user bag before the item group. Both have item
    # cardinality in NFPBAYS1; retain the log's explicit names and order.
    log_groups = ('user', 'user-bag', 'item', 'item-bag', 'day', 'item-bin', 'session', 'drift')
    modes = ','.join(f'{name}:{stage["groups"][name]["mode"]}' for name in log_groups)
    if fields.get('modes') != modes:
        raise ValueError('Bayes feature modes differ')
    info, mean = checkpoint(root / 'model.bin', plan['dataset'], stage)
    if (root / 'model.f32').read_bytes() != mean.tobytes():
        raise ValueError('Bayes predictions differ from serialized posterior average')
    if bound[str(root / 'model.f32')] != bound[str(root / 'reload.f32')]:
        raise ValueError('Bayes reload differs')
    reload_log = (root / 'reload.log').read_text()
    if re.search(r'^sweep=', reload_log, re.M):
        raise ValueError('Bayes reload unexpectedly trained')
    for body in (log, reload_log):
        finals = re.findall(r'^final probe_rmse=\S+ kept=(\d+) sweep=(\d+)$', body, re.M)
        if finals != [(str(info['kept']), str(info['sweep']))]:
            raise ValueError('Bayes final posterior log differs')
    verify(plan['frozen'])
    verify(started['dependencies'])
    verify(bound)
    if digest(path) != plan_sha:
        raise ValueError('Bayes plan changed')
    result = dict(state='verified_complete', stage=index, name=stage['name'], plan_sha256=plan_sha,
                  final_sweep=info['sweep'], kept=info['kept'], outputs=bound,
                  dependencies=started['dependencies'], qualifying_count=len(mean), metric=None,
                  scope='Ordered qualifying posterior mean and resume state; native sentinel diagnostics are not scores.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
