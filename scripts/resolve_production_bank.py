"""Resolve the complete frozen blender bank only after all 82 production audits."""
import argparse
import copy
import json
from pathlib import Path
import struct

import numpy as np

from production_rbm_audit import digest, verify, write_new

COUNTS = {'GPU': 7, 'CPU-neural': 7, 'RBM': 12, 'DEPS': 4, 'JOINT': 12,
          'KNN': 28, 'AUTOREC': 2, 'BAYES': 4, 'KERNEL': 4, 'USERKNN': 2}


def merge(target, additions):
    for path, sha in additions.items():
        if path in target and target[path] != sha:
            raise ValueError('Conflicting provenance binding: ' + path)
        target[path] = sha


def expected_receipt(phase, plan, index, rows):
    stage = plan['stages'][index]
    expected = dict(state='verified_complete', stage=index, metric=None)
    if phase != 'DEPS':
        expected['qualifying_count'] = rows
    if phase in ('GPU', 'CPU-neural'):
        expected.update(column=stage['expected']['column'], fixed_epochs=stage['expected']['fixed_epochs'])
    elif phase in ('RBM', 'JOINT') or phase == 'USERKNN' and index == 0:
        expected.update(name=stage['name'], final_epoch=stage['fixed_final_epoch'])
    elif phase == 'DEPS':
        expected['kind'] = stage['kind']
    elif phase == 'KNN':
        expected['retained_columns'] = stage['retained_columns']
    elif phase == 'AUTOREC':
        expected.update(group=stage['group'], fixed_epochs=stage['config']['epochs'])
    elif phase == 'BAYES':
        expected.update(name=stage['name'], final_sweep=stage['final_sweep'],
                        kept=stage['final_sweep'] - stage['burn_in'])
    elif phase == 'KERNEL':
        expected.update(name=stage['name'], family=stage['family'],
                        fixed_epochs=6 if stage['family'] == 'local_residual' else 0)
    elif phase == 'USERKNN':
        expected['name'] = stage['name']
    else:
        raise ValueError('Unknown production phase')
    return expected


def unknown_queries(path, rows):
    with Path(path).open('rb') as stream:
        raw = stream.read(56)
    if len(raw) != 56:
        raise ValueError('Truncated qualifying dataset header')
    magic, users, items, train, query, implicit, first, last, mean = struct.unpack('<8sIIQQQHH4xd', raw)
    if magic != b'NFPDAT2\0' or query != rows:
        raise ValueError('Qualifying dataset query count differs')
    offset = 56 + 8 * (users + 1) + 5 * train
    dtype = np.dtype(dict(names=['rating'], formats=['u1'], offsets=[8], itemsize=12))
    labels = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(rows,))['rating']
    if np.any(labels != 0):
        raise ValueError('Qualifying rows must have unknown zero labels')


def resolve(mapping_path, bank_path, destination):
    mapping_path, bank_path, destination = (Path(p).resolve() for p in (mapping_path, bank_path, destination))
    if destination.exists():
        raise FileExistsError(destination)
    mapping_sha, bank_sha = digest(mapping_path), digest(bank_path)
    mapping = json.loads(mapping_path.read_text())
    bank = json.loads(bank_path.read_text())
    if (mapping['state'] != 'complete_planned_production_bank_coverage' or
            mapping['prediction_occurrences'] != 79 or mapping['support_alias_occurrences'] != 8 or
            mapping['required_stage_audits'] != 82 or set(mapping['phases']) != set(COUNTS)):
        raise ValueError('Complete frozen production bank mapping required')
    if mapping['frozen'].get(str(bank_path)) != bank_sha:
        raise ValueError('Mapping refers to a different frozen blender bank')
    frozen = dict(mapping['frozen'])
    merge(frozen, bank['sources'])
    query_spec = bank['qualifying']
    merge(frozen, {query_spec['dataset']['path']: query_spec['dataset']['sha256']})
    for group in ('predictions', 'auxiliary', 'context'):
        for entry in query_spec[group]:
            if not entry.get('pending'):
                merge(frozen, {entry['path']: entry['sha256']})
    receipts, audits, all_outputs, phases = {}, {}, {}, {}
    rows = bank['qualifying']['rows']
    for name, spec in mapping['phases'].items():
        path = Path(spec['plan'])
        if digest(path) != spec['plan_sha256'] or mapping['frozen'].get(str(path)) != spec['plan_sha256']:
            raise ValueError('Production phase plan changed')
        phase = json.loads(path.read_text())
        phases[name] = phase
        if len(phase['stages']) != COUNTS[name] or phase['data'] != bank['qualifying']['dataset']['path']:
            raise ValueError('Phase count or qualifying dataset differs')
        merge(frozen, phase['frozen'])
        for index, stage in enumerate(phase['stages']):
            audit_path = Path(stage['output_dir']) / 'audit.json'
            receipt = json.loads(audit_path.read_text())
            expected = dict(expected_receipt(name, phase, index, rows), plan_sha256=spec['plan_sha256'])
            if any(key not in receipt or receipt[key] != value for key, value in expected.items()):
                raise ValueError('Incomplete or mismatched production audit: ' + str(audit_path))
            if not receipt['outputs']:
                raise ValueError('Production audit binds no outputs')
            merge(all_outputs, receipt['outputs'])
            merge(all_outputs, receipt.get('dependencies', {}))
            audits[str(audit_path)] = digest(audit_path)
            receipts[name, index] = receipt
    if len(receipts) != 82:
        raise ValueError('All 82 production audits required')
    verify(frozen)
    verify(all_outputs)
    verify(audits)
    unknown_queries(bank['qualifying']['dataset']['path'], rows)
    result = copy.deepcopy(bank)
    replacements = {}
    entries = mapping['predictions'] + mapping['support_aliases']
    if len(mapping['predictions']) != 79 or len(mapping['support_aliases']) != 8:
        raise ValueError('Incomplete prediction/support occurrence mapping')
    for entry in entries:
        key = entry['name']
        if key in replacements:
            raise ValueError('Duplicate mapped blender input')
        name, index = entry['phase'], entry['stage']
        spec = mapping['phases'][name]
        stage = phases[name]['stages'][index]
        audit_path = str(Path(stage['output_dir']) / 'audit.json')
        if (entry['plan'] != spec['plan'] or entry['plan_sha256'] != spec['plan_sha256'] or
                entry['required_audit'] != audit_path):
            raise ValueError('Mapped occurrence refers to a different producer')
        target = entry['planned_output']
        receipt = receipts[name, index]
        if target not in receipt['outputs']:
            raise ValueError('Mapped output is not bound by its producer audit')
        if Path(target).stat().st_size != rows * 4:
            raise ValueError('Qualifying prediction/support count differs')
        values = np.memmap(target, '<f4', mode='r')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite qualifying input')
        replacements[key] = dict(name=key, path=target, sha256=receipt['outputs'][target])
    expected_order = [entry['name'] for entry in bank['qualifying']['predictions']]
    if [entry['name'] for entry in mapping['predictions']] != expected_order:
        raise ValueError('Retained prediction order differs')
    expected_supports = [entry['name'] for entry in bank['qualifying']['auxiliary'] if entry.get('pending')]
    if [entry['name'] for entry in mapping['support_aliases']] != expected_supports:
        raise ValueError('Retained support order differs')
    replaced = 0
    for group in ('predictions', 'auxiliary', 'context'):
        for index, original in enumerate(bank['qualifying'][group]):
            if original.get('pending'):
                replacement = replacements.pop(original['name'])
                mapped = next(entry for entry in entries if entry['name'] == original['name'])
                if mapped['development_sha256'] != original['development_sha256']:
                    raise ValueError('Retained development input hash differs')
                result['qualifying'][group][index] = replacement
                replaced += 1
    if replacements or replaced != 87:
        raise ValueError('Exactly 79 predictions and eight supports must be resolved')
    result['state'] = 'complete_audited_qualifying_bank_ready_for_fixed_blender'
    merge(result['sources'], {str(mapping_path): mapping_sha, str(bank_path): bank_sha,
                              str(Path(__file__).resolve()): digest(__file__)})
    merge(result['sources'], audits)
    merge(result['sources'], mapping['frozen'])
    result['production_provenance'] = dict(mapping=str(mapping_path), mapping_sha256=mapping_sha,
                                           stage_audits=audits, audited_bindings=all_outputs,
                                           production_stage_count=82, metric=None)
    result['limitation'] = ('All fixed-recipe qualifying inputs resolved from audited production stages. '
                           'Final blender fitting/inference and raw-order export verification remain required. '
                           'Development settings and frozen latent basis preserved; no qualifying score available.')
    if result['development'] != bank['development'] or result['settings'] != bank['settings'] or result['schema'] != bank['schema']:
        raise ValueError('Frozen development bank, budgets or schema changed')
    verify(frozen)
    verify(all_outputs)
    verify(audits)
    if digest(mapping_path) != mapping_sha or digest(bank_path) != bank_sha:
        raise ValueError('Mapping or frozen blender plan changed during resolution')
    write_new(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(resolve(args.mapping, args.bank, args.output)['state'])
