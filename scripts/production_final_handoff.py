"""Bind each fixed blender stage to its bank and model before execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from production_rbm_audit import digest, verify, write_new
from production_submission import write_submission

BLENDER = Path(__file__).with_name('production_blend.py').resolve()


def bank_record(path):
    path = Path(path).resolve()
    bank = json.loads(path.read_text())
    provenance = bank['production_provenance']
    if (bank['state'] != 'complete_audited_qualifying_bank_ready_for_fixed_blender' or
            provenance['production_stage_count'] != 82 or len(provenance['stage_audits']) != 82 or
            provenance['metric'] is not None):
        raise ValueError('Complete audited production bank required')
    verify(bank['sources'])
    verify(provenance['stage_audits'])
    verify(provenance['audited_bindings'])
    for split in ('development', 'qualifying'):
        for group in ('predictions', 'auxiliary', 'context'):
            if any(entry.get('pending') for entry in bank[split][group]):
                raise ValueError('Pending final bank input')
            if [entry['name'] for entry in bank[split][group]] != bank['schema'][group]:
                raise ValueError('Final bank input order differs')
    return bank


def input_bindings(spec):
    result = {spec['dataset']['path']: spec['dataset']['sha256']}
    for group in ('predictions', 'auxiliary', 'context'):
        for entry in spec[group]:
            if entry['path'] in result and result[entry['path']] != entry['sha256']:
                raise ValueError('Conflicting final bank inputs')
            result[entry['path']] = entry['sha256']
    return result


def fit_bindings(bank_path, bank, model_dir):
    root = Path(model_dir).resolve()
    report = json.loads((root / 'model.json').read_text())
    run = json.loads((root / 'run.json').read_text())
    if (report['state'] != 'complete' or report['settings'] != bank['settings'] or
            report['schema'] != bank['schema'] or report['training_rows'] != bank['development']['rows']):
        raise ValueError('Fitted blender settings, schema or training rows differ')
    for key, kind, pattern in [('trees', 'tree', 'tree-{}.txt'), ('networks', 'neural', 'neural-{}.pt')]:
        expected = [dict(file=pattern.format(i), **arm) for i, arm in enumerate(bank['settings']['budgets'][kind])]
        if report[key] != expected:
            raise ValueError('Fitted blender fixed budgets differ')
    artifacts = {str(root / name): sha for name, sha in report['artifacts'].items()}
    verify(artifacts)
    verify(report['sources'])
    if (run['action'] != 'fit' or run['bank'] != str(Path(bank_path).resolve()) or
            run['bank_sha256'] != digest(bank_path) or run['rows'] != bank['development']['rows'] or
            run['input_hashes'] != input_bindings(bank['development'])):
        raise ValueError('Fitted blender run binding differs')
    outputs = {str(root / name): sha for name, sha in run['outputs'].items()}
    if outputs.get(str(root / 'model.json')) != digest(root / 'model.json'):
        raise ValueError('Fitted model report is not bound')
    verify(outputs)
    return dict(outputs, **{str(root / 'run.json'): digest(root / 'run.json')})


def stage_record(bank_path, bank, action, output, model_dir):
    command = [sys.executable, str(BLENDER), action, '--bank', str(Path(bank_path).resolve()), '--output', str(output)]
    model = {}
    if action == 'infer':
        model = fit_bindings(bank_path, bank, model_dir)
        model_audit = Path(model_dir).resolve() / 'audit.json'
        receipt = json.loads(model_audit.read_text())
        if receipt['state'] != 'verified_final_blender_stage' or receipt['action'] != 'fit' or receipt['bank_sha256'] != digest(bank_path):
            raise ValueError('Fitted blender has no matching handoff audit')
        verify(receipt['outputs'])
        model[str(model_audit)] = digest(model_audit)
        command += ['--model', str(Path(model_dir).resolve())]
    return dict(state='prepared_final_blender_stage', action=action, bank=str(Path(bank_path).resolve()),
                bank_sha256=digest(bank_path), command=command, model_bindings=model,
                sources={str(BLENDER): digest(BLENDER), str(Path(__file__).resolve()): digest(__file__)})


def run_stage(bank_path, action, output, model_dir=None):
    if action not in ('fit', 'infer'):
        raise ValueError('Unknown final stage')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    bank = bank_record(bank_path)
    record = stage_record(bank_path, bank, action, output, model_dir)
    started = output.with_name(output.name + '.started.json')
    write_new(started, record)
    log = output.with_name(output.name + '.command.log')
    with log.open('x') as stream:
        subprocess.run(['nice', '-n', '10', *record['command']], stdout=stream, stderr=subprocess.STDOUT, check=True)
    if record != stage_record(bank_path, bank_record(bank_path), action, output, model_dir):
        raise ValueError('Final bank, model or source changed during execution')
    run = json.loads((output / 'run.json').read_text())
    split = 'development' if action == 'fit' else 'qualifying'
    if (run['action'] != action or run['bank_sha256'] != record['bank_sha256'] or
            run['bank'] != record['bank'] or run['rows'] != bank[split]['rows'] or
            run['input_hashes'] != input_bindings(bank[split])):
        raise ValueError('Final stage run binding differs')
    verify(run['input_hashes'])
    outputs = {str(output / name): sha for name, sha in run['outputs'].items()}
    verify(outputs)
    if action == 'fit':
        fit_bindings(bank_path, bank, output)
    else:
        if set(run['outputs']) != {name + '.f32' for name in ('tree', 'neural', 'average', 'predictions')}:
            raise ValueError('Incomplete final inference outputs')
        values = {}
        for name in ('tree', 'neural', 'average', 'predictions'):
            path = output / (name + '.f32')
            if path.stat().st_size != bank['qualifying']['rows'] * 4:
                raise ValueError('Final inference count differs')
            values[name] = np.memmap(path, dtype='<f4', mode='r')
            if not np.isfinite(values[name]).all():
                raise ValueError('Nonfinite final inference')
        if (not np.allclose(values['average'], (values['tree'].astype(float) + values['neural']) / 2, atol=1e-6, rtol=0) or
                not np.array_equal(values['predictions'], np.clip(values['average'], 1, 5))):
            raise ValueError('Final 50/50 blend or clipping differs')
    outputs.update({str(output / 'run.json'): digest(output / 'run.json'), str(started): digest(started), str(log): digest(log)})
    receipt = dict(state='verified_final_blender_stage', action=action, bank_sha256=record['bank_sha256'],
                   outputs=outputs, model_bindings=record['model_bindings'], rows=bank[split]['rows'], metric=None)
    write_new(output / 'audit.json', receipt)
    return receipt


def export(bank_path, manifest_path, model_dir, inference_dir, replay_dir, output):
    bank = bank_record(bank_path)
    bank_sha = digest(bank_path)
    model = fit_bindings(bank_path, bank, model_dir)
    bound = {str(Path(bank_path).resolve()): bank_sha, **model}
    bound.update(bank['sources'])
    bound.update(json.loads((Path(model_dir) / 'model.json').read_text())['sources'])
    roots = [Path(path).resolve() for path in (inference_dir, replay_dir)]
    if roots[0] == roots[1]:
        raise ValueError('Independent repeated inference directory required')
    for root in roots:
        receipt = json.loads((root / 'audit.json').read_text())
        if (receipt['state'] != 'verified_final_blender_stage' or receipt['action'] != 'infer' or
                receipt['bank_sha256'] != bank_sha or receipt['metric'] is not None or
                receipt['rows'] != bank['qualifying']['rows']):
            raise ValueError('Final inference is not audited')
        verify(receipt['outputs'])
        verify(receipt['model_bindings'])
        bound.update(receipt['outputs'])
        bound.update(receipt['model_bindings'])
        bound[str(root / 'audit.json')] = digest(root / 'audit.json')
        if any(receipt['model_bindings'].get(path) != sha for path, sha in model.items()):
            raise ValueError('Inference used a different fitted blender')
    for name in ('tree', 'neural', 'average', 'predictions'):
        if digest(roots[0] / (name + '.f32')) != digest(roots[1] / (name + '.f32')):
            raise ValueError('Repeated full inference differs')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    data = str(manifest_path.parent / 'train-probe-qualifying.bin')
    if bank['qualifying']['dataset'] != dict(path=data, sha256=manifest['outputs']['train-probe-qualifying.bin']):
        raise ValueError('Export raw mapping refers to a different qualifying dataset')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = write_submission(manifest_path, roots[0] / 'predictions.f32', output / 'submission.txt')
    verify(bound)
    if digest(bank_path) != bank_sha:
        raise ValueError('Resolved bank changed during export')
    result.update(state='verified_final_production_submission', bank=str(Path(bank_path).resolve()),
                  bank_sha256=bank_sha, model_bindings=model,
                  inference_audits={str(root / 'audit.json'): digest(root / 'audit.json') for root in roots},
                  scope='Audited production bank, fixed-budget development-only fitted blender, two exact full '
                        'inference runs and raw-order text serialization. No qualifying labels, score or target-achievement claim.')
    result['bindings'].update(bound)
    write_new(output / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'infer', 'export'])
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--manifest')
    parser.add_argument('--inference')
    parser.add_argument('--replay')
    args = parser.parse_args()
    result = (export(args.bank, args.manifest, args.model, args.inference, args.replay, args.output)
              if args.action == 'export' else run_stage(args.bank, args.action, args.output, args.model))
    print(result['state'])
