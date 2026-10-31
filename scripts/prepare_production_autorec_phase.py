"""Freeze two AutoRec MPS trajectories behind primary-GPU and device-parity gates."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--checksums', type=Path, required=True)
    args = parser.parse_args()
    plan_path, runner_path, checksums_path = (p.resolve() for p in (args.plan, args.runner, args.checksums))
    if any(p.exists() for p in (plan_path, runner_path, checksums_path)):
        raise FileExistsError('Choose new phase files')
    master_path = ROOT / 'experiments/P0-production-20261031-plan.json'
    primary_path = ROOT / 'experiments/P0-GPU-20261031-plan.json'
    registry_path = ROOT / 'experiments/production-autorec-recipes.json'
    source_proof_path = ROOT / 'experiments/P0-autorec-source-verification-20261031.json'
    master, primary, registry, source_proof = [json.loads(p.read_text()) for p in (
        master_path, primary_path, registry_path, source_proof_path)]
    if primary['master_sha256'] != digest(master_path) or primary['data'] != master['data']:
        raise ValueError('Primary GPU phase uses a different production master')
    if len(primary['stages']) != 7:
        raise ValueError('Revalidate primary GPU completion gate')
    source_root = Path(source_proof['source_root'])
    source = source_root / 'autorec.py'
    frozen = dict(master['frozen'])
    for name, expected in source_proof['sources'].items():
        frozen[str(source_root / name)] = expected
    frozen[source_proof['archive']] = source_proof['archive_sha256']
    for path in [master_path, primary_path, registry_path, source_proof_path,
                 ROOT / 'scripts/production_autorec_source.py', ROOT / 'scripts/production_autorec_audit.py',
                 ROOT / 'scripts/production_autorec_resources.py', ROOT / 'scripts/verify_production_autorec_device.py',
                 Path(__file__), ROOT / 'scripts/production_rbm_audit.py',
                 ROOT / 'tests/test_production_autorec_source.py', ROOT / 'tests/test_production_autorec_audit.py',
                 ROOT / 'tests/test_production_autorec_resources.py', ROOT / 'tests/test_joint.py',
                 ROOT / 'tests/test_native_fixed_epochs.py',
                 ROOT / 'experiments/P0-autorec-device-CPU-verification-20261031.json',
                 ROOT / 'experiments/P0-autorec-handoff-verification-v2-20261031.json']:
        frozen[str(path)] = digest(path)
    root = Path(master['output_root'])
    verification_output = root / 'autorec-device-mps'
    verification_report = root / 'autorec-device-mps-verification.json'
    if verification_output.exists() or verification_report.exists():
        raise FileExistsError('MPS verification output already exists')
    groups = ['A2-autorec512-norm', 'A3-autorec512-reg']
    python = str(Path(sys.executable).absolute())
    launcher = ROOT / 'scripts/production_autorec_source.py'
    stages, lookup = [], {}
    defaults = dict(hidden=512, layers=2, epochs=60, fixed_epoch=True, save_epoch=[], batch=512,
                    lr=.0003, weight_decay=0.0, input_drop=.25, dropout=.5, patience=5, seed=2026,
                    device='mps', reload_checkpoint=None)
    options = {'--hidden': ('hidden', int), '--layers': ('layers', int), '--batch': ('batch', int),
               '--lr': ('lr', float), '--weight-decay': ('weight_decay', float),
               '--input-drop': ('input_drop', float), '--dropout': ('dropout', float),
               '--patience': ('patience', int), '--seed': ('seed', int)}
    for index, group in enumerate(groups):
        recipes = [r for r in registry['recipes'] if r['training_group'] == group]
        if not recipes or any(r['original_command'] != recipes[0]['original_command'] for r in recipes):
            raise ValueError('Shared AutoRec trajectory recipes differ')
        output = root / 'autorec' / group
        if output.exists():
            raise FileExistsError(output)
        config = dict(defaults, data=master['data'], predictions=str(output / 'model.f32'))
        original = recipes[0]['original_command'][2:]
        for i in range(0, len(original), 2):
            flag, value = original[i:i + 2]
            if flag in ('--data', '--predictions'):
                continue
            if flag not in options:
                raise ValueError('Unrecognized historical AutoRec option: ' + flag)
            field, convert = options[flag]
            config[field] = convert(value)
        selected = sorted(r['fixed_epoch'] for r in recipes)
        config['epochs'], config['save_epoch'] = selected[-1], selected[:-1]
        if (group == groups[0] and selected != [1]) or (group == groups[1] and selected != [14, 44]):
            raise ValueError('Retained AutoRec budgets differ')
        cli = ['--data', config['data'], '--predictions', config['predictions'], '--fixed-epoch']
        for flag, field in [('--hidden', 'hidden'), ('--layers', 'layers'), ('--epochs', 'epochs'),
                            ('--batch', 'batch'), ('--lr', 'lr'), ('--weight-decay', 'weight_decay'),
                            ('--input-drop', 'input_drop'), ('--dropout', 'dropout'),
                            ('--patience', 'patience'), ('--seed', 'seed'), ('--device', 'device')]:
            cli += [flag, str(config[field])]
        for epoch in config['save_epoch']:
            cli += ['--save-epoch', str(epoch)]
        command = [python, str(launcher), 'run', '--source', str(source), '--threads', '1', '--', *cli]
        outputs = []
        for recipe in sorted(recipes, key=lambda r: r['fixed_epoch']):
            epoch = recipe['fixed_epoch']
            suffix = f'.e{epoch}' if epoch != config['epochs'] else ''
            item = dict(epoch=epoch, prediction=f'model{suffix}.f32', checkpoint=f'model{suffix}.pt',
                        replay=f'replay{suffix}.f32', replay_log=f'replay{suffix}.log',
                        column=recipe['column'], development_sha256=recipe['column_sha256'])
            outputs.append(item)
            lookup[recipe['column']] = dict(producer_stage=index, **item)
        stages.append(dict(group=group, output_dir=str(output), config=config, command=command, outputs=outputs))
    import torch
    torch_version = str(torch.__version__)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Production AutoRec requires MPS; no CPU substitution')
    guard, checker, auditor = [ROOT / 'scripts' / name for name in (
        'production_autorec_resources.py', 'verify_production_autorec_device.py', 'production_autorec_audit.py')]
    parity_command = [python, str(checker), '--source-dir', str(source_root), '--output-dir', str(verification_output),
                      '--report', str(verification_report), '--device', 'mps']
    primary_sha = digest(primary_path)
    required = [dict(path=str(Path(s['output_dir']) / 'audit.json'), stage=i,
                    column=s['expected']['column'], fixed_epochs=s['expected']['fixed_epochs'],
                    qualifying_count=primary['qualifying_count'], plan_sha256=primary_sha)
                for i, s in enumerate(primary['stages'])]
    guard_command = shlex.join([python, str(guard), '--plan', str(plan_path)])
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 AutoRec queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard_command + '; then',
             '            sleep 30', '            if ' + guard_command + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'wait_resources',
             'shasum -a 256 --check ' + shlex.quote(str(checksums_path)) + ' > ' +
             shlex.quote(str(root / 'autorec-device-input-check.log')), 'wait_resources',
             'echo START_AUTOREC_MPS_PARITY_CHECK',
             shlex.join(['nice', '-n', '10', *parity_command]) + ' > ' + shlex.quote(str(root / 'autorec-device-mps.log')) + ' 2>&1',
             'echo COMPLETE_AUTOREC_MPS_PARITY_CHECK']
    for index, stage in enumerate(stages):
        output = Path(stage['output_dir'])
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(checksums_path)) + ' > ' +
                  shlex.quote(str(root / f'autorec-stage-{index}-input-check.log')), 'wait_resources',
                  shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index), '--prepare']),
                  f'echo START_AUTOREC_{index}', shlex.join(['nice', '-n', '10', *stage['command']]) +
                  ' > ' + shlex.quote(str(output / 'train.log')) + ' 2>&1']
        for item in stage['outputs']:
            replay = list(stage['command'])
            replay[replay.index('--predictions') + 1] = str(output / item['replay'])
            replay += ['--reload-checkpoint', str(output / item['checkpoint'])]
            lines.append(shlex.join(['nice', '-n', '10', *replay]) + ' > ' + shlex.quote(str(output / item['replay_log'])) + ' 2>&1')
        lines += [shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(output / 'audit.log')) + ' 2>&1', f'echo COMPLETE_AUDITED_AUTOREC_{index}']
    lines += ['echo COMPLETE_P0_AUTOREC_THREE_COLUMNS_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED']
    with runner_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner_path)], check=True)
    frozen[str(runner_path)] = digest(runner_path)
    verify(frozen)
    plan = dict(state='frozen_before_primary_GPU_handoff_and_AutoRec_MPS_verification', master=str(master_path),
                master_sha256=digest(master_path), data=master['data'], dataset=dict(items=primary['items'],
                query_count=master['qualifying_count']), source_sha256=digest(source), torch_version=torch_version,
                numpy_version=np.__version__, frozen=frozen, stages=stages,
                column_occurrences=[lookup[r['column']] for r in registry['recipes']],
                device_verification=str(verification_report), device_verification_command=parity_command,
                resource_gate=dict(primary_gpu_plan=str(primary_path), primary_gpu_plan_sha256=primary_sha,
                    required_primary_gpu_audits=required, stable_normal_memory_seconds=30, host_threads=1,
                    blocked_process_patterns=['P0-GPU-20261031-run.sh', 'moonshot_sequence.py'] +
                        ['--column ' + s['expected']['column'] for s in primary['stages']],
                    policy='Replace the primary GPU queue only after all seven audits and its process exit. Run MPS parity first; one GPU/host thread, at most two other previously guarded CPU jobs.'),
                limitations=[registry['limitation'], 'Host thread count is fixed at one for resource control; exact historical device/library/thread reproduction is not claimed.',
                             'MPS engineering verification is not yet performed and is required before any full AutoRec trajectory.',
                             'All remaining families, final blender and qualifying export remain required.'])
    with plan_path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    checksums = dict(frozen)
    checksums[str(plan_path)] = digest(plan_path)
    with checksums_path.open('x') as stream:
        for path, expected in sorted(checksums.items()):
            stream.write(expected + '  ' + path + '\n')
    print(json.dumps(dict(state=plan['state'], stages=2, columns=3, frozen_inputs=len(frozen),
                          plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
