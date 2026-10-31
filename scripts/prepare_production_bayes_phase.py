"""Freeze the three retained production Bayes columns and B4 resume chain."""
import argparse
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys

from production_rbm_audit import digest, verify, write_new

ROOT = Path(__file__).resolve().parents[1]


def build_stages(registry, data, dataset, output_root, day_count):
    sizes = dict(user=dataset['users'], item=dataset['items'],
                 **{'user-bag': dataset['items'], 'item-bag': dataset['users'],
                    'day': day_count, 'item-bin': dataset['items'] * 30,
                    'session': dataset['sessions'], 'drift': dataset['users']})
    stages, occurrences = [], []
    for recipe in registry['recipes']:
        parent = None
        for number, source in enumerate(recipe['stages']):
            index = len(stages)
            name = Path(recipe['column']).name.removesuffix('.f32') + f'.stage{number}'
            directory = output_root / name
            if directory.exists():
                raise FileExistsError(directory)
            values = dict(BINARY=registry['binary'], DATASET=data, MODEL=str(directory / 'model.bin'),
                          PREDICTIONS=str(directory / 'model.f32'))
            if parent is not None:
                values['MODEL_SWEEP32'] = str(Path(stages[parent]['output_dir']) / 'model.bin')
            elif len(recipe['stages']) == 2:
                values['MODEL_SWEEP32'] = values['MODEL']
            command = [token.format(**values) for token in source['command_template']]
            argument = lambda flag: command[command.index(flag) + 1]
            final = int(argument('--samples'))
            groups = {name: dict(mode=value['mode'], size=sizes[name])
                      for name, value in recipe['checkpoint_metadata']['groups'].items()}
            stages.append(dict(name=name, output_dir=str(directory), command=command,
                               dependency_index=parent, rank=recipe['checkpoint_metadata']['rank'],
                               threads=int(argument('--threads')), burn_in=recipe['burn_in'], groups=groups,
                               start_sweep=stages[parent]['final_sweep'] if parent is not None else 0,
                               final_sweep=final))
            parent = index
        occurrences.append(dict(retained=recipe['column'], development_sha256=recipe['column_sha256'],
                                producer_stage=parent, path=str(Path(stages[parent]['output_dir']) / 'model.f32')))
    if ([(s['rank'], s['start_sweep'], s['final_sweep'], s['threads']) for s in stages] !=
            [(64, 0, 128, 4), (128, 0, 160, 4), (128, 0, 32, 4), (128, 32, 160, 5)]):
        raise ValueError('Retained Bayes schedule differs')
    return stages, occurrences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--checksums', type=Path, required=True)
    args = parser.parse_args()
    plan_path, runner, sums = (p.resolve() for p in (args.plan, args.runner, args.checksums))
    if any(path.exists() for path in (plan_path, runner, sums)):
        raise FileExistsError('Use new phase files')
    master_path = ROOT / 'experiments/P0-production-20261031-plan.json'
    registry_path = ROOT / 'experiments/production-bayes-recipes.json'
    rbm_path = ROOT / 'experiments/P0-RBM-20261031-plan.json'
    metadata_path = ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json'
    master, registry, rbm, metadata = [json.loads(path.read_text())
                                       for path in (master_path, registry_path, rbm_path, metadata_path)]
    verify(master['frozen'])
    if (rbm['master_sha256'] != digest(master_path) or rbm['data'] != master['data'] or
            metadata['metadata'] != rbm['dataset'] or metadata['data_sha256'] != master['frozen'][master['data']]):
        raise ValueError('Production dataset or prerequisite phase differs')
    verify({registry['binary']: registry['binary_sha256']})
    dataset = rbm['dataset']
    with Path(master['data']).open('rb') as stream:
        header = stream.read(56)
    magic, users, items, train, query, implicit, first, last, mean = struct.unpack('<8sIIQQQHH4xd', header)
    if (magic != b'NFPDAT2\0' or (users, items, train, query) !=
            (dataset['users'], dataset['items'], dataset['train_count'], dataset['query_count'])):
        raise ValueError('Production native header differs')
    stages, occurrences = build_stages(registry, master['data'], dataset,
                                       Path(master['output_root']) / 'bayes', last - first + 1)
    frozen = dict(master['frozen'])
    sources = [master_path, registry_path, rbm_path, metadata_path, Path(__file__),
               ROOT / 'scripts/production_bayes_audit.py', ROOT / 'scripts/production_bayes_resources.py',
               ROOT / 'scripts/production_rbm_audit.py', ROOT / 'scripts/recover_bayes_recipes.py',
               ROOT / 'scripts/verify_production_bayes_fixtures.py',
               ROOT / 'tests/test_production_bayes_audit.py', ROOT / 'tests/test_production_bayes_resources.py',
               ROOT / 'tests/test_bayes_production.py', ROOT / 'tests/test_bayes.py',
               ROOT / 'tests/test_native_fixed_epochs.py',
               ROOT / 'experiments/P0-bayes-budget-fixture-verification-20261031.json']
    for path in sources:
        frozen[str(path)] = digest(path)
    frozen[registry['binary']] = registry['binary_sha256']
    required = [dict(path=str(Path(stage['output_dir']) / 'audit.json'), stage=index,
                     name=stage['name'], final_epoch=stage['fixed_final_epoch'],
                     qualifying_count=dataset['query_count'], plan_sha256=digest(rbm_path))
                for index, stage in enumerate(rbm['stages'])]
    if len(required) != 12:
        raise ValueError('Expected all twelve RBM stage audits')
    blocked = list(rbm['resource_gate']['blocked_process_patterns'])
    blocked += ['P0-RBM-20261031-run.sh', str(Path(rbm['output_root'])) + '/']
    gate = dict(rbm_plan=str(rbm_path), rbm_plan_sha256=digest(rbm_path), required_audits=required,
                blocked_process_patterns=blocked, stable_normal_memory_seconds=30,
                own_threads=5, maximum_combined_threads=7, maximum_full_jobs=3,
                policy='Replace the completed RBM/CPU neural slot. At most GPU host1 + dependency/joint/kNN1 + Bayes5. '
                       'All S0/CPU neural/RBM trainers and runners must exit first. Later large CPU families must wait for this queue.')
    python = str(Path(sys.executable).absolute())
    guard = shlex.join([python, str(ROOT / 'scripts/production_bayes_resources.py'), '--plan', str(plan_path)])
    auditor = ROOT / 'scripts/production_bayes_audit.py'
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 Bayes queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard + '; then',
             '            sleep 30', '            if ' + guard + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_BAYES_RESOURCE_DEPENDENCIES',
             'wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(sums)) + ' > ' +
             shlex.quote(str(Path(master['output_root']) / 'bayes-initial-input-check.log')),
             'wait_resources',
             shlex.join(['nice', '-n', '10', python, '-m', 'unittest', 'discover', '-s', 'tests',
                         '-p', 'test_bayes_production.py', '-v']) + ' > ' +
             shlex.quote(str(Path(master['output_root']) / 'bayes-historical-thread-check.log')) + ' 2>&1']
    for index, stage in enumerate(stages):
        directory = Path(stage['output_dir'])
        reload = [registry['binary'], 'bayes', '--data', master['data'], '--init-model', str(directory / 'model.bin'),
                  '--model', str(directory / 'reload.bin'), '--predictions', str(directory / 'reload.f32'),
                  '--samples', str(stage['final_sweep']), '--burn-in', str(stage['burn_in']),
                  '--threads', str(stage['threads'])]
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(sums)) + ' > ' +
                  shlex.quote(str(Path(master['output_root']) / f'bayes-stage-{index}-input-check.log')),
                  'wait_resources', shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path),
                                               '--stage', str(index), '--prepare']), f'echo START_BAYES_{index}',
                  shlex.join(['nice', '-n', '10', *stage['command']]) + ' > ' +
                  shlex.quote(str(directory / 'train.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', *reload]) + ' > ' + shlex.quote(str(directory / 'reload.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(directory / 'audit.log')) + ' 2>&1', f'echo COMPLETE_AUDITED_BAYES_{index}']
    lines.append('echo COMPLETE_P0_BAYES_3_COLUMNS_OTHER_FAMILIES_AND_FINAL_EXPORT_STILL_REQUIRED')
    with runner.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner)], check=True)
    frozen[str(runner)] = digest(runner)
    verify(frozen)
    plan = dict(state='frozen_before_production_bayes_training', master=str(master_path), master_sha256=digest(master_path),
                data=master['data'], dataset=dataset, output_root=str(Path(master['output_root']) / 'bayes'),
                frozen=frozen, resource_gate=gate, stages=stages, column_occurrences=occurrences,
                runtime_historical_thread_check=str(Path(master['output_root']) / 'bayes-historical-thread-check.log'),
                scope='Exact retained fixed sweep budgets, feature modes, burn-in, seeds and historical thread counts. '
                      'B4 resumes only its newly audited production sweep-32 checkpoint. Posterior averages must reload '
                      'exactly; Hogwild retraining is not claimed bit-identical. No query-score selection, retries or promotion.')
    write_new(plan_path, plan)
    checks = dict(frozen, **{str(plan_path): digest(plan_path)})
    with sums.open('x') as stream:
        for path, sha in sorted(checks.items()):
            stream.write(sha + '  ' + path + '\n')
    print(json.dumps(dict(stages=4, columns=3, frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
