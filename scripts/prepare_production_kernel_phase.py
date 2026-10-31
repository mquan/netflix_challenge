"""Freeze the six retained GP/local columns using the new production J3."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from production_rbm_audit import digest, verify, write_new

ROOT = Path(__file__).resolve().parents[1]


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
    registry_path = ROOT / 'experiments/production-kernel-recipes.json'
    joint_path = ROOT / 'experiments/P0-JOINT-20261031-plan.json'
    bayes_path = ROOT / 'experiments/P0-BAYES-20261031-plan.json'
    knn_path = ROOT / 'experiments/P0-KNN-20261031-plan.json'
    master, registry, joint, bayes, knn = [json.loads(path.read_text())
                                         for path in (master_path, registry_path, joint_path, bayes_path, knn_path)]
    verify(master['frozen'])
    for phase in (joint, bayes, knn):
        if (phase['master_sha256'] != digest(master_path) or phase['data'] != master['data'] or
                any(phase['dataset'][key] != joint['dataset'][key] for key in
                    ('fingerprint', 'users', 'items', 'train_count', 'query_count', 'sessions'))):
            raise ValueError('Kernel prerequisite uses different production data')
    j3 = joint['stages'][2]
    if (j3['name'], j3['rank'], j3['fixed_final_epoch'], j3['day_rank']) != ('J3-joint128-reg005', 128, 9, 0):
        raise ValueError('Production J3 recipe differs')
    model = str(Path(j3['output_dir']) / 'model.bin')
    source = dict(plan=str(joint_path), plan_sha256=digest(joint_path), stage=2, model=model)
    frozen = dict(master['frozen'])
    stages, occurrences = [], []
    for index, recipe in enumerate(registry['recipes']):
        verify({recipe['binary']: recipe['binary_sha256']})
        frozen[recipe['binary']] = recipe['binary_sha256']
        output = Path(master['output_root']) / 'kernel' / recipe['name']
        if output.exists():
            raise FileExistsError(output)
        values = dict(VERIFIED_MOONSHOT_GP_INITIAL_BINARY=recipe['binary'],
                      VERIFIED_MOONSHOT_GP_PERSISTENT_BINARY=recipe['binary'],
                      VERIFIED_MOONSHOT_LOCAL_BINARY=recipe['binary'], PRODUCTION_DATASET=master['data'],
                      PRODUCTION_J3_MODEL=model, OUTPUT_PREFIX=str(output / 'model'))
        command = [token.format(**values) for token in recipe['production_command_template']]
        stages.append(dict(name=recipe['name'], family=recipe['family'], command=command,
                           parameters=recipe['parameters'], output_dir=str(output), threads=1))
        for column in recipe['columns']:
            original = Path(column['reference']).name
            suffix = original.removeprefix(recipe['name'])
            if (recipe['family'] == 'gp' and suffix not in ('.0.f32', '.1.f32', '.2.f32') or
                    recipe['family'] == 'local_residual' and suffix != '.epoch6.f32'):
                raise ValueError('Retained kernel suffix differs')
            occurrences.append(dict(retained=column['reference'], development_sha256=column['sha256'],
                                    producer_stage=index, path=str(output / ('model' + suffix))))
    if len(stages) != 4 or len(occurrences) != 6:
        raise ValueError('Kernel recipe coverage differs')
    sources = [master_path, registry_path, joint_path, bayes_path, knn_path, Path(__file__),
               ROOT / 'scripts/production_kernel_audit.py', ROOT / 'scripts/production_kernel_resources.py',
               ROOT / 'scripts/production_rbm_audit.py', ROOT / 'scripts/verify_production_kernel_fixtures.py',
               ROOT / 'tests/test_production_kernel_audit.py', ROOT / 'tests/test_production_kernel_resources.py',
               ROOT / 'experiments/P0-kernel-fixture-verification-20261031.json',
               ROOT / registry['source_manifest'], ROOT / registry['build_record'], Path(registry['replay_audit']),
               ROOT / 'experiments/production-kernel-sources.zip']
    for path in sources:
        frozen[str(path)] = digest(path)
    if (digest(ROOT / registry['source_manifest']) != registry['source_manifest_sha256'] or
            digest(ROOT / registry['build_record']) != registry['build_record_sha256'] or
            digest(registry['replay_audit']) != registry['replay_audit_sha256'] or
            digest(ROOT / 'experiments/production-kernel-sources.zip') != registry['source_archive_sha256']):
        raise ValueError('Kernel historical source or replay provenance differs')
    required = []
    for name, phase, path in [('bayes', bayes, bayes_path), ('knn', knn, knn_path)]:
        for index, stage in enumerate(phase['stages']):
            expected = dict(path=str(Path(stage['output_dir']) / 'audit.json'), stage=index,
                            plan_sha256=digest(path), qualifying_count=joint['dataset']['query_count'])
            if name == 'bayes':
                expected.update(name=stage['name'], final_sweep=stage['final_sweep'],
                                kept=stage['final_sweep'] - stage['burn_in'])
            else:
                expected['retained_columns'] = stage['retained_columns']
            required.append(expected)
    blocked = list(bayes['resource_gate']['blocked_process_patterns'])
    blocked += ['P0-BAYES-20261031-run.sh', 'P0-KNN-20261031-run.sh', 'P0-JOINT-20261031-run.sh',
                'P0-DEPS-20261031-run.sh', str(Path(bayes['output_root'])) + '/', str(Path(knn['output_root'])) + '/',
                str(Path(joint['output_root'])) + '/', str(Path(master['output_root']) / 'dependencies') + '/']
    gate = dict(required_audits=required, blocked_process_patterns=blocked,
                source_phases=dict(bayes=dict(plan=str(bayes_path), plan_sha256=digest(bayes_path)),
                                   knn=dict(plan=str(knn_path), plan_sha256=digest(knn_path))),
                stable_normal_memory_seconds=30, own_threads=1, maximum_combined_threads=2,
                maximum_full_jobs=2, policy='Wait for Bayes4 and item-kNN28 audits and their runners/trainers to exit. '
                'Then run one kernel stage at a time alongside at most the primary/AutoRec GPU host thread. '
                'User-kNN must follow this queue.')
    python = str(Path(sys.executable).absolute())
    guard = shlex.join([python, str(ROOT / 'scripts/production_kernel_resources.py'), '--plan', str(plan_path)])
    auditor = ROOT / 'scripts/production_kernel_audit.py'
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 kernel queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard + '; then',
             '            sleep 30', '            if ' + guard + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_KERNEL_RESOURCE_DEPENDENCIES']
    for index, stage in enumerate(stages):
        root = Path(stage['output_dir'])
        replay = list(stage['command'])
        replay[replay.index('--output') + 1] = str(root / 'replay')
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(sums)) + ' > ' +
                  shlex.quote(str(Path(master['output_root']) / f'kernel-stage-{index}-input-check.log')),
                  'wait_resources', shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path),
                                               '--stage', str(index), '--prepare']), f'echo START_KERNEL_{index}',
                  shlex.join(['nice', '-n', '10', *stage['command']]) + ' > ' + shlex.quote(str(root / 'run.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', *replay]) + ' > ' + shlex.quote(str(root / 'replay.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(root / 'audit.log')) + ' 2>&1', f'echo COMPLETE_AUDITED_KERNEL_{index}']
    lines.append('echo COMPLETE_P0_KERNEL_6_COLUMNS_USER_KNN_AND_FINAL_BANK_STILL_REQUIRED')
    with runner.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner)], check=True)
    frozen[str(runner)] = digest(runner)
    verify(frozen)
    plan = dict(state='frozen_before_production_kernel_exports', master=str(master_path), master_sha256=digest(master_path),
                data=master['data'], dataset=joint['dataset'], output_root=str(Path(master['output_root']) / 'kernel'),
                source_phase=source, frozen=frozen, stages=stages, column_occurrences=occurrences, resource_gate=gate,
                scope='Six retained GP/local columns from four exact archived-binary commands using only new audited '
                'production J3. Repeat complete inference/six-epoch training and verify all exports, coverage and '
                'base identity. Local factors are not persisted. No retries, query-score selection or promotion.')
    write_new(plan_path, plan)
    checks = dict(frozen, **{str(plan_path): digest(plan_path)})
    with sums.open('x') as stream:
        for path, sha in sorted(checks.items()):
            stream.write(sha + '  ' + path + '\n')
    print(json.dumps(dict(stages=4, columns=6, frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
