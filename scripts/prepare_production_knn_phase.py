"""Freeze all retained production item-kNN recipes and the ordered alias handoff."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def argument(command, flag):
    return command[command.index(flag) + 1]


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
    mapping_path = ROOT / 'experiments/P0-knn-column-map-20261031.json'
    registry_path = ROOT / 'experiments/production-knn-recipes.json'
    master, mapping, registry = [json.loads(p.read_text()) for p in (master_path, mapping_path, registry_path)]
    verify(mapping['frozen_bindings'])
    if mapping['master_plan_sha256'] != digest(master_path):
        raise ValueError('Production master changed')
    source_phases = dict(dependencies=dict(plan=mapping['dependency_plan'], plan_sha256=mapping['dependency_plan_sha256']),
                         joint=dict(plan=mapping['joint_plan'], plan_sha256=mapping['joint_plan_sha256']))
    sources = {}
    for name, spec in source_phases.items():
        if digest(spec['plan']) != spec['plan_sha256']:
            raise ValueError('Source phase changed')
        sources[name] = json.loads(Path(spec['plan']).read_text())
        if sources[name]['master_sha256'] != digest(master_path) or sources[name]['data'] != master['data']:
            raise ValueError('Source phase uses different production data')
    if sources['joint']['dataset'] != sources['dependencies']['dataset']:
        raise ValueError('Source phase datasets differ')
    stages = mapping['stages']
    if len(stages) != 28:
        raise ValueError('Expected 28 unique item-kNN recipes')
    for index, stage in enumerate(stages):
        command, output = stage['command'], Path(stage['output_dir'])
        if stage['index'] != index or stage['threads'] != 1 or output.exists():
            raise ValueError('Stage index/thread count differs or output exists')
        expected = [('--data', master['data']), ('--graph', stage['graph_dependency']['graph']),
                    ('--predictions', str(output / 'model.f32')), ('--support', str(output / 'support.f32'))]
        base = stage['base_dependency']
        expected.append(('--joint-model' if base['phase'] == 'joint' else '--time-model', base['model']))
        if any(argument(command, flag) != value for flag, value in expected):
            raise ValueError('kNN paths differ from the production mapping')
    by_hash = {s['original_prediction_sha256']: i for i, s in enumerate(stages)}
    occurrences = []
    for column in registry['columns']:
        index = by_hash[column['prediction_sha256']]
        if column['retained'] not in stages[index]['retained_columns']:
            raise ValueError('Retained column producer differs')
        occurrences.append(dict(retained=column['retained'], development_sha256=column['prediction_sha256'], producer_stage=index))
    if len(occurrences) != 29 or len(mapping['support_aliases']) != 6:
        raise ValueError('Retained alias coverage differs')
    frozen = dict(master['frozen'])
    frozen.update(mapping['frozen_bindings'])
    for path in [master_path, mapping_path, registry_path, Path(__file__),
                 ROOT / 'scripts/production_knn_audit.py', ROOT / 'scripts/production_knn_resources.py',
                 ROOT / 'scripts/production_knn_handoff.py', ROOT / 'scripts/production_rbm_audit.py',
                 ROOT / 'scripts/verify_production_knn_fixtures.py',
                 ROOT / 'experiments/P0-knn-fixture-verification-20261031.json',
                 ROOT / 'tests/test_production_knn_audit.py', ROOT / 'tests/test_production_knn_resources.py',
                 ROOT / 'tests/test_production_knn_handoff.py']:
        frozen[str(path)] = digest(path)
    required_audits = []
    for name in ('dependencies', 'joint'):
        for index, stage in enumerate(sources[name]['stages']):
            expected = dict(path=str(Path(stage['output_dir']) / 'audit.json'), stage=index,
                            plan_sha256=source_phases[name]['plan_sha256'])
            if name == 'joint':
                expected.update(name=stage['name'], final_epoch=stage['fixed_final_epoch'])
            else:
                expected['kind'] = stage['kind']
            required_audits.append(expected)
    if len(required_audits) != 16:
        raise ValueError('Source stage audit coverage differs')
    python = str(Path(sys.executable).absolute())
    auditor, guard, resolver = [ROOT / 'scripts' / name for name in (
        'production_knn_audit.py', 'production_knn_resources.py', 'production_knn_handoff.py')]
    guard_command = shlex.join([python, str(guard), '--plan', str(plan_path)])
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 kNN queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard_command + '; then',
             '            sleep 30', '            if ' + guard_command + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_KNN_RESOURCE_DEPENDENCIES']
    for index, stage in enumerate(stages):
        output = Path(stage['output_dir'])
        replay = list(stage['command'])
        replay[replay.index('--predictions') + 1] = str(output / 'replay.f32')
        replay[replay.index('--support') + 1] = str(output / 'replay-support.f32')
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(checksums_path)) +
                  ' > ' + shlex.quote(str(Path(master['output_root']) / f'knn-stage-{index}-input-check.log')),
                  'wait_resources', shlex.join(['nice', '-n', '10', python, str(auditor), '--plan',
                                               str(plan_path), '--stage', str(index), '--prepare']),
                  f'echo START_KNN_{index}',
                  shlex.join(['nice', '-n', '10', *stage['command']]) + ' > ' + shlex.quote(str(output / 'run.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', *replay]) + ' > ' + shlex.quote(str(output / 'replay.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(output / 'audit.log')) + ' 2>&1', f'echo COMPLETE_AUDITED_KNN_{index}']
    lines += [shlex.join(['nice', '-n', '10', python, str(resolver), '--plan', str(plan_path)]),
              'echo COMPLETE_P0_KNN_29_COLUMNS_6_SUPPORT_ALIASES_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED']
    with runner_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner_path)], check=True)
    frozen[str(runner_path)] = digest(runner_path)
    verify(frozen)
    output_root = Path(master['output_root']) / 'knn'
    resolved = output_root / 'resolved-columns.json'
    if resolved.exists():
        raise FileExistsError(resolved)
    plan = dict(state='frozen_before_production_knn_inference', master=str(master_path), master_sha256=digest(master_path),
                data=master['data'], dataset=sources['joint']['dataset'], frozen=frozen, source_phases=source_phases,
                stages=stages, column_occurrences=occurrences, support_aliases=mapping['support_aliases'],
                output_root=str(output_root), resolved_manifest=str(resolved),
                resource_gate=dict(required_audits=required_audits, stable_normal_memory_seconds=30,
                    blocked_process_patterns=['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                        'P0-JOINT-20261031-run.sh', 'P0-DEPS-20261031-run.sh',
                        'scripts/round4_boost.py', 'scripts/round8_nn_blend.py'],
                    own_threads=1, maximum_combined_threads=10,
                    policy='Replace the completed joint thread after all 16 source-stage audits. GPU=1 plus kNN=1 plus CPU neural<=3 or RBM<=8. CPU neural and RBM queues cannot overlap.'),
                scope='28 exact native recipes, 29 retained prediction occurrences and six support aliases. Repeat inference and audit each stage, then preserve original alias order in a resolved subset manifest. No retries, query-score selection or promotion; other families and final blender/export remain required.')
    write_plan = json.dumps(plan, indent=2) + '\n'
    with plan_path.open('x') as stream:
        stream.write(write_plan)
    checksums = dict(frozen)
    checksums[str(plan_path)] = digest(plan_path)
    with checksums_path.open('x') as stream:
        for path, expected in sorted(checksums.items()):
            stream.write(expected + '  ' + path + '\n')
    print(json.dumps(dict(state=plan['state'], stages=28, columns=29, support_aliases=6,
                          frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
