"""Freeze the retained production joint/integrated phase without launching it."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def argument(command, flag, default=None):
    return command[command.index(flag) + 1] if flag in command else default


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
    mapping_path = ROOT / 'experiments/P0-joint-stage-build-map-v2-20261031.json'
    dependency_path = ROOT / 'experiments/P0-DEPS-20261031-plan.json'
    registry_path = ROOT / 'experiments/production-joint-recipes.json'
    master, mapping, dependency, registry = [json.loads(p.read_text()) for p in (
        master_path, mapping_path, dependency_path, registry_path)]
    if digest(master_path) != mapping['master_plan_sha256'] or dependency['master_sha256'] != digest(master_path):
        raise ValueError('Production masters differ')
    if dependency['data'] != master['data'] or dependency['dataset']['query_count'] != master['qualifying_count']:
        raise ValueError('Dependency data differs')
    frozen = dict(master['frozen'])
    frozen.update(mapping['frozen_bindings'])
    for path in [master_path, mapping_path, dependency_path, registry_path,
                 ROOT / 'scripts/prepare_production_joint_phase.py', ROOT / 'scripts/production_joint_resources.py',
                 ROOT / 'scripts/production_joint_audit.py', ROOT / 'scripts/production_dependency_audit.py',
                 ROOT / 'scripts/production_rbm_audit.py', ROOT / 'scripts/production_graph.py',
                 ROOT / 'tests/test_production_joint_audit.py', ROOT / 'tests/test_production_joint_resources.py']:
        frozen[str(path)] = digest(path)
    names = [s['name'] for s in mapping['stages']]
    if names != [s['name'] for s in registry['stages']] or len(names) != 12:
        raise ValueError('Joint stage coverage differs')
    root = Path(master['output_root']) / 'joint'
    directories = {name: root / name for name in names}
    graph_stage = dependency['stages'][2]
    if graph_stage['kind'] != 'graph64':
        raise ValueError('Production graph phase differs')
    graph_path = str(Path(graph_stage['output_dir']) / 'derived/neighbors64.bin')
    stages = []
    for index, (mapped, recipe) in enumerate(zip(mapping['stages'], registry['stages'])):
        # Reconstruct directly from the archived recipe so no hyperparameter can
        # drift through an intermediate mapping. Only paths, binary, and an
        # explicit one-thread flag are substituted.
        command = list(recipe['production_command_template'])
        command[0] = mapped['binary']
        for j, token in enumerate(command):
            token = token.replace('{PRODUCTION_DATASET}', master['data']).replace('{PRODUCTION_GRAPH64}', graph_path)
            for name, directory in directories.items():
                token = token.replace('{OUTPUT_DIRECTORY}/' + name + '.bin', str(directory / 'model.bin'))
                token = token.replace('{OUTPUT_DIRECTORY}/' + name + '.f32', str(directory / 'model.f32'))
            if '{' in token or '}' in token:
                raise ValueError('Unresolved joint dependency')
            command[j] = token
        if '--threads' not in command:
            command += ['--threads', '1']
        if int(argument(command, '--threads')) != 1 or mapped['threads'] != 1:
            raise ValueError('Revalidate thread scheduling')
        if '--fixed-epoch' not in command or argument(command, '--patience') != '0':
            raise ValueError('Fixed-budget joint command required')
        if (int(argument(command, '--iterations')) != mapped['fixed_final_epoch'] or
                int(argument(command, '--factors')) != mapped['rank'] or mapped['model_format'] != 2):
            raise ValueError('Budget, rank or observed output format differs')
        parent = mapped['dependency']
        parent_index = names.index(parent) if parent else None
        if parent_index is not None:
            if parent_index >= index or argument(command, '--init-model') != str(directories[parent] / 'model.bin'):
                raise ValueError('Resume parent differs')
            if mapped['name'] == 'J4-joint128-day128':
                if mapped['start_epoch'] != 1 or '--start-epoch' in command or mapped['day_rank'] != 128:
                    raise ValueError('J4 must reset epoch numbering and add day factors')
            elif (mapped['start_epoch'] != mapping['stages'][parent_index]['fixed_final_epoch'] + 1 or
                  int(argument(command, '--start-epoch')) != mapped['start_epoch']):
                raise ValueError('Continuation epoch differs')
        directory = directories[mapped['name']]
        if directory.exists():
            raise FileExistsError(directory)
        frozen[mapped['binary']] = mapped['binary_sha256']
        stages.append(dict(name=mapped['name'], family=mapped['family'], output_dir=str(directory), command=command,
                           dependency_index=parent_index, start_epoch=mapped['start_epoch'],
                           fixed_final_epoch=mapped['fixed_final_epoch'], rank=mapped['rank'], day_rank=mapped['day_rank'],
                           model_format=mapped['model_format'], seed=int(argument(command, '--seed', 2026)), threads=1,
                           retained_columns=mapped['retained_columns'], source_policy=mapped['source_policy']))
    if sum(len(s['retained_columns']) for s in stages) != 13:
        raise ValueError('Retained joint-column aliases differ')
    dep_sha = digest(dependency_path)
    resource_gate = dict(dependency_plan=str(dependency_path), dependency_plan_sha256=dep_sha,
                        blocked_process_patterns=['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                            'P0-DEPS-20261031-run.sh', 'scripts/round4_boost.py', 'scripts/round8_nn_blend.py'],
                        required_dependency_audits=[dict(path=str(Path(s['output_dir']) / 'audit.json'),
                            stage=i, kind=s['kind'], plan_sha256=dep_sha) for i, s in enumerate(dependency['stages'])],
                        stable_normal_memory_seconds=30, own_threads=1, maximum_combined_threads=10,
                        policy='Replace completed single-thread dependency phase. GPU host=1 plus joint=1 plus CPU neural<=3 or RBM<=8. CPU neural and RBM queues cannot overlap each other.')
    python = str(Path(sys.executable).absolute())
    auditor, guard = ROOT / 'scripts/production_joint_audit.py', ROOT / 'scripts/production_joint_resources.py'
    guard_command = shlex.join([python, str(guard), '--plan', str(plan_path)])
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 joint queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard_command + '; then',
             '            sleep 30', '            if ' + guard_command + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_JOINT_RESOURCE_DEPENDENCIES']
    for index, stage in enumerate(stages):
        directory = Path(stage['output_dir'])
        reload = [stage['command'][0], 'eval-' + stage['family'], '--data', master['data'],
                  '--model', str(directory / 'model.bin'), '--predictions', str(directory / 'reload.f32'), '--threads', '1']
        if stage['family'] == 'integrated':
            reload += ['--graph', graph_path]
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(checksums_path)) +
                  ' > ' + shlex.quote(str(Path(master['output_root']) / f'joint-stage-{index}-input-check.log')),
                  'wait_resources', shlex.join(['nice', '-n', '10', python, str(auditor), '--plan',
                                               str(plan_path), '--stage', str(index), '--prepare']),
                  'echo ' + shlex.quote(f'START_{index}_{stage["name"]}'),
                  shlex.join(['nice', '-n', '10', *stage['command']]) + ' > ' + shlex.quote(str(directory / 'train.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', *reload]) + ' > ' + shlex.quote(str(directory / 'reload.log')) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(directory / 'audit.log')) + ' 2>&1',
                  'echo ' + shlex.quote(f'COMPLETE_AUDITED_{index}_{stage["name"]}')]
    lines += ['echo COMPLETE_P0_JOINT_THIRTEEN_COLUMNS_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED']
    with runner_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner_path)], check=True)
    frozen[str(runner_path)] = digest(runner_path)
    verify(frozen)
    plan = dict(state='frozen_before_production_joint_training', master=str(master_path), master_sha256=digest(master_path),
                stage_build_mapping=str(mapping_path), data=master['data'], dataset=dependency['dataset'], output_root=str(root),
                frozen=frozen, stages=stages, resource_gate=resource_gate,
                graph_dependency=dict(plan=str(dependency_path), plan_sha256=dep_sha, stage=2,
                                      audit=str(Path(graph_stage['output_dir']) / 'audit.json'), graph=graph_path, neighbors=64),
                scope='Twelve fixed-budget joint/integrated stages produce thirteen retained column occurrences. Use new production parents and mandatory integrated sidecars. No retries, query-score selection or promotion; remaining families, support arrays and final blender/export are still required.')
    with plan_path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    checksums = dict(frozen)
    checksums[str(plan_path)] = digest(plan_path)
    with checksums_path.open('x') as stream:
        for path, expected in sorted(checksums.items()):
            stream.write(expected + '  ' + path + '\n')
    print(json.dumps(dict(state=plan['state'], stages=12, retained_columns=13,
                          frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
