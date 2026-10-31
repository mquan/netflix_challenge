"""Freeze the serialized production RBM phase without launching training."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent.parent / '.cache/netflix-venv/bin/python'


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
    mapping_path = ROOT / 'experiments/P0-rbm-stage-build-map-20261031.json'
    metadata_path = ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json'
    cpu_path = ROOT / 'experiments/P0-CPU-neural-20261031-plan.json'
    registry_path = ROOT / 'experiments/production-rbm-recipes-v2.json'
    master, mapping, metadata, cpu, registry = [json.loads(p.read_text()) for p in (
        master_path, mapping_path, metadata_path, cpu_path, registry_path)]
    if digest(master_path) != mapping['master_plan_sha256'] or digest(registry_path) != mapping['registry_sha256']:
        raise ValueError('Production master or recipe registry changed')
    frozen = dict(master['frozen'])
    frozen.update(mapping['frozen_bindings'])
    required = [master_path, mapping_path, metadata_path, cpu_path, registry_path,
                ROOT / 'experiments/P0-rbm-complete-sources-20261031.zip',
                ROOT / 'experiments/P0-rbm-handoff-verification-20261031.json',
                ROOT / 'scripts/production_native_metadata.cpp',
                ROOT / 'scripts/production_rbm_audit.py', ROOT / 'scripts/production_rbm_resources.py',
                ROOT / 'scripts/prepare_production_rbm_phase.py',
                ROOT / 'tests/test_production_rbm_audit.py', ROOT / 'tests/test_production_rbm_resources.py']
    frozen.update({str(p): digest(p) for p in required})
    frozen[metadata['binary']] = metadata['binary_sha256']
    for relative, expected in metadata['native_sources'].items():
        frozen[str(Path(metadata['native_source_root']) / relative)] = expected
    if metadata['data'] != master['data'] or metadata['data_sha256'] != frozen[master['data']]:
        raise ValueError('Native metadata refers to different data')
    output_root = Path(master['output_root']) / 'rbm'
    names = [s['name'] for s in mapping['stages']]
    if names != [s['name'] for s in registry['stages']] or len(names) != 12:
        raise ValueError('RBM stage coverage or ordering differs')
    paths = {name: output_root / name for name in names}
    stages = []
    for index, mapped in enumerate(mapping['stages']):
        original = registry['stages'][index]
        if mapped['original_sources'] != original['required_historical_sources']:
            raise ValueError('Historical source mismatch')
        command = list(mapped['command_template'])
        for j, token in enumerate(command):
            token = token.replace('{PRODUCTION_DATASET}', master['data'])
            for name, directory in paths.items():
                token = token.replace('{OUTPUT_DIRECTORY}/' + name + '.bin', str(directory / 'model.bin'))
                token = token.replace('{OUTPUT_DIRECTORY}/' + name + '.f32', str(directory / 'model.f32'))
            if '{' in token or '}' in token:
                raise ValueError('Unresolved production command dependency')
            command[j] = token
        dependency = mapped['dependency']
        dependency_index = names.index(dependency) if dependency else None
        if dependency_index is not None and dependency_index >= index:
            raise ValueError('Non-topological RBM resume dependency')
        directory = paths[mapped['name']]
        if directory.exists():
            raise FileExistsError(directory)
        if argument(command, '--data') != master['data'] or '--fixed-epoch' not in command:
            raise ValueError('Not a fixed-budget production command')
        if int(argument(command, '--iterations')) != mapped['fixed_final_epoch'] or argument(command, '--patience') != '0':
            raise ValueError('Fixed budget differs')
        if dependency:
            if (argument(command, '--init-model') != str(paths[dependency] / 'model.bin') or
                    int(argument(command, '--start-epoch')) != mapped['start_epoch'] or
                    mapped['start_epoch'] != mapping['stages'][dependency_index]['fixed_final_epoch'] + 1):
                raise ValueError('Resume boundary or parent differs')
        for relative, expected in mapped['fixed_sources'].items():
            frozen[str(Path(command[0]).parent / relative)] = expected
        frozen[command[0]] = mapped['binary_sha256']
        stages.append(dict(name=mapped['name'], output_dir=str(directory), command=command,
                           dependency_index=dependency_index, start_epoch=mapped['start_epoch'],
                           fixed_final_epoch=mapped['fixed_final_epoch'],
                           hidden=int(argument(command, '--hidden')), threads=mapped['threads'],
                           seed=int(argument(command, '--seed', 2026)),
                           conditional='--conditional-biases' in command,
                           retained_columns=[c for c in registry['retained_columns'] if c['stage'] == mapped['name']]))
    if sum(len(s['retained_columns']) for s in stages) != 9:
        raise ValueError('Retained RBM column coverage differs')
    cpu_sha = digest(cpu_path)
    resource_gate = dict(cpu_neural_plan=str(cpu_path), cpu_neural_plan_sha256=cpu_sha,
                        blocked_process_patterns=['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                            'P0-CPU-neural-20261031-run.sh', 'scripts/round4_boost.py', 'scripts/round8_nn_blend.py'] +
                            ['--column ' + s['expected']['column'] for s in cpu['stages']],
                        required_cpu_neural_audits=[dict(path=str(Path(s['output_dir']) / 'audit.json'),
                            stage=i, column=s['expected']['column'], plan_sha256=cpu_sha) for i, s in enumerate(cpu['stages'])],
                        stable_normal_memory_seconds=30, maximum_native_training_threads=8,
                        maximum_with_gpu_host_threads=9,
                        policy='Wait for both S0 lanes and all seven audited CPU neural stages. Run one RBM stage at a time alongside at most the single GPU host thread.')
    auditor = ROOT / 'scripts/production_rbm_audit.py'
    guard = ROOT / 'scripts/production_rbm_resources.py'
    python = Path(sys.executable).absolute()
    guard_command = shlex.join([str(python), str(guard), '--plan', str(plan_path)])
    runner = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
              'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
              'trap \'echo "P0 RBM queue exited with status $?"\' EXIT',
              'wait_resources() {', '    while true; do',
              '        if ' + guard_command + '; then', '            sleep 30',
              '            if ' + guard_command + '; then return 0; fi',
              '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_RBM_RESOURCE_DEPENDENCIES']
    for index, stage in enumerate(stages):
        directory = Path(stage['output_dir'])
        command = stage['command']
        reload_command = [command[0], 'eval-rbm', '--data', master['data'], '--model',
                          str(directory / 'model.bin'), '--predictions', str(directory / 'reload.f32'), '--threads', '1']
        runner += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(checksums_path)) +
                   ' > ' + shlex.quote(str(Path(master['output_root']) / f'rbm-stage-{index}-input-check.log')),
                   'wait_resources',
                   shlex.join(['nice', '-n', '10', str(python), str(auditor), '--plan', str(plan_path), '--stage', str(index), '--prepare']),
                   'echo ' + shlex.quote(f'START_{index}_{stage["name"]}'),
                   shlex.join(['nice', '-n', '10', *command]) + ' > ' + shlex.quote(str(directory / 'train.log')) + ' 2>&1',
                   shlex.join(['nice', '-n', '10', *reload_command]) + ' > ' + shlex.quote(str(directory / 'reload.log')) + ' 2>&1',
                   shlex.join(['nice', '-n', '10', str(python), str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                   ' > ' + shlex.quote(str(directory / 'audit.log')) + ' 2>&1',
                   'echo ' + shlex.quote(f'COMPLETE_AUDITED_{index}_{stage["name"]}')]
    runner += ['echo COMPLETE_P0_RBM_NINE_COLUMNS_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED']
    with runner_path.open('x') as stream:
        stream.write('\n'.join(runner) + '\n')
    subprocess.run(['bash', '-n', str(runner_path)], check=True)
    frozen[str(runner_path)] = digest(runner_path)
    verify(frozen)
    plan = dict(state='frozen_before_production_RBM_training', master=str(master_path),
                master_sha256=digest(master_path), stage_build_map=str(mapping_path),
                data=master['data'], dataset=metadata['metadata'], output_root=str(output_root),
                frozen=frozen, resource_gate=resource_gate, stages=stages,
                scope='Nine retained RBM columns via 12 fixed-budget stages. Fresh production parents, no retry, query-score selection or promotion. Remaining families, support arrays, final blender and export are still required.')
    with plan_path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    checksums = dict(frozen)
    checksums[str(plan_path)] = digest(plan_path)
    with checksums_path.open('x') as stream:
        for path, expected in sorted(checksums.items()):
            stream.write(expected + '  ' + path + '\n')
    print(json.dumps(dict(state=plan['state'], stages=len(stages), retained_columns=9,
                          frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
