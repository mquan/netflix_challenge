"""Freeze production baseline/graph/neighborhood commands without launching them."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def build_stages(output, data, binary, python):
    output = Path(output)
    base, g256, g64, nb = [output / name for name in ('baseline', 'graph256', 'graph64', 'neighborhood')]
    prefix = ROOT / 'scripts/production_graph.py'
    model, graph256, graph64 = str(base / 'model.bin'), str(g256 / 'neighbors256.bin'), str(g64 / 'derived/neighbors64.bin')
    stages = [
        dict(kind='baseline', output_dir=str(base), dependency_indices=[],
             files=['model.bin', 'train.log', 'reload.log'],
             commands=[[binary, 'time-baseline', '--data', data, '--model', model,
                        '--iterations', '40', '--rate-scale', '1'],
                       [binary, 'eval-time-baseline', '--data', data, '--model', model]]),
        dict(kind='graph256', output_dir=str(g256), dependency_indices=[],
             files=['neighbors256.bin', 'train.log'],
             commands=[[binary, 'graph', '--data', data, '--graph', graph256,
                        '--neighbors', '256', '--history-cap', '256']]),
        dict(kind='graph64', output_dir=str(g64), dependency_indices=[1], source_graph=graph256,
             files=['derived/neighbors64.bin', 'derived/manifest.json', 'train.log'],
             commands=[[python, str(prefix), '--source', graph256, '--neighbors', '64',
                        '--output-dir', str(g64 / 'derived')]]),
        dict(kind='neighborhood', output_dir=str(nb), dependency_indices=[0, 2], source_graph=graph64,
             files=['model.bin', 'model.f32', 'reload.f32', 'train.log', 'reload.log'],
             commands=[[binary, 'neighborhood', '--data', data, '--graph', graph64,
                        '--time-model', model, '--model', str(nb / 'model.bin'),
                        '--predictions', str(nb / 'model.f32'), '--iterations', '4', '--fixed-epoch',
                        '--learning-rate', '0.005', '--regularization', '0.002', '--time-decay', '0'],
                       [binary, 'eval-neighborhood', '--data', data, '--graph', graph64,
                        '--time-model', model, '--model', str(nb / 'model.bin'),
                        '--predictions', str(nb / 'reload.f32')]]),
    ]
    return stages


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
    source_path = ROOT / 'experiments/P0-dependency-source-recovery-20261031.json'
    metadata_path = ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json'
    master, source, metadata = [json.loads(p.read_text()) for p in (master_path, source_path, metadata_path)]
    dataset_manifest = json.loads(Path(master['dataset_manifest']).read_text())
    frozen = dict(master['frozen'])
    bound_paths = [master_path, source_path, metadata_path,
                   ROOT / 'experiments/PR7-time-baseline-retrain.json',
                   ROOT / 'experiments/PR7-graph256-rebuild.json',
                   ROOT / 'experiments/PR7-dependency-verification.json',
                   ROOT / 'experiments/PR8-neighborhood-retraining-verification.json',
                   ROOT / 'experiments/PN1-native-fixed-epochs.json',
                   ROOT / 'experiments/P0-dependency-format-verification-20261031.json',
                   ROOT / 'experiments/P0-dependency-handoff-verification-20261031.json',
                   ROOT / 'scripts/production_graph.py', ROOT / 'scripts/production_rbm_audit.py',
                   ROOT / 'scripts/production_dependency_audit.py', ROOT / 'scripts/production_dependency_resources.py',
                   ROOT / 'scripts/prepare_production_dependency_phase.py',
                   ROOT / 'tests/test_production_dependency_audit.py', ROOT / 'tests/test_production_dependency_resources.py']
    frozen.update({str(p): digest(p) for p in bound_paths})
    frozen[source['archive']] = source['archive_sha256']
    frozen[source['binary']] = source['binary_sha256']
    for relative, expected in source['sources'].items():
        frozen[str(Path(source['source_root']) / relative)] = expected
    if metadata['data'] != master['data'] or metadata['data_sha256'] != frozen[master['data']]:
        raise ValueError('Native metadata uses different data')
    dataset = dict(metadata['metadata'], mean=dataset_manifest['mean'])
    python = str(Path(sys.executable).absolute())
    output = Path(master['output_root']) / 'dependencies'
    stages = build_stages(output, master['data'], source['binary'], python)
    if any(Path(s['output_dir']).exists() for s in stages):
        raise FileExistsError('Production dependency output already exists')
    guard, auditor = ROOT / 'scripts/production_dependency_resources.py', ROOT / 'scripts/production_dependency_audit.py'
    gate_command = shlex.join([python, str(guard), '--plan', str(plan_path)])
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 dependency queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + gate_command + '; then',
             '            sleep 30', '            if ' + gate_command + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_DEPENDENCY_RESOURCE_GATES']
    for index, stage in enumerate(stages):
        directory = Path(stage['output_dir'])
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(checksums_path)) +
                  ' > ' + shlex.quote(str(Path(master['output_root']) / f'dependency-stage-{index}-input-check.log')),
                  'wait_resources', shlex.join(['nice', '-n', '10', python, str(auditor), '--plan',
                                               str(plan_path), '--stage', str(index), '--prepare']),
                  'echo ' + shlex.quote(f'START_{index}_{stage["kind"]}')]
        for command, logfile in zip(stage['commands'], ['train.log', 'reload.log']):
            lines += [shlex.join(['nice', '-n', '10', *command]) + ' > ' + shlex.quote(str(directory / logfile)) + ' 2>&1']
        lines += [shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(directory / 'audit.log')) + ' 2>&1',
                  'echo ' + shlex.quote(f'COMPLETE_AUDITED_{index}_{stage["kind"]}')]
    lines += ['echo COMPLETE_P0_DEPENDENCIES_AND_ONE_COLUMN_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED']
    with runner_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner_path)], check=True)
    frozen[str(runner_path)] = digest(runner_path)
    verify(frozen)
    plan = dict(state='frozen_before_production_dependency_training', master=str(master_path),
                master_sha256=digest(master_path), data=master['data'], dataset=dataset,
                output_root=str(output), frozen=frozen, stages=stages,
                resource_gate=dict(blocked_process_patterns=['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                    'scripts/round4_boost.py', 'scripts/round8_nn_blend.py'], stable_normal_memory_seconds=30,
                    own_threads=1, maximum_combined_threads=10,
                    policy='Wait for both S0 lanes and their children. One dependency thread may overlap GPU host=1 plus CPU neural<=3, or GPU host=1 plus RBM<=8; those latter queues cannot overlap each other.'),
                scope='New production temporal baseline, graph256 and its exact graph64 prefix, plus one learned-neighborhood qualifying column. No retries, score selection or promotion. Remaining native families and final blender/export are still required.')
    with plan_path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    checksums = dict(frozen)
    checksums[str(plan_path)] = digest(plan_path)
    with checksums_path.open('x') as stream:
        for path, expected in sorted(checksums.items()):
            stream.write(expected + '  ' + path + '\n')
    print(json.dumps(dict(state=plan['state'], stages=4, frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
