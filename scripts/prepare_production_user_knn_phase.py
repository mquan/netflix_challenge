"""Freeze production K1 training and the retained user-kNN column/supports."""
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
    recipe_path = ROOT / 'experiments/production-user-knn-recipe.json'
    kernel_path = ROOT / 'experiments/P0-KERNEL-20261031-plan.json'
    master, recipe, kernel = [json.loads(p.read_text()) for p in (master_path, recipe_path, kernel_path)]
    verify(master['frozen'])
    if kernel['master_sha256'] != digest(master_path) or kernel['data'] != master['data']:
        raise ValueError('Kernel prerequisite uses different production data')
    evidence_path = Path(recipe['inference_evidence'])
    verify({str(evidence_path): recipe['inference_evidence_sha256'],
            recipe['source_archive']: recipe['source_archive_sha256']})
    evidence = json.loads(evidence_path.read_text())
    k1 = evidence['builds']['K1']['binary']
    original = str(ROOT / recipe['user_knn']['original_binary'])
    verify({k1: evidence['builds']['K1']['binary_sha256'], original: recipe['user_knn']['original_binary_sha256']})
    root = Path(master['output_root']) / 'user-knn'
    k1root, knnroot = root / 'K1', root / 'inference'
    if k1root.exists() or knnroot.exists():
        raise FileExistsError('Production K1/user-kNN outputs already exist')
    values = dict(REBUILT_K1_BINARY=k1, PRODUCTION_DATASET=master['data'],
                  PRODUCTION_K1_MODEL=str(k1root / 'model.bin'),
                  PRODUCTION_K1_QUERY_PREDICTIONS=str(k1root / 'model.f32'),
                  VERIFIED_USER_KNN_BINARY=original, QUALIFYING_PREDICTIONS=str(knnroot / 'model.f32'),
                  QUALIFYING_SUPPORT=str(knnroot / 'model.support.f32'), QUALIFYING_COUNTS=str(knnroot / 'model.count.f32'))
    training = [token.format(**values) for token in recipe['K1']['production_command_template']]
    inference = [token.format(**values) for token in recipe['user_knn']['production_command_template']]
    stages = [dict(name='K1-joint256-production', family='joint', command=training, output_dir=str(k1root),
                   dependency_index=None, start_epoch=1, fixed_final_epoch=41, rank=256,
                   day_rank=0, model_format=2, seed=2026, threads=6),
              dict(name='C-uknn-K1-production', family='user_knn', command=inference, output_dir=str(knnroot), threads=3)]
    if (training[training.index('--threads') + 1], inference[inference.index('--threads') + 1]) != ('6', '3'):
        raise ValueError('Historical K1/user-kNN thread counts differ')
    reference = evidence['comparisons']['full_replay']
    columns = [dict(retained=reference['.f32']['reference'], development_sha256=reference['.f32']['reference_sha256'],
                    producer_stage=1, path=str(knnroot / 'model.f32'))]
    supports = [dict(retained=reference[suffix]['reference'], development_sha256=reference[suffix]['reference_sha256'],
                     producer_stage=1, path=str(knnroot / ('model' + suffix))) for suffix in ('.support.f32', '.count.f32')]
    frozen = dict(master['frozen'])
    paths = [master_path, recipe_path, kernel_path, evidence_path, Path(recipe['source_archive']), Path(__file__),
             Path(evidence['sources_manifest']), ROOT / 'scripts/production_user_knn_audit.py',
             ROOT / 'scripts/production_user_knn_resources.py', ROOT / 'scripts/production_joint_audit.py',
             ROOT / 'scripts/production_dependency_audit.py', ROOT / 'scripts/production_rbm_audit.py',
             ROOT / 'scripts/verify_production_user_knn_fixtures.py', ROOT / 'tests/test_production_user_knn_audit.py',
             ROOT / 'tests/test_production_user_knn_resources.py',
             ROOT / 'experiments/P0-user-knn-fixture-verification-20261031.json']
    for p in paths:
        frozen[str(p)] = digest(p)
    frozen.update({k1: evidence['builds']['K1']['binary_sha256'], original: recipe['user_knn']['original_binary_sha256']})
    if digest(evidence['sources_manifest']) != evidence['sources_manifest_sha256']:
        raise ValueError('K1/user-kNN source provenance changed')
    required = [dict(path=str(Path(stage['output_dir']) / 'audit.json'), stage=index,
                     plan_sha256=digest(kernel_path), name=stage['name'], family=stage['family'],
                     fixed_epochs=6 if stage['family'] == 'local_residual' else 0,
                     qualifying_count=kernel['dataset']['query_count']) for index, stage in enumerate(kernel['stages'])]
    gate = dict(kernel_plan=str(kernel_path), kernel_plan_sha256=digest(kernel_path), required_audits=required,
                blocked_process_patterns=kernel['resource_gate']['blocked_process_patterns'] +
                    ['P0-KERNEL-20261031-run.sh', str(Path(kernel['output_root'])) + '/'],
                stable_normal_memory_seconds=30, own_threads=6, maximum_combined_threads=7, maximum_full_jobs=2,
                policy='Wait for all four kernel audits and previous CPU queues to exit. Then six-thread K1 or '
                'three-thread user-kNN alongside at most the primary/AutoRec GPU host thread. Final blender waits for all outputs.')
    python = str(Path(sys.executable).absolute())
    guard = shlex.join([python, str(ROOT / 'scripts/production_user_knn_resources.py'), '--plan', str(plan_path)])
    auditor = ROOT / 'scripts/production_user_knn_audit.py'
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 user-kNN queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard + '; then',
             '            sleep 30', '            if ' + guard + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_USER_KNN_RESOURCE_DEPENDENCIES']
    for index, stage in enumerate(stages):
        directory = Path(stage['output_dir'])
        replay = [k1, 'eval-joint', '--data', master['data'], '--model', str(directory / 'model.bin'),
                  '--predictions', str(directory / 'reload.f32')] if index == 0 else list(inference)
        if index == 1:
            for flag, suffix in [('--predictions', '.f32'), ('--support', '.support.f32'), ('--support-count', '.count.f32')]:
                replay[replay.index(flag) + 1] = str(directory / ('replay' + suffix))
        logname, replaylog = ('train.log', 'reload.log') if index == 0 else ('run.log', 'replay.log')
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(sums)) + ' > ' +
                  shlex.quote(str(Path(master['output_root']) / f'user-knn-stage-{index}-input-check.log')),
                  'wait_resources', shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path),
                                               '--stage', str(index), '--prepare']), f'echo START_USER_KNN_{index}',
                  shlex.join(['nice', '-n', '10', *stage['command']]) + ' > ' + shlex.quote(str(directory / logname)) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', *replay]) + ' > ' + shlex.quote(str(directory / replaylog)) + ' 2>&1',
                  shlex.join(['nice', '-n', '10', python, str(auditor), '--plan', str(plan_path), '--stage', str(index)]) +
                  ' > ' + shlex.quote(str(directory / 'audit.log')) + ' 2>&1', f'echo COMPLETE_AUDITED_USER_KNN_{index}']
    lines.append('echo COMPLETE_P0_USER_KNN_COLUMN_AND_TWO_SUPPORTS_ALL_FAMILY_OUTPUTS_AND_FINAL_BANK_STILL_REQUIRED')
    with runner.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner)], check=True)
    frozen[str(runner)] = digest(runner)
    verify(frozen)
    plan = dict(state='frozen_before_production_user_knn_training', master=str(master_path), master_sha256=digest(master_path),
                data=master['data'], dataset=kernel['dataset'], output_root=str(root), frozen=frozen,
                stages=stages, column_occurrences=columns, support_aliases=supports, resource_gate=gate,
                scope='New K1 at fixed epoch41/rank256 with historical six-thread Hogwild recipe, followed by '
                'exact original retained user-kNN binary with three threads and repeat inference. One prediction '
                'column plus support/count aliases. Production K1 does not replace the frozen blender latent basis. '
                'No exact multithread retraining promise, retries, query-score selection or promotion.')
    write_new(plan_path, plan)
    checks = dict(frozen, **{str(plan_path): digest(plan_path)})
    with sums.open('x') as stream:
        for path, sha in sorted(checks.items()):
            stream.write(sha + '  ' + path + '\n')
    print(json.dumps(dict(stages=2, columns=1, support_aliases=2, frozen_inputs=len(frozen), plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
