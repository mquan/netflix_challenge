"""Freeze final bank resolution, fixed-budget fitting, two inferences and export."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from production_rbm_audit import digest, verify, write_new
from resolve_production_bank import COUNTS, expected_receipt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--checksums', type=Path, required=True)
    args = parser.parse_args()
    plan_path, runner, sums = (p.resolve() for p in (args.plan, args.runner, args.checksums))
    if any(p.exists() for p in (plan_path, runner, sums)):
        raise FileExistsError('Use new final phase files')
    master_path = ROOT / 'experiments/P0-production-20261031-plan.json'
    mapping_path = ROOT / 'experiments/P0-all-family-planned-coverage-20261031.json'
    bank_path = ROOT / 'experiments/production-blender-plan.json'
    master, mapping, bank = [json.loads(p.read_text()) for p in (master_path, mapping_path, bank_path)]
    if ({key: len(value) for key, value in bank['schema'].items()} != dict(predictions=79, auxiliary=64, context=5) or
            bank['settings']['budgets'] != dict(tree=[dict(seed=2026, steps=904), dict(seed=7, steps=969), dict(seed=13, steps=1106)],
                                               neural=[dict(seed=2026, steps=15), dict(seed=7, steps=14), dict(seed=13, steps=14)])):
        raise ValueError('Retained final schema or median budgets differ')
    if bank['settings']['neural_config']['threads'] != 2:
        raise ValueError('Retained final neural thread count differs')
    verify(master['frozen'])
    verify(mapping['frozen'])
    frozen = dict(master['frozen'])
    frozen.update(mapping['frozen'])
    required, blocked = [], ['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                             'scripts/round4_boost.py', 'scripts/round8_nn_blend.py']
    for name, spec in mapping['phases'].items():
        if digest(spec['plan']) != spec['plan_sha256']:
            raise ValueError('Production phase changed')
        source = json.loads(Path(spec['plan']).read_text())
        if source['master_sha256'] != digest(master_path) or len(source['stages']) != COUNTS[name]:
            raise ValueError('Production phase master or stage count differs')
        for path, sha in source['frozen'].items():
            if path in frozen and frozen[path] != sha:
                raise ValueError('Conflicting frozen final input')
            frozen[path] = sha
        frozen[spec['plan']] = spec['plan_sha256']
        blocked.append(f'P0-{name}-20261031-run.sh')
        for index, stage in enumerate(source['stages']):
            expected = expected_receipt(name, source, index, master['qualifying_count'])
            required.append(dict(expected, path=str(Path(stage['output_dir']) / 'audit.json'),
                                 plan_sha256=spec['plan_sha256']))
            blocked.append(str(Path(stage['output_dir'])) + '/')
    if len(required) != 82:
        raise ValueError('All 82 prerequisite stage audits required')
    paths = [master_path, mapping_path, bank_path, Path(__file__),
             *[ROOT / 'scripts' / name for name in ('resolve_production_bank.py', 'production_final_handoff.py',
                'production_submission.py', 'production_final_resources.py')],
             *[ROOT / 'tests' / name for name in ('test_resolve_production_bank.py', 'test_production_final_handoff.py',
                'test_production_submission.py', 'test_production_final_resources.py')],
             *[ROOT / 'experiments' / name for name in ('P0-bank-resolver-verification-20261031.json',
                'P0-final-handoff-verification-20261031.json', 'P0-submission-order-verification-20261031.json')]]
    for path in paths:
        frozen[str(path)] = digest(path)
    root = Path(master['output_root'])
    resolved, model, inference, replay, exported = [root / name for name in
        ('resolved-bank.json', 'final-blender', 'final-inference', 'final-inference-replay', 'final-export')]
    if any(path.exists() for path in (resolved, model, inference, replay, exported)):
        raise FileExistsError('Final production outputs already exist')
    python = str(Path(sys.executable).absolute())
    helper = str(ROOT / 'scripts/production_final_handoff.py')
    commands = [dict(name='resolve-bank', command=[python, str(ROOT / 'scripts/resolve_production_bank.py'),
                    '--mapping', str(mapping_path), '--bank', str(bank_path), '--output', str(resolved)]),
                dict(name='fit', command=[python, helper, 'fit', '--bank', str(resolved), '--output', str(model)]),
                dict(name='infer', command=[python, helper, 'infer', '--bank', str(resolved), '--model', str(model), '--output', str(inference)]),
                dict(name='infer-replay', command=[python, helper, 'infer', '--bank', str(resolved), '--model', str(model), '--output', str(replay)]),
                dict(name='export', command=[python, helper, 'export', '--bank', str(resolved), '--model', str(model),
                     '--manifest', master['dataset_manifest'], '--inference', str(inference), '--replay', str(replay), '--output', str(exported)])]
    guard = shlex.join([python, str(ROOT / 'scripts/production_final_resources.py'), '--plan', str(plan_path)])
    lines = ['#!/bin/bash', 'set -eu', 'cd ' + shlex.quote(str(ROOT)),
             'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
             'trap \'echo "P0 final queue exited with status $?"\' EXIT',
             'wait_resources() {', '    while true; do', '        if ' + guard + '; then',
             '            sleep 30', '            if ' + guard + '; then return 0; fi',
             '        else sleep 30; fi', '    done', '}', 'echo WAIT_P0_FINAL_ALL_PRODUCTION_STAGES']
    for stage in commands:
        name = stage['name']
        lines += ['wait_resources', 'shasum -a 256 --check ' + shlex.quote(str(sums)) + ' > ' +
                  shlex.quote(str(root / f'final-{name}-input-check.log')), 'wait_resources', f'echo START_FINAL_{name}',
                  shlex.join(['nice', '-n', '10', *stage['command']]) + ' > ' +
                  shlex.quote(str(root / f'final-{name}.log')) + ' 2>&1', f'echo COMPLETE_FINAL_{name}']
    lines += ['shasum -a 256 --check ' + shlex.quote(str(sums)) + ' > ' + shlex.quote(str(root / 'final-post-export-input-check.log')),
              'echo COMPLETE_FIXED_PRODUCTION_BANK_BLENDER_AND_RAW_ORDER_EXPORT_NO_HIDDEN_SCORE_OR_TARGET_CLAIM']
    with runner.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(runner)], check=True)
    frozen[str(runner)] = digest(runner)
    verify(frozen)
    plan = dict(state='frozen_before_final_production_bank_blender_export', master=str(master_path), master_sha256=digest(master_path),
                mapping=str(mapping_path), mapping_sha256=digest(mapping_path), original_bank=str(bank_path),
                original_bank_sha256=digest(bank_path), source_phases=mapping['phases'], frozen=frozen,
                settings=bank['settings'], schema=bank['schema'], qualifying_count=master['qualifying_count'],
                commands=commands, resolved_bank=str(resolved), final_model=str(model), inference=str(inference),
                replay=str(replay), export_dir=str(exported), submission=str(exported / 'submission.txt'),
                resource_gate=dict(required_audits=required, blocked_process_patterns=blocked,
                    stable_normal_memory_seconds=30, own_threads=2, maximum_full_jobs=1,
                    policy='Wait for all 82 production audits, every production/S0 runner and stage process to exit, '
                    'and stable normal memory. Final CPU blender uses at most two threads; no GPU allocation. '
                    'Other-session GPU work remains independently owned.'),
                scope='Resolve all 79 columns/eight supports, fit frozen development-only median budgets, '
                'run two exact full qualifying inferences and export in verified raw qualifying order. '
                'No retries, overwrites, query-label fitting, hidden score or target-achievement claim.')
    write_new(plan_path, plan)
    checks = dict(frozen, **{str(plan_path): digest(plan_path)})
    with sums.open('x') as stream:
        for path, sha in sorted(checks.items()):
            stream.write(sha + '  ' + path + '\n')
    print(json.dumps(dict(frozen_inputs=len(frozen), required_stage_audits=82, final_commands=5,
                          plan_sha256=digest(plan_path), launched=False)))


if __name__ == '__main__':
    main()
