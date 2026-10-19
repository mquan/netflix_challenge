#!/usr/bin/env python3
"""Conditional SO1 full-training pair and matched full-bank development comparison.

Prepare only until the frozen P2 gate passes. The full split is original training
with probe still held out, not final train+probe production. No automatic promotion.
"""
import argparse
import datetime
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from gbdt_blend import load
from local_context_chain import arguments, paired, read, resolve, write
from selection_objective_queue import inspect_idle, verify
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]


def require_gate(gate, plan_sha, threshold):
    delta = gate['primary']['delta_rmse']
    if gate['plan_sha256'] != plan_sha or gate['passes_full_refit_gate'] is not True:
        raise ValueError('P2 gate is not a pass for this frozen plan')
    if not math.isfinite(delta) or delta > -threshold:
        raise ValueError('P2 improvement does not meet the frozen threshold')


def make_stages(p2, base, neural, root, data, warm_report, warm_checkpoint, output, run_id, processes):
    stages, predictions = [], {}
    for arm in ('control', 'treatment'):
        cfg = dict(p2['settings'], data=str(data), warm_report=str(warm_report),
            warm_checkpoint=str(warm_checkpoint), warm_stage='refit',
            weight=p2['arms'][arm]['weight'], output=str(output / arm))
        stages.append(dict(name=f'{run_id}-{arm}', kind='training', arm=arm,
            config=cfg, command=arguments('selection_objective.py', cfg)))
        predictions[arm] = cfg['output']+f".e{p2['score_epoch']}.f32"
    for arm, added in (('duplicate-control', [predictions['control']]*2),
                       ('treatment', [predictions['control'], predictions['treatment']])):
        cfg = dict(base['config'])
        cfg.update(data=str(data), bank_root=str((root / cfg['bank_root']).resolve()),
            extra=[str(resolve(p, base['config'], root)) for p in cfg['extra']]+added,
            extra_support=[str(resolve(p, base['config'], root)) for p in cfg['extra_support']],
            output=str(output / (arm+'-s3')), processes=processes)
        stages.append(dict(name=f'{run_id}-{arm}-s3', kind='tree', arm=arm,
            config=cfg, command=arguments('round4_boost.py', cfg)))
        nc = dict(neural['config'], data=str(data), like=cfg['output']+'.json', output=str(output / (arm+'-NN2')))
        stages.append(dict(name=f'{run_id}-{arm}-NN2', kind='neural', arm=arm,
            config=nc, command=arguments('round8_nn_blend.py', nc)))
    return stages


def prepare(a):
    if a.blend_processes < 1 or Path(a.run_id).name != a.run_id:
        raise ValueError('invalid process count or run ID')
    plan_path, reference_path, output_plan = map(Path, (a.p2_plan, a.reference_audit, a.plan))
    if output_plan.exists():
        raise FileExistsError(output_plan)
    p2, reference = read(plan_path), read(reference_path)
    verify(p2['frozen'])
    verify(reference['frozen'])
    base_path, neural_path = Path(reference['base_report']), Path(reference['control_report'])
    base, neural = read(base_path), read(neural_path)
    if base['inputs'] != neural['inputs'] or Path(neural['config']['like']).name != base_path.name:
        raise ValueError('reference blends do not share a bank')
    if base['config']['no_union']:
        raise ValueError('requires the retained full union bank')
    for package, expected in (('lightgbm', base['lightgbm_version']), ('torch', neural['torch_version'])):
        if importlib.metadata.version(package) != expected:
            raise ValueError('blender library version differs: '+package)
    root, data = Path(a.reference_root).resolve(), Path(reference['data']).resolve()
    warm_report, warm_checkpoint = Path(a.warm_report).resolve(), Path(a.warm_checkpoint).resolve()
    warm = read(warm_report)
    if digest(data) != warm['data_sha256'] or digest(warm_checkpoint) != warm['checkpoint_sha256']:
        raise ValueError('full-training warm checkpoint/data differs from its report')
    selection = read(p2['warm_report'])
    if warm['selected_epoch'] != selection['selected_epoch']:
        raise ValueError('full and P2 warm checkpoints have different selected epochs')
    if {k: v for k, v in warm['config'].items() if k != 'device'} != {k: v for k, v in selection['config'].items() if k != 'device'}:
        raise ValueError('full and P2 warm configurations differ')
    if p2['score_epoch'] != p2['settings']['epochs']:
        raise ValueError('full pair must use the fixed final P2 epoch')
    frozen = dict(reference['frozen'])
    for path, sha in p2['frozen'].items():
        if path in frozen and frozen[path] != sha:
            raise ValueError('conflicting frozen input: '+path)
        frozen[path] = sha
    for path in (plan_path, reference_path, warm_report, warm_checkpoint):
        frozen[str(path.resolve())] = digest(path)
    for name in ('selection_objective_full.py', 'selection_objective_queue.py', 'experiment.py'):
        path = ROOT / 'scripts' / name
        frozen[str(path)] = digest(path)
    # Require each actual input to retain the reference report's hash.
    for name, sha in base['inputs'].items():
        path = resolve(name, base['config'], root)
        if digest(path) != sha:
            raise ValueError('retained input changed: '+name)
        frozen[str(path)] = sha
    d = load(data)
    reference_pred = neural_path.with_suffix('.avg.f32')
    values = np.fromfile(reference_pred, '<f4').astype(float)
    score = paired(d['probe']['rating'].astype(float), values, values, d['probe']['user'])['control_rmse']
    if abs(score-neural['avg_rmse']) > 1e-8:
        raise ValueError('reference score does not reproduce')
    output = Path(a.output).resolve()
    stages = make_stages(p2, base, neural, root, data, warm_report, warm_checkpoint,
                         output, a.run_id, a.blend_processes)
    plan = dict(state='prepared_gate_required', run_id=a.run_id, p2_plan=str(plan_path.resolve()),
        p2_plan_sha256=digest(plan_path), p2_gate=str(Path(a.p2_gate).resolve()),
        reference_audit=str(reference_path.resolve()), base_report=str(base_path), neural_report=str(neural_path),
        reference_prediction=str(reference_pred), reference_rmse=score, data=str(data), rows=len(d['probe']),
        output=str(output), stages=stages, frozen=frozen,
        n1_status=str(ROOT / '.claude/worktrees/honest-residual-chains/experiments/N1-chain.status.json'),
        p2_minimum_gain=p2['p2_minimum_gain'], full_bank_minimum_gain=p2['full_bank_minimum_gain'],
        comparisons=['treatment vs duplicate-control', 'treatment vs retained reference', 'duplicate-control vs retained reference'],
        condition='Recompute and verify passing P2 gate; N1 terminal and two idle Python snapshots before each stage.',
        reference_policy='This plan binds the supplied retained bank. Refresh it before execution if N1 or another completed experiment changes the promoted reference.',
        limitation='Full original-training development comparison only. No train+probe production, gate bypass, epoch search, or automatic promotion.')
    output_plan.write_text(json.dumps(plan, indent=2)+'\n')
    return plan


def verify_gate(plan):
    path = Path(plan['p2_gate'])
    gate = read(path)  # Missing evidence fails before any full-training stage.
    require_gate(gate, plan['p2_plan_sha256'], plan['p2_minimum_gain'])
    p2 = read(plan['p2_plan'])
    with tempfile.TemporaryDirectory(prefix='netflix-so1-gate-') as temp:
        output = Path(temp) / 'gate.json'
        command = list(p2['gate_command'])
        command[command.index('--output')+1] = str(output)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        recomputed = read(output)
    require_gate(recomputed, plan['p2_plan_sha256'], plan['p2_minimum_gain'])
    if gate['arms'] != recomputed['arms'] or gate['selected_penalties'] != recomputed['selected_penalties']:
        raise ValueError('P2 gate evidence differs from recomputation')
    if abs(gate['primary']['delta_rmse']-recomputed['primary']['delta_rmse']) > 1e-10:
        raise ValueError('P2 gate score differs from recomputation')
    bound = {str(path): digest(path)}
    for arm in p2['arms'].values():
        for suffix in ('.json', f".e{p2['score_epoch']}.f32"):
            name = arm['output']+suffix
            bound[name] = digest(name)
    return bound


def verify_training(stage, plan):
    cfg = stage['config']
    report_path = cfg['output']+'.json'
    report = read(report_path)
    if report['arguments'] != cfg or report['data_sha256'] != plan['frozen'][plan['data']]:
        raise ValueError('full training arguments/data differ from plan')
    for key in ('warm_checkpoint', 'warm_report'):
        if report[key+'_sha256'] != plan['frozen'][cfg[key]]:
            raise ValueError('warm input changed')
    p2 = read(plan['p2_plan'])
    if report['torch_version'] != p2['libraries']['torch'] or report['numpy_version'] != p2['libraries']['numpy']:
        raise ValueError('training library changed')
    for name, sha in report['sources'].items():
        if plan['frozen'][str(ROOT / 'scripts' / name)] != sha:
            raise ValueError('full training source changed')
    if [r['epoch'] for r in report['records']] != list(range(1, cfg['epochs']+1)):
        raise ValueError('incomplete full-training budget')
    bound = {report_path: digest(report_path)}
    for rec in report['records']:
        for suffix, key in (('f32', 'prediction_sha256'), ('pt', 'checkpoint_sha256')):
            path = cfg['output']+f".e{rec['epoch']}.{suffix}"
            if digest(path) != rec[key]:
                raise ValueError('full-training output hash differs')
            bound[path] = rec[key]
        values = np.fromfile(cfg['output']+f".e{rec['epoch']}.f32", '<f4')
        if values.shape != (plan['rows'],) or not np.isfinite(values).all():
            raise ValueError('invalid full-training predictions')
    return bound


def verify_blend(stage, plan):
    cfg = stage['config']
    path = cfg['output']+'.json'
    report = read(path)
    if report['config'] != cfg:
        raise ValueError('blend configuration differs')
    tree_cfg = cfg if stage['kind'] == 'tree' else read(cfg['like'])['config']
    root = Path(tree_cfg['bank_root'])
    expected = dict(read(root / 'experiments/UNION-round-manifest.json')['reports']['union']['inputs'])
    expected.update(read(root / 'experiments/BOOST-support-inputs.json'))
    expected.update({p: digest(p) for p in tree_cfg['extra']+tree_cfg['extra_support']})
    if expected != report['inputs']:
        raise ValueError('blend input bank differs')
    version_key, package = ('lightgbm_version', 'lightgbm') if stage['kind'] == 'tree' else ('torch_version', 'torch')
    reference = read(plan['base_report'] if stage['kind'] == 'tree' else plan['neural_report'])
    if report[version_key] != reference[version_key] or importlib.metadata.version(package) != reference[version_key]:
        raise ValueError('blend library differs')
    y = load(plan['data'])['probe']['rating'].astype(float)
    bound = {path: digest(path)}
    for suffix in (('.refit.f32', '.ridge.f32') if stage['kind'] == 'tree' else ('.nn.f32', '.avg.f32')):
        name = cfg['output']+suffix
        values = np.fromfile(name, '<f4')
        if values.shape != (plan['rows'],) or not np.isfinite(values).all():
            raise ValueError('invalid blend predictions')
        score = float(np.sqrt(np.mean((np.clip(values.astype(float), 1, 5)-y)**2)))
        recorded = report[suffix.split('.')[1]+'_rmse']
        if not math.isfinite(recorded) or abs(score-recorded) > 1e-7:
            raise ValueError('saved blend score does not reproduce')
        bound[name] = digest(name)
    return bound


def compare_full(y, users, reference, control, treatment, threshold):
    primary = paired(y, control, treatment, users)
    versus_reference = paired(y, reference, treatment, users)
    control_reference = paired(y, reference, control, users)
    return dict(primary=primary, treatment_vs_reference=versus_reference,
        control_vs_reference=control_reference,
        treatment_passes_reference_rule=versus_reference['all_folds_improve'],
        control_passes_reference_rule=control_reference['all_folds_improve'],
        auxiliary_passes_all_folds=primary['all_folds_improve'],
        substantial_auxiliary_investment=(primary['all_folds_improve'] and versus_reference['all_folds_improve']
            and primary['delta_rmse'] <= -threshold and versus_reference['delta_rmse'] <= -threshold),
        approximate_dev_target=.86465, target_reached_on_development=versus_reference['treatment_rmse'] <= .86465,
        metric='Reused original-probe development CV; calibrated development target is not a hidden-test victory.')


def execute(stages, check, wait, run_stage, validate):
    for stage in stages:
        check()
        wait(stage)
        check()
        wait(stage)  # Recheck after hashing large inputs.
        run_stage(stage)
        validate(stage)


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    os.chdir(ROOT)
    prefix = ROOT / 'experiments' / (plan['run_id']+'-chain')
    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if prefix.with_suffix('.status.json').exists() or Path(plan['output']).exists():
            raise FileExistsError('existing full-stage run; inspect it before recovery')
        verify(plan['frozen'])
        generated = verify_gate(plan)
        generated[str(plan_path)] = digest(plan_path)
        for stage in plan['stages']:
            if list((ROOT / 'experiments').glob(stage['name']+'.*')):
                raise FileExistsError('existing experiment: '+stage['name'])
        Path(plan['output']).mkdir(parents=True)
        completed = []
        def status(**fields):
            write(prefix.with_suffix('.status.json'), dict(pid=os.getpid(),
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=completed, **fields))
        def check():
            verify(plan['frozen']); verify(generated)
        def wait(stage):
            consecutive = 0
            while consecutive < 2:
                ready, evidence = inspect_idle(Path(plan['n1_status']), os.getpid())
                consecutive = consecutive+1 if ready else 0
                status(state='waiting', stage=stage['name'], idle_samples=consecutive, **evidence)
                if consecutive < 2:
                    time.sleep(30)
        def stage_run(stage):
            status(state='running', stage=stage['name'])
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', stage['name'],
                '--strategy', 'Conditional frozen SO1 full-training pair and matched full-bank comparison',
                '--', *stage['command']], check=True)
        def validate(stage):
            generated.update(verify_training(stage, plan) if stage['kind'] == 'training' else verify_blend(stage, plan))
            completed.append(stage['name'])
        try:
            execute(plan['stages'], check, wait, stage_run, validate)
            check()
            d = load(plan['data'])
            output = Path(plan['output'])
            predictions = {}
            for arm in ('duplicate-control', 'treatment'):
                avg = np.fromfile(output / (arm+'-NN2.avg.f32'), '<f4').astype(float)
                nn = np.fromfile(output / (arm+'-NN2.nn.f32'), '<f4').astype(float)
                tree = np.fromfile(output / (arm+'-s3.refit.f32'), '<f4').astype(float)
                if np.max(np.abs(avg-(nn+tree)/2)) > 1e-6:
                    raise ValueError('saved blend is not the fixed 50/50 average')
                predictions[arm] = avg
            result = compare_full(d['probe']['rating'].astype(float), d['probe']['user'],
                np.fromfile(plan['reference_prediction'], '<f4').astype(float),
                predictions['duplicate-control'], predictions['treatment'], plan['full_bank_minimum_gain'])
            result.update(plan_sha256=digest(plan_path), generated=generated)
            write(output / 'comparison.json', result)
            status(state='complete', comparison=str(output / 'comparison.json'))
        except BaseException as error:
            status(state='failed', error=repr(error))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    make = sub.add_parser('prepare')
    for name in ('p2-plan', 'p2-gate', 'reference-audit', 'reference-root', 'warm-report', 'warm-checkpoint', 'output', 'plan', 'run-id'):
        make.add_argument('--'+name, required=True)
    make.add_argument('--blend-processes', type=int, default=3)
    execute_parser = sub.add_parser('run')
    execute_parser.add_argument('--plan', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'prepare':
        prepared = prepare(args)
        print(json.dumps(dict(state=prepared['state'], stages=len(prepared['stages']),
            reference_rmse=prepared['reference_rmse'], frozen_inputs=len(prepared['frozen']))), flush=True)
    else:
        run(args.plan)
