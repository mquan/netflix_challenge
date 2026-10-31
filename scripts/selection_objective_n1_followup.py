"""Fixed N1 refit/blend prelude; refresh SO1's bank only on five-fold success."""
import argparse
import json
from pathlib import Path

import numpy as np

import selection_objective_full as full
from selection_objective_queue import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def execute(plan, run_stage, wait_idle, check):
    verify(plan['frozen'])
    gate = json.loads(Path(plan['gate']).read_text())
    if gate['candidate_beyond_A2h']['delta_rmse'] > -0.001:
        raise ValueError('N1 gate no longer passes')
    for stage in plan['stages']:
        check()
        wait_idle(stage['name'])
        check()
        run_stage(stage['name'], stage['command'])
        if stage['kind'] == 'refit':
            report = json.loads(Path(plan['refit']+'.json').read_text())
            if (report['selected_epoch'] != 13 or len(report['refit']) != 13
                    or report['arguments'] != plan['refit_arguments']
                    or report['data_sha256'] != plan['frozen'][plan['data']]
                    or report['predictions_sha256'] != digest(plan['refit']+'.f32')
                    or report['source_sha256'] != plan['trainer_sha256']):
                raise ValueError('N1 refit provenance or fixed budget differs')
    tree_path, neural_path = Path(plan['tree_report']), Path(plan['neural_report'])
    tree, neural = json.loads(tree_path.read_text()), json.loads(neural_path.read_text())
    if tree['inputs'] != neural['inputs']:
        raise ValueError('N1 blender banks differ')
    avg = np.fromfile(neural_path.with_suffix('.avg.f32'), '<f4').astype(float)
    nn = np.fromfile(neural_path.with_suffix('.nn.f32'), '<f4').astype(float)
    boost = np.fromfile(tree_path.with_suffix('.refit.f32'), '<f4').astype(float)
    d = full.load(plan['data'])
    if len(avg) != len(d['probe']) or not np.isfinite(avg).all():
        raise ValueError('invalid N1 predictions')
    if np.max(np.abs(avg-(nn+boost)/2)) > 1e-6:
        raise ValueError('N1 blend is not the fixed 50/50 average')
    old_full = json.loads((ROOT/'experiments/SO1-full-plan.json').read_text())
    reference = np.fromfile(old_full['reference_prediction'], '<f4').astype(float)
    paired = full.paired(d['probe']['rating'].astype(float), reference, avg, d['probe']['user'])
    if abs(paired['treatment_rmse']-neural['avg_rmse']) > 1e-8:
        raise ValueError('N1 score does not reproduce')
    eligible = paired['all_folds_improve'] and paired['delta_rmse'] < 0
    result = dict(comparison=paired, passes_promotion_rule=eligible,
                  substantial_continuation=eligible and paired['delta_rmse'] <= -0.0003,
                  metric='Development evidence; no hidden-test claim')
    result_path = ROOT/'experiments/N1-resume-20261030-comparison.json'
    with result_path.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    if not eligible:
        return None
    # Preserve the old plan; bind a new SO1 comparison to the successful bank.
    audit = json.loads((ROOT/'experiments/LC6-plan.json').read_text())
    audit.update(base_report=str(tree_path), control_report=str(neural_path))
    for path in [tree_path, neural_path, result_path,
                 tree_path.with_suffix('.refit.f32'), neural_path.with_suffix('.avg.f32'),
                 neural_path.with_suffix('.nn.f32'), Path(plan['refit']+'.f32')]:
        audit['frozen'][str(path)] = digest(path)
    audit_path = ROOT/'experiments/SO1-post-N1-reference-20261030.json'
    with audit_path.open('x') as stream:
        stream.write(json.dumps(audit, indent=2)+'\n')
    new_path = ROOT/'experiments/SO1-post-N1-full-plan-20261030.json'
    full.prepare(argparse.Namespace(p2_plan=old_full['p2_plan'], p2_gate=old_full['p2_gate'],
        reference_audit=str(audit_path), reference_root=str(ROOT),
        warm_report=old_full['stages'][0]['config']['warm_report'],
        warm_checkpoint=old_full['stages'][0]['config']['warm_checkpoint'],
        output=str(ROOT/'artifacts/selection-objective/SO1-post-N1-full-20261030'),
        plan=str(new_path), run_id='SO1-post-N1-full-20261030', blend_processes=3))
    return new_path
