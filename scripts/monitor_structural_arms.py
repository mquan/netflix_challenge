"""Audit newly completed fixed S0 arms; never train, select or retune anything."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from gbdt_blend import load
from production_rbm_audit import digest, verify, write_new
from structural_scoping_report import audit_arm


def check(plan_path):
    plan_path = Path(plan_path).resolve()
    plan_sha = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    # Small control files every pass; large immutable banks only for a new audit.
    verify(plan['control_bindings'])
    execution_path, binding_path = Path(plan['execution']), Path(plan['reporter_binding'])
    execution = json.loads(execution_path.read_text())
    binding = json.loads(binding_path.read_text())
    arms = {arm['name']: arm for arm in execution['arms'] if arm['kind'] != 'reference'}
    if len(arms) != 25 or set(arms) != set(plan['receipts']):
        raise ValueError('Complete fixed S0 arm coverage required')
    complete, created = 0, []
    labels = None
    processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True).splitlines()
    for name, destination in plan['receipts'].items():
        destination = Path(destination)
        if destination.exists():
            receipt = json.loads(destination.read_text())
            if (receipt['state'] != 'verified_complete_arm' or receipt['arm'] != name or
                    receipt['execution_sha256'] != plan['control_bindings'][str(execution_path)] or
                    receipt['reporter_binding_sha256'] != plan['control_bindings'][str(binding_path)]):
                raise ValueError('Existing arm receipt differs: ' + name)
            complete += 1
            continue
        arm = arms[name]
        if any('--output ' + stage['config']['output'] + ' ' in line or
               line.endswith('--output ' + stage['config']['output'])
               for stage in arm['stages'] for line in processes):
            continue
        suffixes = [('.json', '.refit.f32', '.ridge.f32'), ('.json', '.nn.f32', '.avg.f32')]
        if any(not Path(stage['config']['output'] + suffix).is_file()
               for stage, group in zip(arm['stages'], suffixes) for suffix in group):
            continue
        verify(execution['frozen'])
        verify(binding['frozen'])
        average, tree, neural, outputs = audit_arm(arm, dict(execution,
            base_report=execution['reference_tree'], neural_report=execution['reference_neural']))
        if labels is None:
            labels = load(execution['data'])['probe']['rating'].astype(float)
        if average.shape != labels.shape:
            raise ValueError('S0 average row count differs')
        error = float(np.max(np.abs(average - (tree + neural) / 2)))
        if error > 1e-6:
            raise ValueError('Saved fixed 50/50 average differs')
        rmse = lambda values: float(np.sqrt(np.mean((np.clip(values, 1, 5) - labels) ** 2)))
        record = dict(state='verified_complete_arm', arm=name,
            execution_sha256=digest(execution_path), reporter_binding_sha256=digest(binding_path),
            config_bank_library_and_saved_scores_verified=True, all_five_tree_and_neural_folds_complete=True,
            saved_average_max_component_error=error, tree_rmse=rmse(tree), neural_rmse=rmse(neural),
            average_rmse=rmse(average), outputs=outputs,
            scope='Completed fixed nonreference arm. All 25 arms required for synthesis; no selection, '
                  'retuning, promotion, training intervention or forecast. Final reporter remains separately managed.')
        verify(execution['frozen'])
        verify(binding['frozen'])
        verify(outputs)
        verify(plan['control_bindings'])
        if digest(plan_path) != plan_sha:
            raise ValueError('S0 monitor plan changed')
        write_new(destination, record)
        created.append(name)
        complete += 1
        print(json.dumps(dict(state='new_fixed_arm_audited', arm=name, average_rmse=record['average_rmse'])), flush=True)
    print(json.dumps(dict(state='all_25_arm_receipts_present' if complete == 25 else 'waiting_for_fixed_arms',
                          complete=complete, total=25, new_receipts=created)), flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    raise SystemExit(10 if check(args.plan) == 25 else 0)
