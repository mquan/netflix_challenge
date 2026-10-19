#!/usr/bin/env python3
"""Fixed P2 ensemble gate for a completed, matched selection-objective pair."""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from train_holdout import digest


def ridge_predict(features, y, folds):
    x = np.column_stack([np.ones(len(y)), features])
    fit, select = folds >= 2, folds == 1
    if not all(np.any(folds == f) for f in range(5)):
        raise ValueError('all five user folds must be present')
    gram, rhs = x[fit].T @ x[fit], x[fit].T @ y[fit]
    choices = []
    for penalty in (1, 10, 100, 1000):
        w = np.linalg.solve(gram + penalty * np.eye(x.shape[1]), rhs)
        choices.append((float(np.mean((np.clip(x[select] @ w, 1, 5) - y[select]) ** 2)), penalty))
    penalty = min(choices)[1]
    fit = folds >= 1
    w = np.linalg.solve(x[fit].T @ x[fit] + penalty * np.eye(x.shape[1]), x[fit].T @ y[fit])
    return x[folds == 0] @ w, penalty


def compare(bank, control, treatment, y, users):
    if bank.ndim != 2 or any(v.shape != y.shape for v in (control, treatment, users)) or len(bank) != len(y):
        raise ValueError('prediction shapes differ')
    if not all(np.isfinite(v).all() for v in (bank, control, treatment, y)):
        raise ValueError('nonfinite input')
    folds = blend_fold(users)
    designs = {
        'duplicate_control': np.column_stack([bank, control, control]),
        'control_plus_treatment': np.column_stack([bank, control, treatment]),
        'control_only': np.column_stack([bank, control]),
        'treatment_only': np.column_stack([bank, treatment]),
    }
    predictions, penalties = {}, {}
    for name, x in designs.items():
        predictions[name], penalties[name] = ridge_predict(x, y, folds)
    held = folds == 0
    primary = paired(y[held], predictions['duplicate_control'], predictions['control_plus_treatment'], users[held])
    replacement = paired(y[held], predictions['control_only'], predictions['treatment_only'], users[held])
    return dict(primary=primary, replacement_diagnostic=replacement, selected_penalties=penalties), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(args.plan).read_text())
    for path, sha in plan['frozen'].items():
        if digest(path) != sha:
            raise ValueError('frozen input changed: ' + path)
    data = load(plan['data'])
    y, users = data['probe']['rating'].astype(float), data['probe']['user']
    predictions, evidence = {}, {}
    for name, arm in plan['arms'].items():
        prefix = arm['output']
        record = json.loads(Path(prefix + '.json').read_text())
        if record['torch_version'] != plan['libraries']['torch'] or record['numpy_version'] != plan['libraries']['numpy']:
            raise ValueError('training library version differs')
        for key, expected in dict(plan['settings'], weight=arm['weight'], output=prefix,
                                  data=plan['data'], warm_checkpoint=plan['warm_checkpoint'],
                                  warm_report=plan['warm_report']).items():
            if record['arguments'][key] != expected:
                raise ValueError(f'{name} configuration differs: {key}')
        if record['warm_checkpoint_sha256'] != plan['frozen'][plan['warm_checkpoint']]:
            raise ValueError('warm checkpoint differs')
        if record['warm_report_sha256'] != plan['frozen'][plan['warm_report']]:
            raise ValueError('warm training report differs')
        if record['data_sha256'] != plan['frozen'][plan['data']]:
            raise ValueError('training data differs')
        for filename, sha in record['sources'].items():
            if sha != plan['frozen'][str(Path(__file__).resolve().with_name(filename))]:
                raise ValueError('training source differs')
        if [r['epoch'] for r in record['records']] != list(range(1, plan['settings']['epochs'] + 1)):
            raise ValueError('incomplete fixed-budget training')
        path = prefix + f".e{plan['settings']['epochs']}.f32"
        expected = record['records'][-1]['prediction_sha256']
        if digest(path) != expected:
            raise ValueError('prediction hash differs')
        predictions[name] = np.fromfile(path, dtype='<f4').astype(float)
        evidence[name] = dict(report_sha256=digest(prefix + '.json'), prediction_sha256=expected)
    bank = np.column_stack([np.fromfile(path, dtype='<f4').astype(float) for path in plan['bank']])
    result, _ = compare(bank, predictions['control'], predictions['treatment'], y, users)
    result.update(plan_sha256=digest(args.plan), arms=evidence,
                  passes_full_refit_gate=result['primary']['delta_rmse'] <= -plan['p2_minimum_gain'],
                  metric='Reused P2 development gate. Epoch 2 fixed in advance; no untouched-test or full-bank score claim.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
