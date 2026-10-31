"""Audit completed E3 training, then run its two existing registered P2 gates."""
import argparse
import json
from pathlib import Path

import numpy as np

from e3_all_fold_gate import compare_all_folds
from gbdt_blend import load
from p2_candidate_gate import SOURCES, score
from selection_objective_queue import verify
from train_holdout import digest

SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def audit_report(plan, report_path, expected, config, sources):
    report = read(report_path)
    if report['arguments'] != expected or report['config'] != config:
        raise ValueError('training arguments or architecture differ')
    if report['sources'] != sources:
        raise ValueError('training sources differ from matched control')
    for name, sha in sources.items():
        if plan['frozen'][str(SCRIPTS/name)] != sha:
            raise ValueError('training source differs from frozen plan')
    for package in ('torch', 'numpy'):
        if report[package+'_version'] != plan['libraries'][package]:
            raise ValueError('training library differs')
    for key in ('data', 'warm_report', 'warm_checkpoint'):
        if report[key+'_sha256'] != plan['frozen'][expected[key]]:
            raise ValueError('training data or warm input differs')
    if expected['max_batches'] != 0 or expected['epochs'] != plan['score_epoch']:
        raise ValueError('smoke-only or unfixed training budget')
    if [r['epoch'] for r in report['records']] != list(range(1, plan['score_epoch']+1)):
        raise ValueError('incomplete fixed training budget')
    bound = {str(Path(report_path).resolve()): digest(report_path)}
    for record in report['records']:
        for suffix, key in [('f32', 'prediction_sha256'), ('pt', 'checkpoint_sha256')]:
            path = expected['output']+f".e{record['epoch']}.{suffix}"
            if digest(path) != record[key]:
                raise ValueError('saved training output differs: '+path)
            bound[path] = record[key]
    prediction = expected['output']+f".e{plan['score_epoch']}.f32"
    return dict(state='complete', smoke_only=False, experiment='E3',
        data_sha256=plan['frozen'][plan['data']], prediction=prediction,
        prediction_sha256=bound[prediction], report=str(Path(report_path).resolve()),
        report_sha256=bound[str(Path(report_path).resolve())]), bound


def run(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = read(plan_path)
    verify(plan['frozen'])
    parent = read(plan['parent_SO1_plan'])
    control = read(plan['control']['report'])
    expected_control = dict(parent['settings'], data=parent['data'],
        warm_checkpoint=parent['warm_checkpoint'], warm_report=parent['warm_report'],
        output=parent['arms']['control']['output'], weight=0.0)
    expected_treatment = plan['treatment']['expected_arguments']
    trainer = Path(plan['treatment']['command'][1])
    if expected_treatment['e3_script_sha256'] != digest(trainer):
        raise ValueError('E3 trainer hash differs')
    frozen = dict(plan['frozen'])
    frozen[str(plan_path)] = digest(plan_path)
    receipts = {}
    for arm, path, expected in [('control', plan['control']['report'], expected_control),
            ('treatment', plan['treatment']['output']+'.json', expected_treatment)]:
        receipt, bound = audit_report(plan, path, expected, control['config'], control['sources'])
        receipt['arm'] = arm
        receipt['training_plan_sha256'] = digest(plan_path)
        frozen.update(bound)
        receipts[arm] = receipt
    for name in (*SOURCES, 'e3_all_fold_gate.py', 'e3_candidate_gate.py'):
        path = SCRIPTS/name
        frozen[str(path)] = digest(path)
    verify(plan['frozen'])
    output.mkdir(parents=True)
    arms = {}
    for arm, receipt in receipts.items():
        path = output/(arm+'-receipt.json')
        write(path, receipt)
        frozen[str(path)] = digest(path)
        arms[arm] = dict(receipt=str(path), prediction=receipt['prediction'])
    manifest = dict(experiment='E3', protocol='SO1-P2-fold0-matched-pair-v1',
        minimum_gain=plan['primary_gate']['minimum_gain'], protocol_reference=str(plan_path),
        data=plan['data'], bank=plan['bank'], frozen=frozen, arms=arms)
    manifest_path = output/'shared-plan.json'
    write(manifest_path, manifest)
    primary, _ = score(manifest_path, output/'fold0.json')
    d = load(plan['data'])
    bank = np.column_stack([np.fromfile(p, '<f4').astype(float) for p in plan['bank']])
    control_values, treatment_values = [np.fromfile(arms[a]['prediction'], '<f4').astype(float)
                                      for a in ('control', 'treatment')]
    all_folds, predictions = compare_all_folds(bank, control_values, treatment_values,
        d['probe']['rating'].astype(float), d['probe']['user'], plan['additional_gate']['minimum_gain'])
    verify(frozen)
    if digest(plan_path) != frozen[str(plan_path)] or digest(manifest_path) != primary['plan_sha256']:
        raise ValueError('scoring plan changed')
    saved = {}
    for name, values in predictions.items():
        path = output/(name+'.npy')
        np.save(path, values, allow_pickle=False)
        saved[str(path)] = digest(path)
    result = dict(state='complete', experiment='E3', training_plan_sha256=digest(plan_path),
        shared_plan_sha256=digest(manifest_path), primary=primary, all_folds=all_folds,
        saved_all_fold_predictions=saved,
        passes_both_P2_gates=primary['passes_p2_gate'] and all_folds['passes_all_fold_gate'],
        scope='Both fixed P2 gates required; a pass permits the registered full-training follow-up, never automatic promotion.')
    write(output/'comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-dir', required=True)
    a = parser.parse_args()
    result = run(a.plan, a.output_dir)
    print(json.dumps(dict(state=result['state'], passes_both_P2_gates=result['passes_both_P2_gates'])), flush=True)
