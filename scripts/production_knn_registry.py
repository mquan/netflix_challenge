#!/usr/bin/env python3
"""Consolidate exact item-kNN recovery evidence for every retained bank column."""
import json
from pathlib import Path

from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'experiments/production-knn-recipes.json'
    if output.exists():
        raise FileExistsError(output)
    inventory_path = ROOT / 'experiments/production-bank-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    base_path = Path(inventory['base_report'])
    if digest(base_path) != inventory['base_report_sha256']:
        raise ValueError('retained bank report changed')
    base = json.loads(base_path.read_text())
    checked = {}

    def check(path, expected):
        path = (ROOT / path).resolve()
        if path not in checked:
            checked[path] = digest(path)
        if checked[path] != expected:
            raise ValueError('artifact changed: ' + str(path))
        return str(path)

    candidates = [('PR0-J6-knn64', '9d80c2ecc225c70d3cac88fd94b7b54ed08768eb21ba76923ad1a49092ee2a6a',
                   'experiments/PR0-J6-knn64-verification.json'),
                  ('PR10-knn-J1-p2-shrink1', 'e297b67618c63e08d08a92fe705d33cff5516050833c71b676de1d9a8ed1a641',
                   'experiments/PR10-knn-shrink-recovery-verification.json')]
    sources = {}
    for run in ('PR1-knn', 'PR2-knn-decay', 'PR9-knn-remaining'):
        path = ROOT / 'experiments' / (run + '-verification.json')
        report = json.loads(path.read_text())
        sources[str(path)] = digest(path)
        plans = {p['name']: (i, p) for i, p in enumerate(report['plans'])}
        for result in report['results']:
            if not result['prediction_byte_identical']:
                continue
            index, plan = plans[result['name']]
            if result['prediction_sha256'] != plan['expected_sha256']:
                raise ValueError('inconsistent recovery report')
            check(plan['target'], plan['expected_sha256'])
            candidates.append((f'{run}-{index:02d}', plan['expected_sha256'], str(path)))

    recipes, supports = {}, {}
    for experiment, expected, verification in candidates:
        path = ROOT / 'experiments' / (experiment + '.json')
        record = json.loads(path.read_text())
        if record['exit_code'] != 0 or record['predictions_artifact']['sha256'] != expected:
            raise ValueError('invalid recovery execution: ' + experiment)
        prediction = check(record['predictions_artifact']['path'], expected)
        binary = check(record['command'][0], record['binary_sha256'])
        for p, info in record['inputs'].items():
            check(p, info['sha256'])
        sources[str(path)] = digest(path)
        verification_path = ROOT / verification
        sources[str(verification_path)] = digest(verification_path)
        entry = dict(execution_record=str(path), execution_record_sha256=sources[str(path)],
                     verification=str(verification_path), prediction_sha256=expected,
                     replay=prediction, binary=binary, binary_sha256=record['binary_sha256'],
                     command=record['command'], inputs=record['inputs'])
        recipes[expected] = entry
        command = record['command']
        if '--support' in command:
            support_path = (ROOT / command[command.index('--support') + 1]).resolve()
            support_sha = digest(support_path)
            supports[support_sha] = dict(replay=str(support_path), sha256=support_sha,
                                        execution_record=str(path), command=command)

    columns, missing = [], []
    for column in inventory['columns']:
        if not Path(column['path']).name.startswith('knn'):
            continue
        check(column['path'], column['sha256'])
        if column['sha256'] not in recipes:
            missing.append(column['path'])
        else:
            columns.append(dict(retained=column['path'], **recipes[column['sha256']]))
    support_columns = []
    for path, expected in base['inputs'].items():
        name = Path(path).name
        if not (name.startswith('knn') and name.endswith('.support.f32')):
            continue
        if expected not in supports:
            missing.append(path)
        else:
            support_columns.append(dict(retained_input=path, expected_sha256=expected, **supports[expected]))
    if missing:
        raise ValueError('unresolved retained columns: ' + ', '.join(missing))
    result = dict(inventory=str(inventory_path), inventory_sha256=digest(inventory_path),
                  base_report=str(base_path), base_report_sha256=digest(base_path),
                  columns=columns, support_columns=support_columns,
                  retained_column_count=len(columns), unique_prediction_count=len({c['prediction_sha256'] for c in columns}),
                  retained_support_count=len(support_columns), unique_support_count=len(supports.keys() & {s['expected_sha256'] for s in support_columns}),
                  evidence=sources,
                  scope='All retained item-kNN inference recipes verified against frozen digests. Includes duplicate bank aliases. User-kNN is a separate family. Production runs must substitute freshly trained production models and graphs and the production dataset; these are original-data inference replays, not production training or score improvements.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('retained_column_count', 'unique_prediction_count', 'retained_support_count', 'unique_support_count')}))


if __name__ == '__main__':
    main()
