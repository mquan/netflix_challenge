#!/usr/bin/env python3
"""Verify documented retained kNN variants by replay; never replace frozen outputs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from local_context_chain import write
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/production-replay/PR1-knn'
BASE = Path('/Users/quan/workspace/netflix_round9/artifacts/r9/R9-LAT-A2h-20261010-s3.json')
# Filename/document-derived candidate recipes. A mismatch is recorded as
# unresolved, not searched over or accepted on a merely similar RMSE.
RECIPES = [
    ('knn-J6-p2', 'neighbors64', 'joint256-reg01', 64, 2, .5, 0),
    ('knn-J6-k16', 'neighbors64', 'joint256-reg01', 16, 1, .5, 0),
    ('knn-J6-s5', 'neighbors64', 'joint256-reg01', 64, 1, 5, 0),
    ('knn-J6-td.003', 'neighbors64', 'joint256-reg01', 64, 1, .5, .003),
    ('knn-J6-td.1', 'neighbors64', 'joint256-reg01', 64, 1, .5, .1),
    ('knn256-J6-k256-p1-s.5', 'r4/neighbors256', 'joint256-reg01', 256, 1, .5, 0),
    ('knn256-J6-k128-p1-s.5', 'r4/neighbors256', 'joint256-reg01', 128, 1, .5, 0),
    ('knn256-J6-k256-p2-s.5', 'r4/neighbors256', 'joint256-reg01', 256, 2, .5, 0),
    ('knn-T6R-p1', 'neighbors64', 'T6R-f256', 64, 1, .5, 0),
    ('knn256td-T6R', 'r4/neighbors256', 'T6R-f256', 256, 1, .5, .003),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round4-decay', action='store_true',
                        help='Replay the three unresolved J6 graph-256 variants with ROUND4.md documented decay .003')
    args = parser.parse_args()
    run = 'PR2-knn-decay' if args.round4_decay else 'PR1-knn'
    destination = ROOT / 'artifacts/production-replay' / run
    recipes = RECIPES
    if args.round4_decay:
        recipes = [(*row[:-1], .003) for row in RECIPES if row[0].startswith('knn256-J6-')]
    os.chdir(ROOT)
    base = json.loads(BASE.read_text())
    plans = []
    for name, graph, model, neighbors, power, shrink, decay in recipes:
        target = f'artifacts/r4/{name}.f32'
        support = f'artifacts/r4/{name}.support.f32'
        expected = base['inputs'][target]
        assert digest(target) == expected
        prediction, support_out = destination / (name + '.f32'), destination / (name + '.support.f32')
        cmd = ['build/netflix', 'knn', '--data', 'data/processed/netflix.bin',
               '--graph', f'artifacts/{graph}.bin', '--joint-model', f'artifacts/{model}.bin',
               '--neighbors', str(neighbors), '--power', str(power), '--shrink', str(shrink),
               '--time-decay', str(decay), '--predictions', str(prediction), '--support', str(support_out)]
        plans.append(dict(name=name, target=target, expected_sha256=expected,
                          support_target=support if support in base['inputs'] else None,
                          support_expected_sha256=base['inputs'].get(support), command=cmd))
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(base_report=str(BASE), base_report_sha256=digest(BASE), plans=plans, results=[],
                  limitation='Replays existing base models and graphs; their train+probe rebuilds remain separate requirements.')
    if args.round4_decay:
        report['recipe_evidence'] = dict(path='ROUND4.md', sha256=digest(ROOT / 'ROUND4.md'),
            detail='Base-model work table: G2 top-256 graph, k=256, lambda=.003 on J6; apply that documented decay to the three unresolved variants.')
    write(ROOT / f'experiments/{run}-plan.json', report)
    status_path = ROOT / f'experiments/{run}.status.json'
    try:
        for number, plan in enumerate(plans):
            write(status_path, dict(state='running', pid=os.getpid(), stage=plan['name'], completed=len(report['results'])))
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', f'{run}-{number:02d}',
                            '--strategy', 'Frozen-column recipe recovery; byte equality required, no tuning or replacement',
                            '--', *plan['command']], check=True)
            pred_path = Path(plan['command'][plan['command'].index('--predictions') + 1])
            actual = digest(pred_path)
            result = dict(name=plan['name'], prediction_sha256=actual,
                          prediction_byte_identical=actual == plan['expected_sha256'])
            if not result['prediction_byte_identical']:
                old = np.fromfile(plan['target'], dtype='<f4')
                new = np.fromfile(pred_path, dtype='<f4')
                result['max_absolute_difference'] = float(np.max(np.abs(old - new)))
            if plan['support_target']:
                path = Path(plan['command'][plan['command'].index('--support') + 1])
                actual_support = digest(path)
                assert digest(plan['support_target']) == plan['support_expected_sha256']
                result.update(support_sha256=actual_support,
                              support_byte_identical=actual_support == plan['support_expected_sha256'])
            report['results'].append(result)
            print(json.dumps(result), flush=True)
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(ROOT / f'experiments/{run}-verification.json', report)
        write(status_path, dict(state='complete', pid=os.getpid(), completed=len(report['results']),
              exact_predictions=sum(row['prediction_byte_identical'] for row in report['results'])))
    except BaseException as error:
        write(status_path, dict(state='failed', pid=os.getpid(), completed=len(report['results']), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
