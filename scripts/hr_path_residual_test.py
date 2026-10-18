#!/usr/bin/env python3
"""Do local-path features add beyond honest strong-model predictions on P2?

Reads a completed local-path pilot run (read-only) and adds honest P2 columns,
i.e. predictions from models trained without P2 ratings, to its 27-feature
control. Retrieval, features, user folds and penalty grid are the pilot's own
(`local_paths_pilot.signal_test`). Original probe labels are never read.
"""
import argparse
import json
import pathlib

import numpy as np

from gbdt_blend import blend_fold, load
from local_paths_pilot import rmse, signal_test
from train_holdout import digest

REHEARSAL = pathlib.Path('/Users/quan/workspace/netflix_challenge/artifacts/rehearsal')
A2H = REHEARSAL / 'r12/A-A2h-select.p2.f32'
BANK = [REHEARSAL / name for name in (
    'A-MH3-categorical-mse.p2.f32', 'A-MT0-select.p2.f32', 'A-MT1-select.p2.f32',
    'A-MT2-select.p2.f32', 'A-MT3-select.p2.f32', 'A-K1.p2.f32',
    'A-knn256td.train-matched-20260927.f32', 'A-knn64.train-matched-20260927.f32',
    'A-rbm100c.p2.f32', 'r12/A-B4-union.p2.f32', 'r12/A-A2h-select.p2.f32')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run, output = pathlib.Path(a.run), pathlib.Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((run / 'report.json').read_text())
    data = REHEARSAL.parent.parent / report['arguments']['data']  # relative to the primary checkout
    if digest(data) != report['data_sha256'] or digest(run / 'sample.npz') != report['sample_sha256']:
        raise ValueError('pilot input hash mismatch')
    sample = np.load(run / 'sample.npz')
    q, users = sample['queries'], sample['users']
    y = load(data)['probe']['rating'][q].astype(float)
    control, paths, knn = sample['control'], sample['paths'], sample['knn']
    a2h = np.fromfile(A2H, dtype='<f4').astype(float)[q]
    bank = np.column_stack([np.fromfile(c, dtype='<f4').astype(float)[q] for c in BANK])
    held = blend_fold(users) == 0
    comparisons = {}
    for name, base in [('pilot_control', control),
                       ('control_plus_A2h', np.column_stack([control, a2h])),
                       ('control_plus_honest_bank11', np.column_stack([control, bank]))]:
        comparisons[name], _ = signal_test(base, paths, y, users, knn)
    result = dict(run=str(run), pilot_report_sha256=digest(run / 'report.json'),
                  inputs={str(c): digest(c) for c in BANK},
                  source_sha256=digest(pathlib.Path(__file__)),
                  a2h_alone_evaluation_rmse=rmse(y[held], np.clip(a2h[held], 1, 5)),
                  note='P2 development test; all added columns come from models trained without P2. '
                       'Linear ridge only; a gain here is necessary, not sufficient, for ensemble value.',
                  comparisons=comparisons)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    for name, c in comparisons.items():
        m = c['models']
        print(f"{name:28s} base {m['pooled_knn']['evaluation_rmse']:.6f}  +paths {m['pooled_knn_paths']['evaluation_rmse']:.6f}"
              f"  delta {c['delta_rmse']:+.6f}  95% MSE CI {c['cluster_normal_95_mse_interval']}")
    print('A2h alone', result['a2h_alone_evaluation_rmse'])


if __name__ == '__main__':
    main()
