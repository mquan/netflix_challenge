#!/usr/bin/env python3
"""Compare frozen B6 columns with their union with the moonshot predictors.

Requires hash-verified snapshots under artifacts/other-branch. The fixed equal
blend fits no additional parameters to already cross-fitted model outputs.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
from gbdt_blend import load, blend_fold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--mode', choices=['average', 'other', 'union'], required=True)
    a = p.parse_args()
    source = json.loads(pathlib.Path('experiments/UNION-b6-source.json').read_text())
    own = json.loads(pathlib.Path('experiments/NN-round-manifest.json').read_text())
    columns = []
    for name, record in source['inputs'].items():
        if not name.endswith('.f32'):
            continue
        path = pathlib.Path('artifacts/other-branch') / pathlib.Path(name).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
        columns.append(str(path))
    if a.mode == 'average':
        names = ['artifacts/other-branch/blend-b6.f32', 'artifacts/NN2-blend.f32']
        expected = [source['predictions_artifact']['sha256'], own['outputs'][names[1]]['sha256']]
        raw = [pathlib.Path(name).read_bytes() for name in names]
        for value, sha in zip(raw, expected):
            assert hashlib.sha256(value).hexdigest() == sha
        d = load(a.data); y = d['probe']['rating'].astype(np.float64)
        predictions = [np.frombuffer(value, dtype='<f4').astype(np.float64) for value in raw]
        assert all(len(col) == len(y) and np.isfinite(col).all() for col in predictions)
        # A predeclared 50/50 average, with clipping applied after combining.
        average = (predictions[0] + predictions[1]) / 2
        average.astype('<f4').tofile('artifacts/UNION-equal.f32')
        folds = blend_fold(d['probe']['user'])
        scores = []
        for name, col in zip(names + ['artifacts/UNION-equal.f32'], predictions + [average]):
            error = (np.clip(col, 1, 5) - y)**2
            scores.append(dict(name=name, rmse=float(np.sqrt(error.mean())),
                               fold_rmse=[float(np.sqrt(error[folds == f].mean())) for f in range(5)]))
        report = dict(weights=[.5, .5], inputs=dict(zip(names, expected)), scores=scores,
                      protocol='Fixed equal average; no additional fitting or weight search. Development CV.')
        pathlib.Path('experiments/UNION-equal.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        return
    if a.mode == 'union':
        # Six additional columns: three GP, one temporal GP, two residual factors.
        for name, sha in own['nn2_report']['inputs'].items():
            if name.startswith('artifacts/snapshot/'):
                continue
            assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == sha
            columns.append(name)
    output = 'UNION-' + a.mode
    command = [sys.executable, '-u', 'scripts/moonshot_nn_blend.py', '--data', a.data,
               '--output', 'artifacts/'+output, '--hidden', '16', '--epochs', '60']
    for name in columns:
        command += ['--prediction', name]
    with open('experiments/'+output+'.log', 'w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
