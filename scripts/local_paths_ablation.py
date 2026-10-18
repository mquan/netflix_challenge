#!/usr/bin/env python3
"""Attribute a completed P2 pilot's gain without changing retrieval or penalties."""
import argparse
import json
import pathlib

import numpy as np

from gbdt_blend import load
from local_paths_pilot import signal_test
from train_holdout import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run, output = pathlib.Path(a.run), pathlib.Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((run / 'report.json').read_text())
    data = report['arguments']['data']
    if digest(data) != report['data_sha256'] or digest(run / 'sample.npz') != report['sample_sha256']:
        raise ValueError('pilot input hash mismatch')
    sample = np.load(run / 'sample.npz')
    y = load(data)['probe']['rating'][sample['queries']].astype(float)
    control, paths = sample['control'], sample['paths']
    comparisons = {}
    for name, base, extra in [
        ('agreement_only', control, paths[:, 125:]),
        ('histogram_only', control, paths[:, :125]),
        ('histogram_beyond_agreement', np.column_stack([control, paths[:, 125:]]), paths[:, :125]),
    ]:
        comparisons[name], _ = signal_test(base, extra, y, sample['users'], sample['knn'])
    result = dict(run=str(run), pilot_report_sha256=digest(run / 'report.json'),
                  source_sha256={str(path): digest(path) for path in
                                 (pathlib.Path(__file__), pathlib.Path(__file__).with_name('local_paths_pilot.py'))},
                  note='Development attribution on the same P2 evaluation users; no independent-test claim.',
                  comparisons=comparisons)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
