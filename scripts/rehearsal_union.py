#!/usr/bin/env python3
"""Pair a holdout's training ratings with a probe of P2 rows followed by original probe rows.

For the Stage C rehearsal of models that cannot re-score a saved state on a
second dataset (the Bayes sampler's prediction is a posterior mean accumulated
during the chain): one chain on this dataset yields the P2 column (blender fit)
and the original-probe column (rehearsal a) from the same posterior. Training,
implicit arrays and header mean are byte-identical to the holdout, as in
rehearsal_view.py. `split` cuts the chain's predictions back into the two columns.
"""
import argparse
import json
import pathlib

import numpy as np

from gbdt_blend import HEADER, load
from train_holdout import digest


def create(holdout, original, output):
    holdout, original, output = map(pathlib.Path, (holdout, original, output))
    if output.exists() or pathlib.Path(str(output) + '.json').exists():
        raise FileExistsError('choose a new output path')
    h, o = load(holdout), load(original)
    for key in ('users', 'items', 'min_day', 'max_day'):
        if h[key] != o[key]:
            raise ValueError(key)
    for key in ('implicit_offsets', 'implicit_items', 'implicit_days'):
        if not np.array_equal(h[key], o[key]):
            raise ValueError(key)
    if h['probe'].dtype != o['probe'].dtype:
        raise ValueError('probe layouts differ')
    probe = np.concatenate([h['probe'], o['probe']])
    raw = np.memmap(holdout, mode='r', dtype=np.uint8)
    header = raw[:HEADER.itemsize].view(HEADER).copy()
    header[0]['probe_count'] = len(probe)
    temporary = pathlib.Path(str(output) + '.tmp')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open('xb') as stream:
            for part in (header, h['train_offsets'], h['train_items'], h['train_days'], h['train_ratings'],
                         probe, h['implicit_offsets'], h['implicit_items'], h['implicit_days']):
                np.asarray(part).tofile(stream)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    check = load(output)
    assert len(check['probe']) == len(probe) and check['mean'] == h['mean']
    assert np.array_equal(check['probe'][:len(h['probe'])], h['probe'])
    report = dict(holdout=str(holdout), holdout_sha256=digest(holdout), original=str(original),
                  original_sha256=digest(original), output=str(output), output_sha256=digest(output),
                  train_count=int(len(h['train_ratings'])), p2_rows=int(len(h['probe'])),
                  original_rows=int(len(o['probe'])))
    pathlib.Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def split(union, predictions, p2_out, probe_out):
    meta = json.loads(pathlib.Path(str(union) + '.json').read_text())
    pred = np.fromfile(predictions, dtype='<f4')
    n = meta['p2_rows']
    if len(pred) != n + meta['original_rows']:
        raise ValueError('prediction length does not match the union probe')
    for path, part in ((p2_out, pred[:n]), (probe_out, pred[n:])):
        if pathlib.Path(path).exists():
            raise FileExistsError(path)
        part.astype('<f4').tofile(path)
    return dict(p2=p2_out, probe=probe_out, p2_rows=n, original_rows=int(len(pred) - n))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('create')
    c.add_argument('--holdout', required=True)
    c.add_argument('--original', required=True)
    c.add_argument('--output', required=True)
    s = sub.add_parser('split')
    s.add_argument('--union', required=True)
    s.add_argument('--predictions', required=True)
    s.add_argument('--p2-output', required=True)
    s.add_argument('--probe-output', required=True)
    a = p.parse_args()
    if a.command == 'create':
        print(json.dumps(create(a.holdout, a.original, a.output), indent=2))
    else:
        print(json.dumps(split(a.union, a.predictions, a.p2_output, a.probe_output), indent=2))


if __name__ == '__main__':
    main()
