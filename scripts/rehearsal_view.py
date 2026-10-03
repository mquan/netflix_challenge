#!/usr/bin/env python3
"""Pair a training-only holdout's training ratings with the original probe.

For the Stage C rehearsal: models are trained and epoch-selected on a holdout
dataset (its probe is P2, a set of original training ratings). This writes a
second dataset with byte-identical header mean, training and implicit arrays,
whose probe is the original probe, so `eval-*` commands can score the same
models on the original probe. Header mean and training arrays come from the
holdout, so a model's derived state (means, sessions, frequencies) is unchanged.
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
    raw = np.memmap(holdout, mode='r', dtype=np.uint8)
    header = raw[:HEADER.itemsize].view(HEADER).copy()
    header[0]['probe_count'] = len(o['probe'])
    temporary = pathlib.Path(str(output) + '.tmp')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open('xb') as stream:
            for part in (header, h['train_offsets'], h['train_items'], h['train_days'], h['train_ratings'],
                         o['probe'], h['implicit_offsets'], h['implicit_items'], h['implicit_days']):
                np.asarray(part).tofile(stream)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    check = load(output)
    assert len(check['probe']) == len(o['probe']) and check['mean'] == h['mean']
    report = dict(holdout=str(holdout), holdout_sha256=digest(holdout), original=str(original),
                  original_sha256=digest(original), output=str(output), output_sha256=digest(output),
                  train_count=int(len(h['train_ratings'])), probe_count=int(len(o['probe'])))
    pathlib.Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--holdout', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(create(a.holdout, a.original, a.output), indent=2))


if __name__ == '__main__':
    main()
