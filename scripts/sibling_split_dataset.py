#!/usr/bin/env python3
"""Build the sibling-split Stage C rehearsal dataset (HEADROOM.md, A7).

Splits the original probe into half A and half B with the same rule as
scripts/sibling_pooling.py (seed 20261031: row in A if uniform < .5, in probe
order). Writes a dataset whose training ratings are T plus the A rows (each
user's A rows appended to the end of that user's training segment) and whose
probe is the B rows in original probe order. Implicit arrays are copied
unchanged; they already contain every probe/qualifying pair. The header mean
is recomputed over the new training ratings, as in the original full dataset.
The A/B mask is saved as uint8 (1 = A) beside the output.
"""
import argparse
import json
import pathlib

import numpy as np

from gbdt_blend import HEADER, load
from train_holdout import digest


def split_mask(n, seed=20261031):
    return np.random.default_rng(seed).random(n) < .5


def build(original, output, seed=20261031):
    output = pathlib.Path(output)
    for path in (output, pathlib.Path(str(output) + '.json'), pathlib.Path(str(output) + '.amask')):
        if path.exists():
            raise FileExistsError(path)
    o = load(original)
    p = o['probe']
    a = split_mask(len(p), seed)
    users = int(o['users'])
    counts = np.diff(o['train_offsets']).astype(np.int64)
    add = np.bincount(p['user'][a].astype(np.int64), minlength=users)
    new_offsets = np.concatenate([[0], np.cumsum(counts + add)]).astype(np.int64)
    n = int(new_offsets[-1])
    items, days, ratings = (np.empty(n, '<u2'), np.empty(n, '<u2'), np.empty(n, 'u1'))
    # existing rows go to the front of each user's new segment
    uid = np.repeat(np.arange(users), counts)
    dest = new_offsets[:-1][uid] + (np.arange(len(uid)) - o['train_offsets'][:-1].astype(np.int64)[uid])
    items[dest], days[dest], ratings[dest] = o['train_items'], o['train_days'], o['train_ratings']
    # A rows follow, in probe order within each user
    pa = p[a]
    au = pa['user'].astype(np.int64)
    order = np.argsort(au, kind='stable')
    au_s = au[order]
    first = np.searchsorted(au_s, au_s, 'left')
    rank = np.arange(len(au_s)) - first
    dest_a = new_offsets[:-1][au_s] + counts[au_s] + rank
    items[dest_a], days[dest_a], ratings[dest_a] = pa['item'][order], pa['day'][order], pa['rating'][order]
    raw = np.memmap(original, mode='r', dtype=np.uint8)
    header = raw[:HEADER.itemsize].view(HEADER).copy()
    header[0]['train_count'] = n
    header[0]['probe_count'] = int((~a).sum())
    header[0]['mean'] = float(ratings.astype(np.float64).mean())
    tmp = pathlib.Path(str(output) + '.tmp')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open('xb') as f:
            for part in (header, new_offsets.astype('<u8'), items, days, ratings, p[~a],
                         o['implicit_offsets'], o['implicit_items'], o['implicit_days']):
                np.asarray(part).tofile(f)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    a.astype('u1').tofile(str(output) + '.amask')
    check = load(output)
    assert len(check['train_ratings']) == len(o['train_ratings']) + int(a.sum())
    assert np.array_equal(check['probe'], p[~a])
    report = dict(original=str(original), original_sha256=digest(original), seed=seed,
                  a_rows=int(a.sum()), b_rows=int((~a).sum()), train_count=n, mean=float(header[0]['mean']),
                  output=str(output), output_sha256=digest(output), amask_sha256=digest(str(output) + '.amask'))
    pathlib.Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original', default='data/processed/netflix.bin')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.original, args.output), indent=2))


if __name__ == '__main__':
    main()
