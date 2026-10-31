#!/usr/bin/env python3
"""Training-only, uniformly sampled other-rater context for the deferred E2 bet.

No model or rating-score fit. Query ratings and implicit-only events are never
used as context. The source query user is excluded even for masked train rows.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from gbdt_blend import load
from local_paths_pilot import TrainingIndex
from train_holdout import digest


def build(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    before = digest(data)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    d = load(data)
    if len(d['train_items']) >= 2**32:
        raise ValueError('training rows exceed uint32 index capacity')
    index = TrainingIndex(d)
    # Stable item sorting must preserve strictly increasing users per movie.
    # A duplicated user/movie rating would require a different exclusion rule.
    for item in range(1, d['items']+1):
        _, users = index.item_rows(item)
        if np.any(users[1:] <= users[:-1]):
            raise ValueError('duplicate or unordered user/movie training ratings')
    for name in ('rows', 'users', 'offsets'):
        np.save(output/(name+'.npy'), getattr(index, name))
    if digest(data) != before:
        raise ValueError('dataset changed while building context index')
    report = dict(data=str(data), data_sha256=before, users=d['users'], items=d['items'],
        train_rows=len(d['train_items']), seconds=time.monotonic()-started,
        source_sha256=digest(__file__), index_source_sha256=digest(Path(__file__).with_name('local_paths_pilot.py')),
        outputs={path.name: digest(path) for path in output.glob('*.npy')},
        contract='Index contains training ratings only; query labels and implicit-only pairs are absent.')
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


class Coraters:
    def __init__(self, data, directory):
        self.d = load(data)
        directory = Path(directory)
        report = json.loads((directory/'manifest.json').read_text())
        if digest(data) != report['data_sha256']:
            raise ValueError('context dataset differs from index provenance')
        for name, expected in report['outputs'].items():
            if digest(directory/name) != expected:
                raise ValueError('index payload changed')
        for name in ('rows', 'users', 'offsets'):
            setattr(self, name, np.load(directory/(name+'.npy'), mmap_mode='r'))
        if (len(self.rows) != report['train_rows'] or len(self.users) != len(self.rows)
                or len(self.offsets) != self.d['items']+2):
            raise ValueError('invalid index shape')

    def sample(self, user, item, day, *, seed, cap=64):
        user, item, day = int(user), int(item), int(day)
        if not (0 <= user < self.d['users'] and 1 <= item <= self.d['items']):
            raise ValueError('query ID outside dataset')
        if min(cap, seed+1) < 1 or day < 0:
            raise ValueError('invalid sampling settings')
        begin, end = map(int, self.offsets[item:item+2])
        owners = self.users[begin:end]
        position = int(np.searchsorted(owners, user))
        excluded = position < len(owners) and owners[position] == user
        eligible = len(owners)-int(excluded)
        size = min(cap, eligible)
        # Sampling is uniform over the complement of the one excluded owner.
        # No rating, date proximity, popularity or overlap score affects picks.
        rng = np.random.default_rng(np.random.SeedSequence([seed, user, item, day]))
        chosen = rng.choice(eligible, size, replace=False, shuffle=False)
        if excluded:
            chosen += chosen >= position
        chosen.sort()
        source = self.rows[begin+chosen].astype(np.int64)
        result = dict(rows=np.full(cap, -1, dtype=np.int64),
                      users=np.zeros(cap, dtype=np.uint32), ratings=np.zeros(cap, dtype=np.uint8),
                      days=np.zeros(cap, dtype=np.uint16), gaps=np.zeros(cap, dtype=np.int32),
                      mask=np.zeros(cap, dtype=bool), eligible=eligible)
        result['rows'][:size] = source
        result['users'][:size] = owners[chosen]
        result['ratings'][:size] = self.d['train_ratings'][source]
        result['days'][:size] = self.d['train_days'][source]
        result['gaps'][:size] = result['days'][:size].astype(np.int32)-day
        result['mask'][:size] = True
        return result

    def queries(self, rows, *, seed, cap=64):
        # Read named identity/date fields only, even if a rating field is present.
        result = [self.sample(row['user'], row['item'], row['day'], seed=seed, cap=cap) for row in rows]
        if not result:
            raise ValueError('empty query batch')
        return {key: np.stack([record[key] for record in result]) for key in result[0]}


def pilot(data, directory, output, count, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    index = Coraters(data, directory)
    ready = time.monotonic()
    queries = np.sort(np.random.default_rng(seed).choice(len(index.d['probe']),
        min(count, len(index.d['probe'])), replace=False))
    batch = index.queries(index.d['probe'][queries], seed=seed)
    seconds = time.monotonic()-ready
    valid = batch['mask']
    query_users = index.d['probe']['user'][queries]
    query_items = index.d['probe']['item'][queries]
    assert not np.any((batch['users'] == query_users[:, None]) & valid)
    assert np.all(index.d['train_items'][batch['rows'][valid]] == np.broadcast_to(query_items[:, None], valid.shape)[valid])
    assert np.array_equal(index.d['train_ratings'][batch['rows'][valid]], batch['ratings'][valid])
    np.savez(output/'sample.npz', queries=queries, **batch)
    counts = valid.sum(axis=1)
    report = dict(data=str(Path(data).resolve()), data_sha256=digest(data),
        index_manifest_sha256=digest(Path(directory)/'manifest.json'), source_sha256=digest(__file__),
        rows=len(queries), cap=64, seed=seed, sampling_seconds=seconds,
        queries_per_second=len(queries)/seconds, total_seconds=time.monotonic()-started,
        sampled_context_edges=int(valid.sum()), nonempty_query_fraction=float(np.mean(counts>0)),
        mean_context=float(counts.mean()), p5_context=float(np.percentile(counts,5)),
        output_sha256=digest(output/'sample.npz'),
        note='Feasibility/exclusion evidence only. No model training, query labels exported, or score gain.')
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'pilot'))
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index')
    parser.add_argument('--queries', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    if args.action == 'build':
        report = build(args.data, args.output)
    else:
        if not args.index or args.queries < 1:
            parser.error('pilot requires --index and positive --queries')
        report = pilot(args.data, args.index, args.output, args.queries, args.seed)
    print(json.dumps(report, indent=2), flush=True)
