#!/usr/bin/env python3
"""Export P2 neighbor structure and a degree-preserving correspondence shuffle."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from gbdt_blend import load
from local_paths_pilot import TrainingIndex, summarize
from train_holdout import digest

CAP = 32
FIELDS = dict(history_rating=('u1', (CAP,)), history_gap=('<i2', (CAP,)),
              neighbor_rating=('u1', (CAP,)), neighbor_gap=('<i2', (CAP,)),
              cross_rating=('u1', (CAP, CAP)), cross_gap=('<i2', (CAP, CAP)),
              shuffled_rating=('u1', (CAP, CAP)), shuffled_gap=('<i2', (CAP, CAP)),
              pooled=('<f4', (27,)))
STATE = {}


def shuffle_correspondence(ratings, gaps, seed):
    """Permute observed (rating, date) payloads among neighbors within each movie.

    Keeps the mask, both degree sequences, and each shared movie's rating/date
    multiset exactly. Per-neighbor rating histograms and triplets can change.
    This destroys rating correspondence; it does not alter the mask topology.
    """
    mask = ratings > 0
    rng = np.random.default_rng(seed)
    source = np.argsort(np.where(mask, rng.random(mask.shape), 2), axis=0, kind='stable')
    destination = np.argsort(~mask, axis=0, kind='stable')
    cols = np.broadcast_to(np.arange(ratings.shape[1]), ratings.shape)
    shuffled_r, shuffled_g = np.zeros_like(ratings), np.zeros_like(gaps)
    shuffled_r[destination, cols] = ratings[source, cols]
    shuffled_g[destination, cols] = gaps[source, cols]
    return shuffled_r, shuffled_g


def encode_query(d, index, q, seed):
    row = d['probe'][q]
    user, item, day = int(row['user']), int(row['item']), int(row['day'])
    graph = index.retrieve(user, item, day, history=CAP, neighbors=CAP, candidates=256, seed=seed)
    result = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in FIELDS.items()}
    for prefix, key in [('history', 'history_rows'), ('neighbor', 'target_rows')]:
        rows = graph[key]
        gaps = d['train_days'][rows].astype(np.int32) - day
        if np.any(np.abs(gaps) > 32767):
            raise ValueError('date gap exceeds int16 encoding')
        result[prefix + '_rating'][:len(rows)] = d['train_ratings'][rows]
        result[prefix + '_gap'][:len(rows)] = gaps
    cross = graph['cross_rows']
    v, j = np.nonzero(cross >= 0)
    rows = cross[v, j]
    gaps = d['train_days'][rows].astype(np.int32) - day
    if np.any(np.abs(gaps) > 32767):
        raise ValueError('date gap exceeds int16 encoding')
    result['cross_rating'][v, j] = d['train_ratings'][rows]
    result['cross_gap'][v, j] = gaps
    result['shuffled_rating'], result['shuffled_gap'] = shuffle_correspondence(
        result['cross_rating'], result['cross_gap'], [seed, user, item, day, 991])
    result['pooled'][:] = summarize(d, graph, day)[0]
    r, s = result['cross_rating'], result['shuffled_rating']
    g, t = result['cross_gap'], result['shuffled_gap']
    counts = lambda a: np.stack([(a == rating).sum(axis=1) for rating in range(1, 6)])
    diagnostics = np.array([
        (r > 0).sum(), np.count_nonzero((r > 0) != (s > 0)),
        np.count_nonzero(np.sort(r, axis=0) != np.sort(s, axis=0)),
        np.count_nonzero(np.sort(g, axis=0) != np.sort(t, axis=0)),
        np.count_nonzero(r != s), np.count_nonzero(g != t),
        np.abs(counts(r) - counts(s)).sum(), np.count_nonzero(np.any(counts(r) != counts(s), axis=0)),
        len(graph['target_rows']),
    ], dtype=np.int64)
    return result, diagnostics


def extract_chunk(bounds):
    start, stop = bounds
    out = {name: np.empty((stop - start, *shape), dtype=dtype) for name, (dtype, shape) in FIELDS.items()}
    diagnostics = np.zeros(9, dtype=np.int64)
    for k, q in enumerate(STATE['queries'][start:stop]):
        row, stats = encode_query(STATE['data'], STATE['index'], int(q), STATE['seed'])
        for name, values in row.items():
            out[name][k] = values
        diagnostics += stats
    return start, out, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True, help='frozen P2 base-column spec')
    p.add_argument('--output', required=True)
    p.add_argument('--queries', type=int, default=200000)
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--processes', type=int, default=4)
    a = p.parse_args()
    if a.queries < 1 or a.processes < 1:
        p.error('queries and processes must be positive')
    spec = json.loads(Path(a.spec).read_text())
    if digest(spec['data']) != spec['data_sha256']:
        raise ValueError('P2 data hash mismatch')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    d = load(spec['data'])
    n = min(a.queries, len(d['probe']))
    queries = np.sort(np.random.default_rng(a.seed).choice(len(d['probe']), n, replace=False))
    np.save(out / 'queries.npy', queries)
    matrices = {name: np.lib.format.open_memmap(out / (name + '.npy'), mode='w+',
                                              dtype=dtype, shape=(n, *shape))
                for name, (dtype, shape) in FIELDS.items()}
    STATE.update(data=d, index=TrainingIndex(d), queries=queries, seed=a.seed)
    total = np.zeros(9, dtype=np.int64)
    done, last_log = 0, time.monotonic()
    bounds = [(i, min(i + 500, n)) for i in range(0, n, 500)]
    with mp.get_context('fork').Pool(a.processes) as pool:
        for offset, rows, stats in pool.imap_unordered(extract_chunk, bounds):
            count = len(rows['pooled'])
            for name, values in rows.items():
                matrices[name][offset:offset + count] = values
            total += stats
            done += count
            if time.monotonic() - last_log > 20 or done == n:
                print(json.dumps(dict(rows=done, total=n, seconds=time.monotonic() - started)), flush=True)
                last_log = time.monotonic()
    STATE.clear()
    for matrix in matrices.values():
        matrix.flush()
    if np.any(total[1:4]):
        raise ValueError('correspondence shuffle failed preservation invariants')
    if digest(spec['data']) != spec['data_sha256']:
        raise ValueError('dataset changed during extraction')
    report = dict(config=vars(a), data=spec['data'], data_sha256=spec['data_sha256'],
                  spec_sha256=digest(a.spec), rows=n, full_p2_rows=len(d['probe']),
                  seconds=time.monotonic() - started,
                  outputs={path.name: digest(path) for path in sorted(out.glob('*.npy'))},
                  shuffle=dict(observed_edges=int(total[0]), mask_mismatches=int(total[1]),
                               column_rating_multiset_mismatches=int(total[2]),
                               column_date_multiset_mismatches=int(total[3]),
                               changed_rating_fraction=float(total[4] / max(total[0], 1)),
                               changed_date_fraction=float(total[5] / max(total[0], 1)),
                               neighbor_rating_histogram_l1=int(total[6]),
                               changed_neighbor_rating_histogram_fraction=float(total[7] / max(total[8], 1))),
                  note='Only training ratings enter context. Query labels are not exported. '
                       'Shuffle preserves mask topology and movie marginals, not neighbor rating distributions.')
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
