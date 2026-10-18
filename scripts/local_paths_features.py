#!/usr/bin/env python3
"""Export label-free local-path features in original query order for blending."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from gbdt_blend import load
from local_paths_pilot import TrainingIndex, summarize
from train_holdout import digest


POOLED_NAMES = (
    ['baseline', 'knn', 'user_mean', 'item_mean']
    + ['log_' + x for x in ['user_count', 'candidate_count', 'history_count', 'neighbor_count', 'path_count']]
    + [f'{source}_rating_{r}' for source in ['history', 'target_raters', 'shared'] for r in range(1, 6)]
    + ['cross_query_gap_years', 'shared_pair_gap_years', 'neighbor_target_gap_years']
)
PATH_NAMES = [f'triplet_{u}{v}{i}' for u in range(1, 6) for v in range(1, 6) for i in range(1, 6)] + [
    'difference_mean', 'difference_mse', 'exact_agreement', 'neighbor_difference_std',
    'neighbor_mse_mean', 'overlap_max', 'overlap_mean', 'overlap_std',
]
STATE = {}


def extract_rows(d, index, start, stop, settings):
    """Only query identities/dates and training ratings enter the features."""
    result = np.empty((stop - start, len(POOLED_NAMES) + len(PATH_NAMES)), dtype='<f4')
    for k, row in enumerate(d['probe'][start:stop]):
        user, item, day = int(row['user']), int(row['item']), int(row['day'])
        graph = index.retrieve(user, item, day, **settings)
        pooled, paths, _ = summarize(d, graph, day)
        result[k] = np.r_[pooled, paths]
    if not np.isfinite(result).all():
        raise ValueError(f'nonfinite features in rows {start}:{stop}')
    return start, result


def worker(bounds):
    return extract_rows(STATE['data'], STATE['index'], *bounds, STATE['settings'])


def export_features(d, output, *, settings, processes=1, chunk_size=2000, max_queries=0):
    if processes < 1 or chunk_size < 1 or max_queries < 0:
        raise ValueError('positive processes/chunk_size and nonnegative max_queries required')
    if min(settings[k] for k in ('history', 'neighbors', 'candidates')) < 1 or settings['candidates'] < settings['neighbors']:
        raise ValueError('positive retrieval caps and candidates >= neighbors required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    n = min(max_queries, len(d['probe'])) if max_queries else len(d['probe'])
    if not n:
        raise ValueError('dataset has no queries')
    start = time.monotonic()
    index = TrainingIndex(d)
    index_seconds = time.monotonic() - start
    print(json.dumps(dict(stage='index_complete', seconds=index_seconds, query_rows=n)), flush=True)
    matrix_path = output / 'features.npy'
    matrix = np.lib.format.open_memmap(matrix_path, mode='w+', dtype='<f4', shape=(n, 160))
    STATE.update(data=d, index=index, settings=settings)
    bounds = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    pool = None
    extract_start = time.monotonic()
    done, last_log = 0, extract_start
    try:
        if processes == 1:
            results = map(worker, bounds)
        else:
            # The read-only ~0.8 GB inverted index is shared, not rebuilt per worker.
            pool = mp.get_context('fork').Pool(processes)
            results = pool.imap_unordered(worker, bounds, chunksize=1)
        for offset, values in results:
            matrix[offset:offset + len(values)] = values
            done += len(values)
            now = time.monotonic()
            if now - last_log >= 20 or done == n:
                print(json.dumps(dict(stage='extract', rows=done, total=n,
                                      seconds=now - extract_start,
                                      queries_per_second=done / (now - extract_start))), flush=True)
                last_log = now
        if pool:
            pool.close()
            pool.join()
    except BaseException:
        if pool:
            pool.terminate()
            pool.join()
        raise
    finally:
        STATE.clear()
    matrix.flush()
    extraction_seconds = time.monotonic() - extract_start
    groups, hashes = {}, {}
    for group, names, offset in [('pooled', POOLED_NAMES, 0), ('paths', PATH_NAMES, len(POOLED_NAMES))]:
        groups[group] = []
        for k, name in enumerate(names):
            path = output / f'{group}-{name}.f32'
            matrix[:, offset + k].tofile(path)
            groups[group].append(str(path.resolve()))
            hashes[str(path.resolve())] = digest(path)
    report = dict(query_rows=n, full_query_rows=len(d['probe']), settings=settings,
                  processes=processes, index_seconds=index_seconds, extraction_seconds=extraction_seconds,
                  queries_per_second=n / extraction_seconds, features=groups, outputs=hashes,
                  matrix_sha256=digest(matrix_path),
                  note='Training ratings only; no query labels read; original query row order.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--history', type=int, default=32)
    p.add_argument('--neighbors', type=int, default=32)
    p.add_argument('--candidates', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--max-queries', type=int, default=0, help='smoke test only; 0 exports every query')
    a = p.parse_args()
    before = digest(a.data)
    report = export_features(load(a.data), a.output, processes=a.processes, max_queries=a.max_queries,
                             settings={k: getattr(a, k) for k in ('history', 'neighbors', 'candidates', 'seed')})
    if digest(a.data) != before:
        raise ValueError('dataset changed during extraction')
    report.update(data=str(Path(a.data).resolve()), data_sha256=before,
                  sources={str(Path(__file__).with_name(f)): digest(Path(__file__).with_name(f))
                           for f in ('local_paths_features.py', 'local_paths_pilot.py', 'gbdt_blend.py')})
    (Path(a.output) / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('features', 'outputs', 'sources')}), flush=True)


if __name__ == '__main__':
    main()
