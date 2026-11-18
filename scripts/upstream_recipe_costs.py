#!/usr/bin/env python3
"""Resolve diagnostic replay bundles and source-derived memory lower bounds."""
import argparse
import datetime as dt
import subprocess
import numpy as np
import upstream_repro as u
import upstream_missing_screen as s


def metadata_counts():
    records = []
    for train, query in [('trainx', 'probex'), ('fulltrain', 'qual')]:
        paths = [u.FRESH/'data'/split/(name+'.npy') for split in [train, query]
                 for name in ['user_idxs', 'item_idxs', 'dates']]
        data = [np.load(p, mmap_mode='r', allow_pickle=False) for p in paths]
        keys = np.empty(len(data[0])+len(data[3]), dtype=np.uint64)
        pos = 0
        for idx in [0, 3]:
            for lo in range(0, len(data[idx]), 1000000):
                hi = min(lo+1000000, len(data[idx]))
                keys[pos+lo:pos+hi] = ((data[idx][lo:hi].astype(np.uint64) << 16) |
                                     data[idx+2][lo:hi].astype(np.uint16).astype(np.uint64))
            pos += len(data[idx])
        keys.sort()
        n_ud = 1+sum(np.count_nonzero(keys[lo:hi] != keys[lo-1:hi-1])
                     for lo in range(1, len(keys), 1000000) for hi in [min(lo+1000000, len(keys))])
        records.append(dict(training=train, query=query, users=int(data[0].max())+1,
                            items=int(data[1].max())+1, user_day_pairs=int(n_ud),
                            rows=len(data[0])+len(data[3]),
                            metadata_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}))
        del keys
    return dict(records=records, provenance='Independent metadata-only replication of Tx SparseUD union counts. This is not measured native training peak RAM.')


def tx_lower_bound(record, factors, low_memory, neighborhood_matrices=0):
    # tx.rs allocates ufeat/ufeat2/ycache, ifeat/yfeat/ifeat_freq;
    # non-low-memory also allocates yfeat_day and TWO dense user-day arrays.
    nu, ni, nud = record['users'], record['items'], record['user_day_pairs']
    factor_cells = 3*nu+18*ni+(0 if low_memory else ni+2*nud)
    factor_bytes = 4*factors*factor_cells
    neighbor_bytes = 4*ni*ni*neighborhood_matrices
    # Dataset owns i32 users, i32 items, i8 raw rating, f32 residual, i16 date.
    # Excludes is_test/count vectors, model biases/indices, intermediate buffers and allocator overhead.
    dataset_bytes = 15*record['rows']
    return dict(factor_bytes=factor_bytes, neighborhood_bytes=neighbor_bytes,
                dataset_bytes=dataset_bytes, minimum_total_bytes=factor_bytes+neighbor_bytes+dataset_bytes)


def verified_receipts():
    result = {}
    for path in sorted((u.BASE/'fresh-receipts').glob('*.json')):
        receipt = u.read(path)
        if receipt['source_commit'] != u.PIN or receipt['job'] != path.stem:
            raise ValueError('Invalid local execution receipt')
        for name, digest in receipt['outputs'].items():
            if not (u.FRESH/name).is_file() or u.sha(u.FRESH/name) != digest:
                raise ValueError('Changed native parent output: '+name)
        result[path.stem] = dict(receipt_sha256=u.sha(path), outputs_verified=len(receipt['outputs']),
            measured_wall_seconds=(dt.datetime.fromisoformat(receipt['finished'])-
                                   dt.datetime.fromisoformat(receipt['started'])).total_seconds(),
            source='Local receipt wall time; includes both native phases and job build. Peak RAM not recorded.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recount-memory', action='store_true')
    args = ap.parse_args()
    u.validate_fresh()
    metadata_path = s.REPORT/'memory-metadata.json'
    if args.recount_memory or not metadata_path.exists():
        u.write(metadata_path, metadata_counts())
    metadata = u.read(metadata_path)
    for row in metadata['records']:
        for path, digest in row['metadata_sha256'].items():
            if u.sha(u.ROOT/path) != digest:
                raise ValueError('Memory-count input changed')
    native = verified_receipts()
    manifest = u.read(s.REPORT/'manifest.json')
    screen = u.read(s.REPORT/'evaluation.json')
    if u.sha(u.BASE/'registry.json') != manifest['registry_sha256']:
        raise ValueError('Screened registry changed')
    if u.sha(s.REPORT/'manifest.json') != screen['manifest_sha256']:
        raise ValueError('Screened inputs changed')
    jobs = u.read(u.BASE/'registry.json')['jobs']
    # Explicit dispatcher settings, not a capacity inferred from model names.
    settings = {'tsvdx5-1000': (1000, False, 2), 'tsvdx5-400': (400, False, 2),
                'tsvdx5-300': (300, False, 2), 'tsvdx5-150': (150, False, 2),
                'tsvdx5-150o': (150, False, 2), 'tsvdx5-120o': (120, False, 2),
                'tsvdx6-300': (300, False, 4), 'tsvdx6-300-v2': (300, False, 4),
                'tsvdx4-4800lm': (4800, True, 0), 'tsvdx4-2400lm': (2400, True, 0)}
    hardware = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    bundles = []
    for root in sorted({entry['bundle'] for entry in manifest['entries']}):
        entries = [entry for entry in manifest['entries'] if entry['bundle'] == root]
        order = list(dict.fromkeys(job for entry in entries for job in entry['full_dependency_order']))
        active = [job for job in order if job not in s.SETUP]
        reusable = [job for job in active if job in native]
        remaining = [job for job in active if job not in native]
        scores = [r for r in screen['records'] if r['arm'] in
                  ['single:'+entry['name'] for entry in entries]+['bundle:'+root]]
        best = min(scores, key=lambda r:r['rmse'])
        record = dict(bundle=root, columns=[entry['name'] for entry in entries],
                      best_diagnostic_arm=best['arm'], best_diagnostic_rmse=best['rmse'],
                      diagnostic_delta=best['delta_vs_fresh'],
                      full_dependency_order=order, distinct_non_setup_jobs=len(active),
                      reusable_jobs=reusable, new_jobs=remaining, commands={k:jobs[k]['command'] for k in remaining},
                      unresolved=any(entry['producer'] is None for entry in entries),
                      historical_epoch_coefficients=any(entry['historical_epoch_coefficients'] for entry in entries),
                      exact_new_bundle_time_seconds=None, exact_peak_memory_bytes=None,
                      missing_cost_evidence='Unmeasured new jobs require profiling; family/capacity is insufficient to claim actual execution time.')
        if root in settings:
            factors, low_memory, neighbors = settings[root]
            record['tx_settings'] = dict(factors=factors, low_memory=low_memory, neighborhood_matrices=neighbors)
            record['source_derived_memory_lower_bounds'] = [dict(training=row['training'], **tx_lower_bound(row, factors, low_memory, neighbors)) for row in metadata['records']]
            record['exceeds_physical_ram_before_overheads'] = any(v['minimum_total_bytes'] > hardware for v in record['source_derived_memory_lower_bounds'])
        bundles.append(record)
    sources = ['src/tx.rs','src/lib.rs','src/bin/tsvdx4-new.rs','src/bin/tsvdx5-new.rs',
               'src/bin/tsvdx6-300-new.rs','src/bin/tsvdx6-300-v2-new.rs']
    u.write(s.REPORT/'recipe-costs.json',dict(source_commit=u.PIN, hardware_memory_bytes=hardware,
             scope='Complete native dependencies from resolved registry, verified local reusable outputs, measured historical job times, source-derived memory LOWER BOUNDS. New-job times and native RAM are unknown unless measured separately.',
             memory_metadata_sha256=u.sha(metadata_path),
             source_sha256={name:u.sha(u.SOURCE/name) for name in sources}, verified_local_receipts=native,
             ranked_bundles=sorted(bundles, key=lambda row:row['best_diagnostic_rmse'])))


if __name__ == '__main__':
    main()
