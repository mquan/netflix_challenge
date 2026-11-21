#!/usr/bin/env python3
"""Independently admit complete own source/native queries and every compact projection."""
import argparse
import datetime as dt
from pathlib import Path
import zipfile
import numpy as np
import upstream_remote_weekend_return as transport
import upstream_split3_pilot as rows

u, ops = transport.u, transport.ops


def compact_headers(path, counts=None):
    """Bound declared compressed array sizes/shapes before NumPy allocation."""
    counts = {'probex':1128164,'qual':2817131} if counts is None else counts
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {e.filename for e in entries} != {q+'.npy' for q in counts}:
            raise ValueError('Exact unique original paired compact array members required')
        for entry in entries:
            count = counts[entry.filename[:-4]]
            if entry.flag_bits&1 or not 4*count < entry.file_size <= 4*count+10000:
                raise ValueError('Bounded original float32 compact payload required')
            with archive.open(entry) as source:
                version = np.lib.format.read_magic(source)
                if version == (1,0): shape,fortran,dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2,0): shape,fortran,dtype = np.lib.format.read_array_header_2_0(source)
                else: raise ValueError('Original compact NPY header required')
                if shape != (count,) or fortran or dtype != np.dtype('<f4') or source.tell()+4*count != entry.file_size:
                    raise ValueError('Exact bounded original compact numerical shape required')


def verify(job, report, root, check):
    output = report/'independent-verification.json'
    if output.exists(): raise ValueError('Original full admission retained')
    plan_path, proof_path = report/'plan.json', report/'transport-verification.json'
    plan, proof = u.read(plan_path), u.read(proof_path); base = u.read(transport.PLAN)
    blobs = {name:(report/name).read_bytes() for name in plan['metadata_sha256']}
    if any(transport.digest(blobs[name]) != sha for name,sha in plan['metadata_sha256'].items()):
        raise ValueError('Original pinned execution metadata changed')
    index, exported, expected = transport.validate_metadata(job, blobs)
    if (plan['job'] != job['name'] or plan['state'] != 'declared_full_weekend_archive_import'
        or proof['state'] != 'complete_all_weekend_archive_bytes_members_and_native_finiteness'
        or proof['plan_sha256'] != ops.sha(plan_path) or set(proof['members']) != set(expected)
        or proof['qualifying_labels_read'] is not False
        or plan['qualifying_labels_read'] is not False or plan['numerical_public_features_read'] is not False
        or proof['parts'] != [{k:p[k] for k in ['file','bytes','sha256']} for p in index['parts']]):
        raise ValueError('Full original compressed/member transport proof required')
    bindings = dict(plan['source_sha256'])
    def bind(path, sha):
        relative = str(path.relative_to(u.ROOT)); ops.merge_bindings(bindings, {relative:sha})
        if ops.sha(path) != sha: raise ValueError('Original immutable source differs: '+relative)
        check()
    for name, sha in expected.items():
        record = proof['members'][name]
        materialized = not name.startswith('native/') or name.endswith(('.probe_3.npy','.qual.npy'))
        if record['sha256'] != sha or record['materialized'] != materialized:
            raise ValueError('Every original archive member identity/scope required')
        if name.startswith('native/') and name.endswith('.npy'):
            shape = transport.native_shapes(job)[name[len('native/'):]]
            if record['native_shape'] != list(shape) or record['native_finite_checked'] is not True:
                raise ValueError('Every native training/query/factor payload must have been fully checked')
        if materialized: bind(root/name, sha)
    for rel, sha in dict(bindings).items():
        if ops.sha(u.ROOT/rel) != sha: raise ValueError('Frozen import source changed')
        check()
    # Exported original plan and preparation must bind the admitted binaries,
    # complete split, row map and unchanged original numerical source.
    if u.read(root/'plan.json') != base: raise ValueError('Original coordinator plan required')
    prep = u.read(root/'preparation.json'); launch = u.read(root/(job['name']+'-launch.json'))
    if (prep['state'] != 'prepared_own_data_only' or prep['plan_sha256'] != ops.sha(transport.PLAN)
        or prep['source_pin'] != base['source_pin'] or prep['row_map_sha256'] != base['row_map_sha256']
        or launch['preparation_sha256'] != ops.sha(root/'preparation.json')
        or any(prep[k] is not True for k in ['known_rows_absent_from_training','qualifying_targets_zero'])
        or any(prep[k] is not False for k in ['qualifying_labels_read','public_prediction_values_read'])
        or set(prep['binary_sha256']) != {'target/release/fit','target/release/split_3'}):
        raise ValueError('Full original source/data/split/binary preparation required')
    for rel, sha in prep['binary_sha256'].items(): bind(root/'source'/rel, sha)
    split = u.ROOT/'artifacts/upstream-split3-pilot/source/data'
    for directory, collection in [(u.FRESH, base['source_files_sha256']),
                                   (u.FRESH/'data', base['data_sha256']), (split, base['split3_sha256'])]:
        for rel, sha in collection.items(): bind(directory/rel, sha)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy', mmap_mode='r', allow_pickle=False)):
        raise ValueError('Qualifying fitting targets must remain zero')
    source, known = rows.arrays(split/'probe_3', labels=True), rows.arrays(u.FRESH/'data/probex', labels=True)
    mapping = rows.projection(source, known); check()
    # Canonical NPY projection is hashed without writing another matrix/cache.
    import io, hashlib
    payload = io.BytesIO(); np.save(payload, mapping, allow_pickle=False)
    if hashlib.sha256(payload.getbuffer()).hexdigest() != base['row_map_sha256']:
        raise ValueError('Independent complete known projection hash differs')
    ordered = np.sort(rows.keys(*source[:3])); training = rows.arrays(split/'train_3')
    for lo in range(0,len(training[0]),262144):
        block = [a[lo:lo+262144] for a in training]
        keys = rows.keys(*block); ix = np.searchsorted(ordered, keys); inside = ix<len(ordered)
        if np.any(ordered[ix[inside]] == keys[inside]): raise ValueError('Native holdout identity appears in training')
        check()
    del training, block, keys, ix, inside
    compared = []; canonical = {}
    for row in exported['records']:
        column = row['column']; native = {}
        for query, count in [('probe_3',1760494),('qual',2817131)]:
            values = np.load(root/'native/preds_3'/(column+'.'+query+'.npy'), mmap_mode='r', allow_pickle=False)
            if values.dtype != np.dtype('<f4') or values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError('Every original full native query must remain finite and complete')
            native[query] = values
        compact_headers(root/row['file'])
        with np.load(root/row['file'], allow_pickle=False) as saved:
            if set(saved.files) != {'probex','qual'}: raise ValueError('Exact canonical paired compact keys required')
            for query, actual in [('probex',native['probe_3'][mapping]),('qual',native['qual'])]:
                compact = saved[query]
                if compact.dtype != np.dtype('<f4') or compact.shape != actual.shape or not np.array_equal(compact.view('u4'),actual.view('u4')):
                    raise ValueError('Every compact known/qualifying bit must replay from full native queries')
                # Complete-pair clipped hashes support later score-independent aliases.
                canonical.setdefault(column,{})[query] = hashlib.sha256(np.clip(actual,0,6).astype('<f4').tobytes()).hexdigest()
        compared.append(column); check()
    name = job['name']; final = name+'_ep%02d'%job['epochs_per_phase']
    for query in ['probe_3','qual']:
        a = np.load(root/'native/preds_3'/(name+'.'+query+'.npy'), mmap_mode='r', allow_pickle=False)
        b = np.load(root/'native/preds_3'/(final+'.'+query+'.npy'), mmap_mode='r', allow_pickle=False)
        if not np.array_equal(a.view('u4'), b.view('u4')): raise ValueError('Terminal/final epoch original bits differ')
    for path in [Path(__file__), Path(rows.__file__), plan_path, proof_path]:
        bind(path, ops.sha(path))
    for name, sha in bindings.items():
        if ops.sha(u.ROOT/name) != sha: raise ValueError('Full immutable admission closure changed')
        check()
    u.write(output, dict(state='complete_independent_weekend_full_source_queries_projection_disjointness_and_all_compact_bits',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(), job=job['name'], remote_commit=plan['remote_commit'],
        transport_verification_sha256=ops.sha(proof_path),source_sha256=bindings,
        rows=dict(native_holdout=1760494,known=1128164,qualifying=2817131,training_scanned=98720013),
        all_pairs=compared,clipped_pair_sha256=canonical,qualifying_labels_read=False,
        numerical_public_features_read=False, models_fitted=0,score_claimed=False,
        full_native_training_and_factor_payloads_stream_verified=True,
        scope='Admit all original completed epoch/terminal pairs without outcome selection. Large side arrays remain losslessly preserved in original verified remote release archives.'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--job',required=True); args=parser.parse_args()
    from upstream_remote_weekend_return_guard import Guard
    guard=Guard(args.job)
    verify(transport.job_spec(args.job),u.ROOT/'experiments/upstream-remote-weekend-return'/args.job,
           u.ROOT/'artifacts/upstream-remote-weekend-return'/args.job,guard.check)


if __name__=='__main__':main()
