#!/usr/bin/env python3
"""Independent complete native query/source/known-map admission after transport."""
import datetime as dt
from pathlib import Path
import numpy as np
import upstream_remote15h_native_stream as transport
import upstream_remote15h as original
import upstream_own6_tight_calibration as checks

u = transport.u


def main():
    report, root = transport.REPORT, transport.RUN
    output = report/'independent-verification.json'
    if output.exists():
        raise ValueError('Existing independent full native query admission retained')
    plan_path = report/'plan.json'; plan = u.read(plan_path)
    manifest_path = report/transport.ARCHIVE; manifest = u.read(manifest_path)
    proof_path = report/'transport-verification.json'; proof = u.read(proof_path)
    if (proof['state'] != 'complete_all_native_archive_part_and_member_hashes_with_materialized_queries'
            or proof['plan_sha256'] != u.sha(plan_path) or plan['manifest_sha256'] != u.sha(manifest_path)
            or proof['recovered_labels_read'] is not False or manifest['qualifying_labels_included'] is not False
            or set(proof['members']) != set(manifest['members'])
            or [r['name'] for r in proof['parts']] != sorted(manifest['parts'])):
        raise ValueError('Complete original two-part/member stream proof required')
    for part in proof['parts']:
        if (part['bytes'] != manifest['parts'][part['name']]['bytes']
                or part['sha256'] != manifest['parts'][part['name']]['sha256']):
            raise ValueError('Complete original compressed part identity differs')
    bindings = dict(plan['source_sha256'])
    def bind(path, expected=None):
        key = str(path.relative_to(u.ROOT)); digest = u.sha(path) if expected is None else expected
        if key in bindings and bindings[key] != digest:
            raise ValueError('Original source/hash expectation conflicts')
        bindings[key] = digest
    for name, expected in manifest['members'].items():
        saved = proof['members'][name]
        if saved['bytes'] != expected['bytes'] or saved['sha256'] != expected['sha256']:
            raise ValueError('Every original plaintext archive member identity differs')
        if saved['materialized'] != (name in plan['selected']):
            raise ValueError('Original native materialization scope changed')
        if saved['materialized']:
            bind(root/name, expected['sha256'])
    base_path = u.ROOT/'experiments/upstream-remote15h/plan.json'
    base = u.read(base_path)
    if base['source_pin'] != u.PIN or base['qualifying_labels_read'] or base['public_prediction_values_read']:
        raise ValueError('Original own source/data plan required')
    split = u.ROOT/'artifacts/upstream-split3-pilot/source/data'
    for directory, collection in [(u.FRESH, base['source_files_sha256']),
            (u.FRESH/'data', base['data_sha256']), (split, base['split3_sha256'])]:
        for name, digest in collection.items():
            bind(directory/name, digest)
    for job in transport.JOBS:
        metadata = report/'remote'/job
        if u.read(metadata/'plan.json') != base:
            raise ValueError('Native producer must retain original source/data/full recipe plan')
        receipt = u.read(metadata/(job+'-receipt.json'))
        for name, digest in receipt['outputs'].items():
            if manifest['members']['source/'+name]['sha256'] != digest:
                raise ValueError('Every native numeric member must bind its original producer')
        for path in metadata.iterdir():
            bind(path)
    recent_path = u.ROOT/'experiments/upstream-recent15-own-pairs/verification.json'
    recent = u.read(recent_path)
    if recent['state'] != 'complete_fifteen_fresh_own_pairs_and_original_producer_lineage':
        raise ValueError('Original compact native pair admission required')
    for name, digest in recent['source_sha256'].items():
        bind(u.ROOT/name, digest)
    for path in [Path(__file__), Path(original.__file__), base_path, recent_path,
                 plan_path, manifest_path, proof_path]:
        bind(path)
    checks.verify_bindings(bindings)
    source = original.arrays(split/'probe_3', labels=True)
    known = original.arrays(u.FRESH/'data/probex', labels=True)
    mapping = original.projection(source, known)
    if not np.array_equal(mapping, np.load(root/'probex-row-map.npy')):
        raise ValueError('Independent complete canonical native row map differs')
    if u.sha(root/'probex-row-map.npy') != base['row_map_sha256']:
        raise ValueError('Original full native row map hash differs')
    # The exact canonical map proves known is a subset of this full holdout.
    # Check every native holdout identity, including all632,330 extra rows.
    original.assert_disjoint(np.sort(original.keys(*source[:3])), original.arrays(split/'train_3'))
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying fitting targets must remain zero')
    queries = []
    for name in plan['selected']:
        if name.endswith('.probe_3.npy') or name.endswith('.qual.npy'):
            array = np.load(root/name, mmap_mode='r')
            count = 1760494 if name.endswith('.probe_3.npy') else 2817131
            if array.shape != (count,) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('Every complete finite original native query required')
            queries.append(dict(path=name, rows=count, sha256=manifest['members'][name]['sha256']))
            del array
    if len(queries) != 96:
        raise ValueError('All96 original complete native query vectors required')
    compared = []
    for row in recent['rows']:
        if row['origin'] not in transport.JOBS:
            continue
        native = np.load(root/'source/preds_3'/(row['name']+'.probe_3.npy'), mmap_mode='r')
        q = np.load(root/'source/preds_3'/(row['name']+'.qual.npy'), mmap_mode='r')
        if (not np.array_equal(native[mapping].view('u4'), np.load(u.ROOT/row['paired_files']['probex']).view('u4'))
                or not np.array_equal(q.view('u4'), np.load(u.ROOT/row['paired_files']['qual']).view('u4'))):
            raise ValueError('Every previous compact known/native qualifying prediction bit must match')
        compared.append(row['name']); del native, q
    if len(compared) != 9:
        raise ValueError('All nine original compact prediction pairs must replay from full native queries')
    checks.verify_bindings(bindings)
    u.write(output, dict(state='complete_independent_original_native_queries_source_and_all_previous_compact_projections',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(), transport_verification_sha256=u.sha(proof_path),
        manifest_sha256=u.sha(manifest_path), queries=queries, previous_compact_pairs_replayed=compared,
        complete_local_known_map_and_training_disjointness_recomputed=True,
        complete_native_holdout_absent_from_training=True,native_holdout_rows=1760494,
        original_source_data_and_native_receipt_hashes_verified=True, source_sha256=bindings,
        qualifying_labels_read=False, models_fitted=0, qualifying_score_claimed=False,
        scope='Original fresh own remote producers only. All96 full native query vectors and nine earlier compact projections checked locally. Larger matched holdout studies and new feature batches require separate declarations; current own352 inputs unchanged.'))
    print('Independent full native source/query admission complete; all nine original compact pairs replay', flush=True)


if __name__ == '__main__':
    main()
