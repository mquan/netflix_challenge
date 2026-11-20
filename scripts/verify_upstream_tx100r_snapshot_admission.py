#!/usr/bin/env python3
"""Separately replay all native known projections and complete clipped aliases."""
from pathlib import Path
import hashlib
import numpy as np
import upstream_tx100r_snapshot_admission as study

u = study.u


def digest(values):
    result = hashlib.sha256()
    for lo in range(0, len(values), 32768):
        a = np.asarray(values[lo:lo+32768], dtype=np.float32).copy()
        a[a < 0] = 0; a[a > 6] = 6
        result.update(a.astype('<f4').tobytes())
    return result.hexdigest()


def main():
    output = study.REPORT / 'independent-verification.json'
    if output.exists():
        raise ValueError('Existing independent temporal snapshot proof retained')
    manifest = u.read(study.REPORT / 'manifest.json')
    plan = u.read(study.REPORT / 'plan.json')
    if manifest['plan_sha256'] != u.sha(study.REPORT / 'plan.json') or manifest['source_sha256'] != plan['source_sha256']:
        raise ValueError('Original complete admission source binding differs')
    original, mapping = study.original()
    for key, expected in original.items():
        if manifest['source_sha256'].get(key) != expected:
            raise ValueError('Original producer expected hash not retained')
    study.provenance.verify_bindings(manifest['source_sha256'])
    rows = {r['name']:r for r in manifest['rows']}
    aliases = {r['name']:r for r in manifest['aliases']}
    if set(rows) & set(aliases) or set(rows) | set(aliases) != set(plan['names']) or len(plan['names']) != 23:
        raise ValueError('Every original terminal and22 snapshots required')
    seen = {}
    for name in plan['names']:
        native = np.load(study.producer.TREE / 'preds_3' / (name+'.probe_3.npy'), mmap_mode='r')
        qual = np.load(study.producer.TREE / 'preds_3' / (name+'.qual.npy'), mmap_mode='r')
        projected = native[mapping]
        pair = (digest(projected), digest(qual))
        record = rows.get(name, aliases.get(name))
        if pair != (record['canonical_clipped_payload_sha256'], record['qualifying_clipped_payload_sha256']):
            raise ValueError('Complete independently reconstructed pair hashes differ')
        if name in rows:
            path = u.ROOT / record['paired_files']['probex']; saved = np.load(path, mmap_mode='r')
            if (not np.array_equal(saved.view('u4'), projected.view('u4'))
                    or u.sha(path) != record['probex_sha256']
                    or u.sha(u.ROOT / record['paired_files']['qual']) != record['qual_sha256']
                    or record['paired_files']['qual'] != str((study.producer.TREE / 'preds_3' / (name+'.qual.npy')).relative_to(u.ROOT))
                    or pair in seen):
                raise ValueError('Actual original projection, pair path or exact alias policy differs')
            seen[pair] = name
        elif aliases[name]['exact_clipped_pair_alias_of'] != seen.get(pair):
            raise ValueError('Every omitted pair must equal its preceding complete pair')
    study.provenance.verify_bindings(manifest['source_sha256'])
    u.write(output, dict(state='complete_all_original_temporal100r_snapshots_known_projection_and_both_query_alias_replays',
        manifest_sha256=u.sha(study.REPORT / 'manifest.json'), verifier_sha256=u.sha(Path(__file__)),
        total_original_pairs=23, distinct_clipped_complete_pairs=len(rows), aliases=len(aliases),
        original_full_two_phase_source_and_training_disjointness_verified=True,
        qualifying_labels_read=False, models_fitted=0))


if __name__ == '__main__':
    main()
