#!/usr/bin/env python3
"""Admit every completed original temporal100r snapshot without epoch selection."""
from pathlib import Path
import hashlib
import numpy as np
import upstream_split3_next3 as producer
import upstream_own6_tight_calibration as provenance

u = producer.u
REPORT = u.ROOT / 'experiments/upstream-tx100r-snapshot-admission'
RUN = u.ROOT / 'artifacts/upstream-tx100r-snapshot-admission'
NAME = 's3-tx100r'


def clipped_hash(a):
    digest = hashlib.sha256()
    for lo in range(0, len(a), 65536):
        digest.update(np.clip(a[lo:lo+65536], 0, 6).astype('<f4').tobytes())
    return digest.hexdigest()


def original():
    path = producer.REPORT / (NAME+'-verification.json')
    proof = u.read(path)
    anchor_path = u.ROOT / 'experiments/upstream-tx100-serial-memory-recovery/completion.json'
    anchor = u.read(anchor_path)
    if (anchor['state'] != 'complete_original_TX22_22_then_resumed_same_CF_native_and_owner'
            or anchor['temporal_verification_sha256'] != u.sha(path)):
        raise ValueError('Previously completed exact temporal trajectory proof differs')
    if (proof['state'] != 'complete_original_two_phase_split3_model'
            or proof['source_pin'] != u.PIN or proof['job'] != NAME
            or proof['epochs_per_phase'] != 22 or proof['native_outputs_verified'] != 49
            or not proof['both_full_epoch_traces'] or not proof['full_known_projection_recomputed']
            or not proof['known_training_disjointness_bound_to_unchanged_original_data']
            or not proof['qualifying_targets_zero'] or proof['qualifying_labels_read'] is not False):
        raise ValueError('Complete original22/22 temporal source/data/output proof required')
    bindings = dict(proof['bindings'])
    bindings[str(path.relative_to(u.ROOT))] = u.sha(path)
    bindings[str(anchor_path.relative_to(u.ROOT))] = u.sha(anchor_path)
    provenance.verify_bindings(bindings)
    native = producer.pilot.arrays(producer.TREE / 'data/probe_3', labels=True)
    known = producer.pilot.arrays(u.FRESH / 'data/probex', labels=True)
    mapping = producer.pilot.projection(native, known)
    saved = producer.pilot.RUN / 'probex-row-map.npy'
    if not np.array_equal(mapping, np.load(saved)) or mapping.shape != (1128164,):
        raise ValueError('Original canonical known identity projection differs')
    if np.any(np.load(producer.TREE / 'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying fitting targets must remain zero')
    return bindings, mapping


def admit():
    if REPORT.exists() or RUN.exists():
        raise ValueError('Existing complete or partial temporal admission retained')
    bindings, mapping = original()
    for path in [Path(__file__), u.ROOT / 'TX100R_SNAPSHOT_ADMISSION_PLAN.md',
                 u.ROOT / 'scripts/verify_upstream_tx100r_snapshot_admission.py']:
        key, digest = str(path.relative_to(u.ROOT)), u.sha(path)
        if key in bindings and bindings[key] != digest:
            raise ValueError('Original source expectation conflicts')
        bindings[key] = digest
    REPORT.mkdir(); RUN.mkdir()
    names = [NAME]+[NAME+'_ep%02d' % epoch for epoch in range(1, 23)]
    u.write(REPORT / 'plan.json', dict(state='declared_entire_completed_original_temporal100r_trajectory',
        names=names, source_sha256=bindings, qualifying_labels_read=False,
        numerical_public_features_read=False, epoch_selection=False))
    rows, aliases, payloads = [], [], {}
    for name in names:
        native = producer.TREE / 'preds_3' / (name+'.probe_3.npy')
        qual = producer.TREE / 'preds_3' / (name+'.qual.npy')
        for path in [native, qual]:
            key = str(path.relative_to(u.ROOT))
            if key not in bindings or u.sha(path) != bindings[key]:
                raise ValueError('Native snapshot lacks original full output binding')
        a = np.load(native, mmap_mode='r'); q = np.load(qual, mmap_mode='r')
        if (a.shape != (1760494,) or q.shape != (2817131,)
                or a.dtype != np.float32 or q.dtype != np.float32
                or not np.isfinite(a).all() or not np.isfinite(q).all()):
            raise ValueError('Complete finite original native query pair required')
        projected = np.asarray(a[mapping], dtype='<f4')
        pair = (clipped_hash(projected), clipped_hash(q))
        if pair in payloads:
            aliases.append(dict(name=name, exact_clipped_pair_alias_of=payloads[pair],
                canonical_clipped_payload_sha256=pair[0], qualifying_clipped_payload_sha256=pair[1]))
        else:
            target = RUN / (name+'.probex.npy'); np.save(target, projected)
            payloads[pair] = name
            rows.append(dict(name=name, paired_files=dict(probex=str(target.relative_to(u.ROOT)),
                qual=str(qual.relative_to(u.ROOT))), source_known=str(native.relative_to(u.ROOT)),
                canonical_clipped_payload_sha256=pair[0], qualifying_clipped_payload_sha256=pair[1],
                probex_sha256=u.sha(target), qual_sha256=u.sha(qual)))
    provenance.verify_bindings(bindings)
    u.write(REPORT / 'manifest.json', dict(state='complete_original_temporal100r_full_snapshot_projection_pending_replay',
        plan_sha256=u.sha(REPORT / 'plan.json'), rows=rows, aliases=aliases,
        total_original_pairs=len(names), distinct_clipped_complete_pairs=len(rows),
        qualifying_labels_read=False, neural_models_fitted=0, source_sha256=bindings))
    print('Admitted', len(rows), 'distinct own temporal snapshot pairs;', len(aliases), 'exact aliases', flush=True)


if __name__ == '__main__':
    admit()
