#!/usr/bin/env python3
"""Separate full original352 input/statistic/ridge replay, no qualifying outcomes."""
from pathlib import Path
import hashlib
import numpy as np
import upstream_own352_temporal_data as study

u = study.u


def main():
    output = study.REPORT / 'independent-verification.json'
    if output.exists():
        raise ValueError('Existing independent complete temporal352 replay retained')
    manifest = u.read(study.REPORT / 'manifest.json'); study.check(manifest['source_sha256'])
    labels = np.load(study.RUN / 'labels.npy'); groups = np.load(study.RUN / 'groups.npy')
    reader = study.Inputs(manifest['original_rows'], manifest['recent_rows'])
    gram, rhs = np.zeros((10, 428, 428)), np.zeros((10, 428))
    hashes = [hashlib.sha256(), hashlib.sha256()]
    try:
        for lo in range(0, len(labels), 8192):
            hi = min(lo+8192, len(labels)); z, f = reader.block(lo, hi)
            oldz, oldf = reader.base.block(lo, hi)
            if (not np.array_equal(np.column_stack([z[:, :405], z[:, -1:]]).view('u8'), oldz.view('u8'))
                    or not np.array_equal(f[:, :556].view('u4'), oldf.view('u4'))):
                raise ValueError('Every retained original own330 input bit must match')
            for j, source in enumerate(reader.extra):
                values = source.read(lo, hi).copy(); values[values < 0] = 0; values[values > 6] = 6
                if (not np.array_equal(z[:, 405+j], values)
                        or not np.array_equal(f[:, 556+j].view('u4'), (values-np.float32(3.6)).view('u4'))):
                    raise ValueError('Every complete original temporal scalar transform must match')
            hashes[0].update(z.tobytes(order='C')); hashes[1].update(f.tobytes(order='C'))
            for group in range(10):
                keep = groups[lo:hi] == group; a = z[keep]
                gram[group] += a.T @ a; rhs[group] += a.T @ labels[lo:hi][keep]
        if [h.hexdigest() for h in hashes] != manifest['virtual_c_order_sha256']:
            raise ValueError('All complete original design/nonlinear input bits differ')
        with np.load(study.RUN / 'stats.npz') as saved:
            if not np.array_equal(gram, saved['gram']) or not np.array_equal(rhs, saved['rhs']):
                raise ValueError('All original full group sufficient statistics differ')
        reg = np.eye(428)*1000; reg[-1, -1] = 0
        weights = np.stack([np.linalg.solve(gram[np.arange(10)//2 != fold].sum(0)+reg,
            rhs[np.arange(10)//2 != fold].sum(0)) for fold in range(5)])
        if not np.array_equal(weights, np.load(study.RUN / 'weights.npy')):
            raise ValueError('All complete eligible-user ridge solves differ')
        stored = np.fromfile(study.RUN / 'ridge.f32', dtype='<f4')
        for lo in range(0, len(labels), 8192):
            hi = min(lo+8192, len(labels)); z, _ = reader.block(lo, hi)
            for fold in range(5):
                keep = groups[lo:hi]//2 == fold
                expected = np.clip(z[keep] @ weights[fold], 1.05, 4.95).astype('<f4')
                if not np.array_equal(expected.view('u4'), stored[lo:hi][keep].view('u4')):
                    raise ValueError('Every actual stored canonical held-user ridge prediction must match')
    finally:
        reader.close()
    for name, expected in manifest['prepared_sha256'].items():
        if u.sha(study.RUN / name) != expected:
            raise ValueError('Prepared original canonical arrays changed')
    study.check(manifest['source_sha256'])
    score = study.score(stored, labels)
    if abs(score-manifest['ridge_rmse']) > 1e-12:
        raise ValueError('Separate canonical ridge score differs')
    u.write(output, dict(state='complete_all_original352_temporal_inputs_statistics_solves_and_prediction_bits',
        manifest_sha256=u.sha(study.REPORT / 'manifest.json'), verifier_sha256=u.sha(Path(__file__)),
        ridge_rmse=score, all_original330_input_bits_preserved=True,
        all22_original_temporal_scalar_transforms_equal=True, full_group_statistics_and_all_eligible_solves_equal=True,
        every_actual_stored_prediction_bit_equal=True, qualifying_labels_read=False, neural_models_fitted=0))


if __name__ == '__main__':
    main()
