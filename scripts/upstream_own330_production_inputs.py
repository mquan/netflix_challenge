#!/usr/bin/env python3
"""Bounded paired own330 production transport; no qualifying outcomes."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import datetime as dt
import hashlib
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_latent90_bank as parent
import upstream_warm59_screen as snapshots
import upstream_own330_native149_data as raw
import upstream_own330_mps as recipe

REPORT = u.ROOT / 'experiments/upstream-own330-production-inputs'
COUNTS = dict(probex=1128164, qual=2817131)


class Inputs:
    def __init__(self, query, rows):
        count = COUNTS[query]
        if len(rows) != 223:
            raise ValueError('All original59 plus fixed164 paired additions required')
        self.handles = []
        try:
            self.design = self.open(parent.RUN / (query + '-design.npy'), (count, 183), np.float64)
            self.features = self.open(parent.RUN / (query + '-features.npy'), (count, 333), np.float32)
            self.raw = [self.open(u.ROOT / r['paired_files'][query], (count,), np.float32) for r in rows]
        except BaseException:
            self.close()
            raise

    def open(self, path, shape, dtype):
        value = raw.Rows(path, shape, dtype)
        self.handles.append(value)
        return value

    def block(self, lo, hi):
        oldz, oldf = self.design.read(lo, hi), self.features.read(lo, hi)
        extra = np.column_stack([np.clip(a.read(lo, hi), 0, 6) for a in self.raw])
        z = np.column_stack([oldz[:, :-1], extra, oldz[:, -1:]])
        f = np.column_stack([oldf[:, :243], extra[:, :59] - np.float32(3.6),
                             oldf[:, 243:], extra[:, 59:] - np.float32(3.6)])
        if (z.shape != (hi-lo, 406) or f.shape != (hi-lo, 556) or
                z.dtype != np.float64 or f.dtype != np.float32 or
                not np.isfinite(z).all() or not np.isfinite(f).all()):
            raise ValueError('Complete finite original own330 operators required')
        return z, f

    def close(self):
        for value in self.handles:
            value.close()


def declaration():
    if REPORT.exists():
        raise ValueError('Existing paired production admission retained')
    pm = u.read(parent.REPORT / 'manifest.json')
    pp = parent.REPORT / 'column-verification.json'
    proof = u.read(pp)
    if proof['state'] != 'complete' or proof['manifest_sha256'] != u.sha(parent.REPORT / 'manifest.json'):
        raise ValueError('Original paired own107/all90 proof required')
    development = u.ROOT / 'experiments/upstream-own330-mps'
    manifest = u.read(development / 'manifest.json')
    verified = u.read(development / 'independent-verification.json')
    if (verified['state'] != 'complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays'
            or verified['manifest_sha256'] != u.sha(development / 'manifest.json')):
        raise ValueError('Completed actual original own330 models required')
    first, bindings = snapshots.inputs()
    additions = u.read(u.ROOT / 'experiments/upstream-own330-neural-inputs/manifest.json')['recent_rows']
    if len(first) != 59 or len(additions) != 164:
        raise ValueError('Complete original paired input population required')
    rows = first + additions
    for collection in [pm['source_sha256'], pm['code_sha256'], manifest['source_sha256'], manifest['code_sha256']]:
        for key, digest in collection.items():
            if key in bindings and bindings[key] != digest:
                raise ValueError('Conflicting original source expectation: ' + key)
            bindings[key] = digest
    for filename, digest in pm['files'].items():
        bindings[str((parent.RUN / filename).relative_to(u.ROOT))] = digest
    paths = [Path(__file__), u.ROOT / 'scripts/verify_upstream_own330_production_inputs.py',
             u.ROOT / 'OWN330_FULL_REFIT_PLAN.md', parent.REPORT / 'manifest.json', pp,
             development / 'manifest.json', development / 'independent-verification.json']
    receipts = []
    for fold in range(5):
        path = development / 'nn' / (str(fold) + '.json')
        paths.append(path)
        receipts.append(u.read(path))
    for row in rows:
        for query in COUNTS:
            key = row['paired_files'][query]
            if key not in bindings:
                raise ValueError('Input lacks original source expectation: ' + key)
    for path in paths:
        bindings[str(path.relative_to(u.ROOT))] = u.sha(path)
    raw.check(bindings)
    zeros = np.load(u.FRESH / 'data/qual/ratings.npy', mmap_mode='r')
    if np.any(zeros):
        raise ValueError('Qualifying targets must remain zero')
    epochs = [int(np.median([r['chosen_iterations'][j] for r in receipts])) for j in range(3)]
    if manifest['config'] != recipe.CONFIG or any(not 0 < e <= 40 for e in epochs):
        raise ValueError('Actual original config and selected budgets required')
    REPORT.mkdir()
    u.write(REPORT / 'declaration.json', dict(
        state='declared_complete_paired_own330_before_full_refit',
        created=dt.datetime.now(dt.timezone.utc).isoformat(), source_sha256=bindings,
        rows=rows, counts=COUNTS, config=manifest['config'], full_iterations=epochs,
        seeds=manifest['config']['seeds'], signal_indices=recipe.SIGNALS,
        context_indices=recipe.CONTEXT, design_columns=406, nonlinear_columns=556, nn_inputs=557,
        input_realization='Original paired own107/all90 production B plus original59 and fixed164 own inputs. Retained parent may differ from development A; no matched B-trained CV claim.',
        qualifying_labels_read=False, production_started=False))
    inputs_hashes = {}
    for query, count in COUNTS.items():
        inputs = Inputs(query, rows)
        hashes = [hashlib.sha256(), hashlib.sha256()]
        try:
            for lo in range(0, count, 8192):
                z, f = inputs.block(lo, min(lo+8192, count))
                hashes[0].update(z.tobytes(order='C'))
                hashes[1].update(f.tobytes(order='C'))
        finally:
            inputs.close()
        inputs_hashes[query] = [h.hexdigest() for h in hashes]
        u.write(REPORT / 'status.json', dict(state='complete_query_transport_pending_independent_replay', query=query))
    raw.check(bindings)
    u.write(REPORT / 'transport.json', dict(state='complete_both_full_paired_own330_input_streams',
        declaration_sha256=u.sha(REPORT / 'declaration.json'), c_order_sha256=inputs_hashes,
        qualifying_labels_read=False, numerical_parameters_fitted=False))


if __name__ == '__main__':
    declaration()
