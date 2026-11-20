#!/usr/bin/env python3
"""Complete own330 plus all22 original temporal100r pairs; bounded transport."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import hashlib
from pathlib import Path
import numpy as np
import upstream_own330_neural_inputs as base
import upstream_tx100r_snapshot_admission as added

u, n = base.u, base.raw.n
RUN = u.ROOT / 'artifacts/upstream-own352-temporal-data'
REPORT = u.ROOT / 'experiments/upstream-own352-temporal-data'
ROWS, DESIGN, FEATURES = 1128164, 428, 578
check, score = base.check, base.score


def sources():
    old = u.read(base.REPORT / 'manifest.json')
    old_proof = u.read(base.REPORT / 'independent-verification.json')
    extra = u.read(added.REPORT / 'manifest.json')
    extra_proof = u.read(added.REPORT / 'independent-verification.json')
    if (old_proof['state'] != 'complete_all_original330_design_and_nonlinear_input_bits_replayed'
            or old_proof['manifest_sha256'] != u.sha(base.REPORT / 'manifest.json')):
        raise ValueError('Complete original own330 operators required')
    if (extra_proof['state'] != 'complete_all_original_temporal100r_snapshots_known_projection_and_both_query_alias_replays'
            or extra_proof['manifest_sha256'] != u.sha(added.REPORT / 'manifest.json')
            or len(extra['rows']) != 22 or len(extra['aliases']) != 1):
        raise ValueError('Entire independently replayed temporal100r trajectory required')
    bindings = dict(old['source_sha256'])
    for key, digest in extra['source_sha256'].items():
        if key in bindings and bindings[key] != digest:
            raise ValueError('Original own source expectation conflicts')
        bindings[key] = digest
    for name, digest in old['prepared_sha256'].items():
        bindings[str((base.RUN / name).relative_to(u.ROOT))] = digest
    for row in extra['rows']:
        for query in ['probex', 'qual']:
            key, digest = row['paired_files'][query], row[query+'_sha256']
            if key in bindings and bindings[key] != digest:
                raise ValueError('Original paired prediction expectation conflicts')
            bindings[key] = digest
    for path in [Path(__file__), u.ROOT / 'scripts/verify_upstream_own352_temporal_data.py',
                 u.ROOT / 'OWN352_TEMPORAL_PLAN.md', base.REPORT / 'manifest.json',
                 base.REPORT / 'independent-verification.json', added.REPORT / 'manifest.json',
                 added.REPORT / 'independent-verification.json']:
        key, digest = str(path.relative_to(u.ROOT)), u.sha(path)
        if key in bindings and bindings[key] != digest:
            raise ValueError('Original complete input code/proof changed')
        bindings[key] = digest
    return bindings, old['recent_rows'], extra['rows']


class Inputs:
    def __init__(self, original, recent):
        self.base = base.Inputs(original)
        self.extra = []
        try:
            if len(recent) != 22:
                raise ValueError('All22 temporal predictors required')
            self.extra = [base.raw.Rows(u.ROOT/r['paired_files']['probex'], (ROWS,), np.float32) for r in recent]
        except BaseException:
            self.close(); raise

    def block(self, lo, hi):
        z, f = self.base.block(lo, hi)
        extra = np.column_stack([np.clip(a.read(lo, hi), 0, 6) for a in self.extra])
        return (np.column_stack([z[:, :-1], extra, z[:, -1:]]),
                np.column_stack([f, extra-np.float32(3.6)]))

    def close(self):
        self.base.close()
        for value in self.extra:
            value.close()


def main():
    if RUN.exists() or REPORT.exists():
        raise ValueError('Existing full or partial temporal352 study retained')
    bindings, original, recent = sources(); check(bindings)
    RUN.mkdir(); REPORT.mkdir()
    for name in ['labels.npy', 'users.npy', 'groups.npy']:
        u.link_or_copy(base.RUN / name, RUN / name)
    labels, groups = np.load(RUN / 'labels.npy'), np.load(RUN / 'groups.npy')
    gram, rhs = np.zeros((10, DESIGN, DESIGN)), np.zeros((10, DESIGN))
    hashes = [hashlib.sha256(), hashlib.sha256()]
    retained = [hashlib.sha256(), hashlib.sha256()]
    inputs = Inputs(original, recent)
    try:
        for lo in range(0, ROWS, 8192):
            hi = min(lo+8192, ROWS); z, f = inputs.block(lo, hi)
            if (z.shape != (hi-lo, DESIGN) or f.shape != (hi-lo, FEATURES)
                    or z.dtype != np.float64 or f.dtype != np.float32
                    or not np.isfinite(z).all() or not np.isfinite(f).all()):
                raise ValueError('Complete finite original temporal352 operators required')
            hashes[0].update(z.tobytes(order='C')); hashes[1].update(f.tobytes(order='C'))
            retained[0].update(np.column_stack([z[:, :405], z[:, -1:]]).tobytes(order='C'))
            retained[1].update(f[:, :556].tobytes(order='C'))
            for group in range(10):
                keep = groups[lo:hi] == group; a = z[keep]
                gram[group] += a.T @ a; rhs[group] += a.T @ labels[lo:hi][keep]
        if [h.hexdigest() for h in retained] != u.read(base.REPORT / 'manifest.json')['virtual_c_order_sha256']:
            raise ValueError('Every original own330 design/nonlinear input bit must match')
        weights = np.stack([n.ridge_weights(gram, rhs, np.arange(10)//2 != fold, 1000) for fold in range(5)])
        prediction = np.empty(ROWS, dtype='<f4')
        for lo in range(0, ROWS, 8192):
            hi = min(lo+8192, ROWS); z, _ = inputs.block(lo, hi)
            for fold in range(5):
                keep = groups[lo:hi]//2 == fold
                prediction[lo:hi][keep] = np.clip(z[keep] @ weights[fold], 1.05, 4.95)
    finally:
        inputs.close()
    np.savez(RUN / 'stats.npz', gram=gram, rhs=rhs); np.save(RUN / 'weights.npy', weights)
    prediction.tofile(RUN / 'ridge.f32'); check(bindings)
    u.write(REPORT / 'manifest.json', dict(state='complete_original_own352_temporal_operators_statistics_and_ridge_pending_replay',
        source_sha256=bindings, original_rows=original, recent_rows=recent, rows=ROWS,
        raw_predictions=352, context_columns=90, design_columns=DESIGN, nonlinear_columns=FEATURES, nn_inputs=579,
        virtual_c_order_sha256=[h.hexdigest() for h in hashes],
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()}, ridge_rmse=score(prediction, labels),
        original_own330_ridge_rmse=u.read(u.ROOT / 'experiments/upstream-own330-native149-data/independent-verification.json')['ridge_rmse'],
        ridge_fold_rmse=[score(prediction[groups//2==f], labels[groups//2==f]) for f in range(5)],
        qualifying_labels_read=False, neural_models_fitted=0))
    print('Own352 complete known ridge:', score(prediction, labels), flush=True)


if __name__ == '__main__':
    main()
