#!/usr/bin/env python3
"""Independent full known statistics/scaler and actual three-model Q replay."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch
import upstream_own330_full_refit as fit

u, data, recipe = fit.u, fit.data, fit.recipe


def main():
    report, run = fit.REPORT, fit.RUN
    if (report / 'independent-verification.json').exists():
        raise ValueError('Existing full independent production proof retained')
    plan = u.read(report / 'plan.json')
    complete = u.read(report / 'fit-completion.json')
    declaration = u.read(data.REPORT / 'declaration.json')
    if (complete['state'] != 'complete_original_own330_three_full_refits_and_internal_saved_model_replays'
            or complete['plan_sha256'] != u.sha(report / 'plan.json')
            or plan['declaration_sha256'] != u.sha(data.REPORT / 'declaration.json')):
        raise ValueError('Original complete three-seed full fit required')
    data.raw.check(plan['source_sha256'])
    expected = {'ridge.npy', 'ridge-statistics.npz', 'scaler.npz', 'probex.f32', 'qual.f32'}
    expected |= {'nn-%s.pt' % seed for seed in declaration['seeds']}
    if set(complete['outputs']) != expected or {p.name for p in run.iterdir()} != expected:
        raise ValueError('Complete exact saved production inventory required')
    for name, digest in complete['outputs'].items():
        if u.sha(run / name) != digest:
            raise ValueError('Actual saved production pipeline changed')
    labels = np.load(data.parent.RUN / 'labels.npy').astype(np.float64)
    reader = data.Inputs('probex', declaration['rows'])
    gram, rhs = np.zeros((406, 406)), np.zeros(406)
    try:
        for lo in range(0, len(labels), 8192):
            hi = min(lo+8192, len(labels)); z, _ = reader.block(lo, hi)
            gram += z.T @ z; rhs += z.T @ labels[lo:hi]
        with np.load(run / 'ridge-statistics.npz') as saved:
            if not np.array_equal(gram, saved['gram']) or not np.array_equal(rhs, saved['rhs']):
                raise ValueError('Full known original ridge statistics differ')
        penalty = np.eye(406) * recipe.CONFIG['ridge_lambda']; penalty[-1, -1] = 0
        weights = np.linalg.solve(gram+penalty, rhs)
        if not np.array_equal(weights, np.load(run / 'ridge.npy')):
            raise ValueError('Independent full known ridge solve differs')
        features = np.empty((len(labels), 557), dtype=np.float32)
        for lo in range(0, len(labels), 8192):
            hi = min(lo+8192, len(labels)); z, f = reader.block(lo, hi)
            features[lo:hi, :-1] = f
            features[lo:hi, -1] = np.clip(z @ weights, 1.05, 4.95)
    finally:
        reader.close()
    lo, hi = np.percentile(features, [1, 99], axis=0)
    clipped = np.clip(features, lo, hi).astype(np.float64)
    scaler = dict(lo=lo, hi=hi, mean=clipped.mean(0), std=np.maximum(clipped.std(0), 1e-6))
    del clipped; gc.collect()
    with np.load(run / 'scaler.npz') as saved:
        if set(saved.files) != set(scaler) or any(not np.array_equal(v, saved[k]) for k, v in scaler.items()):
            raise ValueError('Every original full known scaler statistic differs')
    normalized = hashlib.sha256()
    for lo in range(0, len(labels), 8192):
        hi = min(lo+8192, len(labels))
        reference = np.asarray((np.clip(features[lo:hi], scaler['lo'], scaler['hi'])-scaler['mean'])/scaler['std'], dtype=np.float32, order='C')
        normalized.update(reference.tobytes(order='C'))
    if normalized.hexdigest() != u.read(report / 'preprocessing.json')['normalized_known_c_order_sha256']:
        raise ValueError('Every normalized full known input bit must match')
    del features, labels; gc.collect()
    recipe.mps.MODEL_FACTORY = recipe.model_factory
    models = []
    for seed, epochs in zip(declaration['seeds'], declaration['full_iterations']):
        path = run / ('nn-%s.pt' % seed)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['seed'] != seed or saved['epochs'] != epochs or saved['feature_width'] != 557
                or saved['config'] != recipe.CONFIG):
            raise ValueError('Actual original full seed/iteration/architecture changed')
        models.append(recipe.mps.load_model(path, recipe.CONFIG))
    records = []
    for query, count in data.COUNTS.items():
        reader = data.Inputs(query, declaration['rows'])
        stored = np.fromfile(run / (query+'.f32'), dtype='<f4')
        if stored.shape != (count,) or not np.isfinite(stored).all():
            raise ValueError('Complete finite full query vector required')
        try:
            for lo in range(0, count, 65536):
                hi = min(lo+65536, count); z, f = reader.block(lo, hi)
                baseline = np.clip(z @ weights, 1.05, 4.95)
                x = np.empty((hi-lo, 557), dtype=np.float32)
                x[:, :-1] = f; x[:, -1] = baseline
                x = np.asarray((np.clip(x, scaler['lo'], scaler['hi'])-scaler['mean'])/scaler['std'], dtype=np.float32, order='C')
                correction = np.zeros(hi-lo, dtype=np.float64)
                for model in models:
                    correction += recipe.mps.predict(model, x) / 3
                replay = np.clip(baseline+correction, 1.05, 4.95).astype('<f4')
                if not np.array_equal(replay.view('u4'), stored[lo:hi].view('u4')):
                    raise ValueError('Every actual saved model query prediction must match')
        finally:
            reader.close()
        records.append(dict(query=query, rows=count, actual_models_reopened=3,
            all_prediction_bits_equal=True, sha256=u.sha(run / (query+'.f32'))))
    del models; torch.mps.empty_cache()
    data.raw.check(plan['source_sha256'])
    for name, digest in complete['outputs'].items():
        if u.sha(run / name) != digest:
            raise ValueError('Actual production pipeline changed during verification')
    u.write(report / 'independent-verification.json', dict(
        state='complete_original_own330_full_known_preprocessing_and_both_actual_model_queries',
        plan_sha256=u.sha(report / 'plan.json'), completion_sha256=u.sha(report / 'fit-completion.json'),
        verifier_sha256=u.sha(Path(__file__)), outputs=complete['outputs'], records=records,
        complete_known_statistics_scaler_normalized_bits_replayed=True,
        full_iterations=declaration['full_iterations'], qualifying_labels_read=False,
        qualifying_score_claimed=False, goal_complete=False))


if __name__ == '__main__':
    main()
