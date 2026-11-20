#!/usr/bin/env python3
"""Separate complete input/operator/actual15-model qualifying bag replay."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import upstream_own352_fold_bag as production

u, study = production.u, production.study


def main():
    report, root = production.REPORT, production.RUN
    output = report/'independent-verification.json'
    if output.exists():
        raise ValueError('Existing independent complete fold bag replay retained')
    plan = u.read(report/'plan.json'); complete = u.read(report/'inference-completion.json')
    if (complete['state'] != 'complete_all_fifteen_original_models_full_qualifying_inference_pending_independent_replay'
            or complete['plan_sha256'] != u.sha(report/'plan.json')
            or complete['input_verification_sha256'] != u.sha(report/'input-verification.json')
            or plan['folds'] != list(range(5)) or plan['seeds'] != study.CONFIG['seeds']
            or plan['rows'] != 2817131 or plan['chunk_rows'] != 65536
            or plan['config'] != study.CONFIG or plan['qualifying_labels_read'] is not False):
        raise ValueError('Original complete allfold inference and frozen numerical recipe required')
    sources, original, added = production.sources()
    if sources != plan['source_sha256'] or original != plan['original_rows'] or added != plan['added_rows']:
        raise ValueError('Original source expectations or complete input population differ')
    expected = {'fold-%s.f32'%f for f in range(5)}|{'nn-bag.f32'}
    if set(complete['outputs']) != expected or {p.name for p in root.iterdir()} != expected:
        raise ValueError('Exactly all five original fold predictions and bag required')
    for name, value in complete['outputs'].items():
        if u.sha(root/name) != value:
            raise ValueError('Original qualifying fold/bag vector changed')
    study.mps.MODEL_FACTORY = study.model_factory
    records = []; hashes = [hashlib.sha256(), hashlib.sha256()]
    for fold in range(5):
        trained = study.RUN/'nn'/str(fold)
        receipt = u.read(study.REPORT/'nn'/('%s.json'%fold))
        with np.load(trained/'ridge.npz') as saved:
            weights = saved['full'].copy()
        with np.load(trained/'scalers.npz') as saved:
            scaler = {key:saved['full_'+key].copy() for key in ['lo','hi','mean','std']}
        models = []
        for seed, count in zip(study.CONFIG['seeds'], receipt['chosen_iterations']):
            path = trained/('nn-%s.pt'%seed)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if (checkpoint['config'] != study.CONFIG or checkpoint['seed'] != seed+fold
                    or checkpoint['epochs'] != count or checkpoint['feature_width'] != 579):
                raise ValueError('Every actual original complete model metadata must match')
            models.append(study.mps.load_model(path, study.CONFIG))
        inputs = production.Inputs(original, added)
        stored = np.fromfile(root/('fold-%s.f32'%fold), dtype='<f4')
        if stored.shape != (production.ROWS,) or not np.isfinite(stored).all():
            raise ValueError('Complete finite original fold vector required')
        try:
            for lo in range(0, production.ROWS, 65536):
                hi = min(lo+65536, production.ROWS); z, f = inputs.block(lo, hi)
                if fold == 0:
                    oldz, oldf = inputs.parent.block(lo, hi)
                    expected_z = np.empty((hi-lo, 428), dtype=np.float64)
                    expected_f = np.empty((hi-lo, 578), dtype=np.float32)
                    expected_z[:, :405] = oldz[:, :405]; expected_z[:, -1] = oldz[:, -1]
                    expected_f[:, :556] = oldf
                    for j, row in enumerate(added):
                        reader = production.base.raw.Rows(u.ROOT/row['paired_files']['qual'], (production.ROWS,), np.float32)
                        try:
                            value = reader.read(lo, hi)
                        finally:
                            reader.close()
                        value[value < 0] = 0; value[value > 6] = 6
                        expected_z[:, 405+j] = value
                        expected_f[:, 556+j] = value-np.float32(3.6)
                    if (not np.array_equal(z.view('u8'), expected_z.view('u8'))
                            or not np.array_equal(f.view('u4'), expected_f.view('u4'))):
                        raise ValueError('Every original retained and added qualifying operator bit must match')
                    hashes[0].update(expected_z.tobytes(order='C')); hashes[1].update(expected_f.tobytes(order='C'))
                baseline = np.clip(z@weights, 1.05, 4.95)
                x = np.empty((hi-lo, 579), dtype=np.float32)
                x[:, :-1] = f; x[:, -1] = baseline
                x = np.asarray((np.clip(x, scaler['lo'], scaler['hi'])-scaler['mean'])/scaler['std'], dtype=np.float32, order='C')
                correction = np.zeros(hi-lo, dtype=np.float64)
                for model in models:
                    correction += study.mps.predict(model, x)/3
                reference = np.clip(baseline+correction, 1.05, 4.95).astype('<f4')
                if not np.array_equal(reference.view('u4'), stored[lo:hi].view('u4')):
                    raise ValueError('Every actual original fold-model qualifying prediction bit must replay')
        finally:
            inputs.close(); del models; torch.mps.empty_cache()
        records.append(dict(fold=fold, rows=production.ROWS, actual_models_reopened=3,
            every_qualifying_prediction_bit_equal=True, prediction_sha256=u.sha(root/('fold-%s.f32'%fold))))
        u.write(report/'replay-status.json', dict(state='complete_actual_original_fold_query_replay', fold=fold))
    if [h.hexdigest() for h in hashes] != u.read(report/'input-verification.json')['c_order_sha256']:
        raise ValueError('Complete independently reconstructed qualifying input stream differs')
    bag = np.fromfile(root/'nn-bag.f32', dtype='<f4')
    folds = [np.fromfile(root/('fold-%s.f32'%fold), dtype='<f4') for fold in range(5)]
    for lo in range(0, production.ROWS, 8192):
        hi = min(lo+8192, production.ROWS); expected = np.zeros(hi-lo, dtype=np.float64)
        for values in folds:
            expected = expected+values[lo:hi].astype(np.float64)
        if not np.array_equal((expected/5).astype('<f4').view('u4'), bag[lo:hi].view('u4')):
            raise ValueError('Every original uniform fold-bag float32 bit must match')
    study.data.check(sources)
    for name, value in complete['outputs'].items():
        if u.sha(root/name) != value:
            raise ValueError('Original qualifying vectors changed during replay')
    u.write(output, dict(state='complete_independent_all15_original_own352_fold_qualifying_models_and_uniform_bag',
        plan_sha256=u.sha(report/'plan.json'), inference_completion_sha256=u.sha(report/'inference-completion.json'),
        verifier_sha256=u.sha(Path(__file__)), records=records, original_models_reopened=15,
        all428_578_input_operators_independently_reconstructed=True, uniform_five_fold_bag_bitwise_equal=True,
        prediction_sha256=u.sha(root/'nn-bag.f32'), source_sha256=sources,
        qualifying_labels_read=False, models_fitted=0, qualifying_score_claimed=False))
    print('All15 original models and complete qualifying bag independently replayed', flush=True)


if __name__ == '__main__':
    main()
