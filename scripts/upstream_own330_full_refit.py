#!/usr/bin/env python3
"""Three full-known own330 refits with original selected budgets and streamed Q."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch
import upstream_own330_production_inputs as data
import upstream_exact_neural_scaling as scaling

u, recipe = data.u, data.recipe
RUN = u.ROOT / 'artifacts/upstream-own330-full-refit'
REPORT = u.ROOT / 'experiments/upstream-own330-full-refit'


def infer(query, declaration, weights, scaler):
    models = [recipe.mps.load_model(RUN / ('nn-%s.pt' % seed), recipe.CONFIG)
              for seed in declaration['seeds']]
    reader = data.Inputs(query, declaration['rows'])
    result = np.empty(data.COUNTS[query], dtype='<f4')
    try:
        for lo in range(0, len(result), 65536):
            hi = min(lo+65536, len(result))
            z, f = reader.block(lo, hi)
            baseline = np.clip(z @ weights, 1.05, 4.95)
            x = scaling.append_baseline(f, baseline)
            scaling.apply_scaler(x, scaler, output=x)
            correction = np.zeros(hi-lo, dtype=np.float64)
            for model in models:
                correction += recipe.mps.predict(model, x) / 3
            result[lo:hi] = np.clip(baseline+correction, 1.05, 4.95).astype('<f4')
    finally:
        reader.close()
        del models
        torch.mps.empty_cache()
    return result


def main():
    if RUN.exists() or REPORT.exists():
        raise ValueError('Existing full or partial own330 refit retained')
    declaration = u.read(data.REPORT / 'declaration.json')
    proof = u.read(data.REPORT / 'independent-verification.json')
    if (proof['state'] != 'complete_both_full_paired_own330_source_column_reconstructions'
            or proof['declaration_sha256'] != u.sha(data.REPORT / 'declaration.json')):
        raise ValueError('Complete independent full paired input replay required')
    bindings = dict(declaration['source_sha256'])
    budget_path = data.REPORT / 'original-budget-verification.json'
    budget = u.read(budget_path)
    if (budget['state'] != 'complete_full_refit_medians_bound_to_all_fifteen_actual_original_fits'
            or budget['full_iterations'] != declaration['full_iterations']
            or budget['seeds'] != declaration['seeds']):
        raise ValueError('Original independently verified actual fit budgets required')
    for key, digest in budget['source_sha256'].items():
        if key in bindings and bindings[key] != digest:
            raise ValueError('Original development budget binding conflicts')
        bindings[key] = digest
    for path in [Path(__file__), Path(scaling.__file__), data.REPORT / 'declaration.json',
                 data.REPORT / 'transport.json', data.REPORT / 'independent-verification.json', budget_path,
                 u.ROOT / 'scripts/verify_upstream_own330_full_refit.py',
                 u.ROOT / 'scripts/upstream_own330_full_refit_queue.py']:
        key, digest = str(path.relative_to(u.ROOT)), u.sha(path)
        if key in bindings and bindings[key] != digest:
            raise ValueError('Original source expectation changed: ' + key)
        bindings[key] = digest
    data.raw.check(bindings)
    if declaration['config'] != recipe.CONFIG or declaration['seeds'] != recipe.CONFIG['seeds']:
        raise ValueError('Actual original full recipe required')
    for name in ['labels.npy', 'users.npy']:
        if not np.array_equal(np.load(data.parent.RUN / name), np.load(recipe.RUN / name)):
            raise ValueError('Paired production and original development canonical row identity differs')
    RUN.mkdir(); REPORT.mkdir()
    u.write(REPORT / 'plan.json', dict(state='frozen_before_full_known_refit',
        declaration_sha256=u.sha(data.REPORT / 'declaration.json'), source_sha256=bindings,
        config=recipe.CONFIG, seeds=declaration['seeds'], full_iterations=declaration['full_iterations'],
        input_realization=declaration['input_realization'], qualifying_labels_read=False))
    recipe.mps.MODEL_FACTORY = recipe.model_factory
    labels = np.load(data.parent.RUN / 'labels.npy').astype(np.float64)
    if labels.shape != (1128164,):
        raise ValueError('Complete known population required')
    gram, rhs = np.zeros((406, 406)), np.zeros(406)
    reader = data.Inputs('probex', declaration['rows'])
    try:
        for lo in range(0, len(labels), 8192):
            hi = min(lo+8192, len(labels)); z, _ = reader.block(lo, hi)
            gram += z.T @ z; rhs += z.T @ labels[lo:hi]
        weights = recipe.n.ridge_weights(gram[None], rhs[None], np.array([True]), recipe.CONFIG['ridge_lambda'])
        np.savez(RUN / 'ridge-statistics.npz', gram=gram, rhs=rhs)
        np.save(RUN / 'ridge.npy', weights)
        features = np.empty((len(labels), 557), dtype=np.float32)
        baseline = np.empty(len(labels), dtype=np.float64)
        for lo in range(0, len(labels), 8192):
            hi = min(lo+8192, len(labels)); z, f = reader.block(lo, hi)
            baseline[lo:hi] = np.clip(z @ weights, 1.05, 4.95)
            features[lo:hi, :-1] = f; features[lo:hi, -1] = baseline[lo:hi]
    finally:
        reader.close()
    u.write(REPORT / 'status.json', dict(state='full_known_original_scaler'))
    scaler = scaling.fit_scaler(features, np.ones(len(labels), dtype=bool))
    np.savez(RUN / 'scaler.npz', **scaler)
    scaling.apply_scaler(features, scaler, output=features)
    digest = hashlib.sha256()
    for lo in range(0, len(labels), 8192):
        digest.update(features[lo:lo+8192].tobytes(order='C'))
    u.write(REPORT / 'preprocessing.json', dict(normalized_known_c_order_sha256=digest.hexdigest(), rows=len(labels), inputs=557))
    for seed, epochs in zip(declaration['seeds'], declaration['full_iterations']):
        u.write(REPORT / 'status.json', dict(state='original_full_known_refit', seed=seed, epochs=epochs))
        model, count = recipe.mps.train_nn(features, labels-baseline, recipe.CONFIG, seed, epochs)
        if count != epochs:
            raise ValueError('Full selected iteration budget changed')
        torch.save(dict(state=model.state_dict(), config=recipe.CONFIG, feature_width=557,
                        seed=seed, epochs=epochs), RUN / ('nn-%s.pt' % seed))
        del model; torch.mps.empty_cache()
    del features, labels, baseline; gc.collect(); torch.mps.empty_cache()
    for query in data.COUNTS:
        u.write(REPORT / 'status.json', dict(state='streamed_inference_and_saved_model_replay', query=query))
        path = RUN / (query + '.f32')
        infer(query, declaration, weights, scaler).tofile(path)
        replay = infer(query, declaration, weights, scaler)
        stored = np.fromfile(path, dtype='<f4')
        if not np.array_equal(replay.view('u4'), stored.view('u4')):
            raise ValueError('Actual saved model full query replay differs')
    data.raw.check(bindings)
    u.write(REPORT / 'fit-completion.json', dict(
        state='complete_original_own330_three_full_refits_and_internal_saved_model_replays',
        plan_sha256=u.sha(REPORT / 'plan.json'),
        outputs={p.name:u.sha(p) for p in RUN.iterdir()}, qualifying_labels_read=False,
        independent_preprocessing_and_model_verification_remaining=True))


if __name__ == '__main__':
    main()
