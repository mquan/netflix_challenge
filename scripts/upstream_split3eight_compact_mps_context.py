#!/usr/bin/env python3
"""Same complete split3-eight neural recipe with exact cached inputs and bounded temporaries."""
import argparse
from pathlib import Path
import time
import numpy as np
import upstream_split3eight_virtual_mps_context as recipe
import upstream_exact_neural_scaling as scaling
import upstream_split3eight_scaling_parity as parity

u, n = recipe.u, recipe.n
RUN = u.ROOT / 'artifacts/upstream-split3eight-compact-mps-context'
REPORT = u.ROOT / 'experiments/upstream-split3eight-compact-mps-context'
CONFIG = dict(recipe.CONFIG)
ORIGINAL_CODE_HASHES = recipe.code_hashes


def code_hashes():
    return {**ORIGINAL_CODE_HASHES(), **{str(p.relative_to(u.ROOT)): u.sha(p)
            for p in [Path(__file__), Path(scaling.__file__), Path(parity.__file__)]}}


def configure():
    recipe.RUN, recipe.REPORT, recipe.code_hashes = RUN, REPORT, code_hashes
    recipe.configure()


def verified_cache():
    proof_path = parity.REPORT / 'verification.json'
    proof = u.read(proof_path)
    if (proof['state'] != 'complete_real_data_bitwise_input_and_scaler_parity'
            or proof['rows'] != 1128164 or proof['columns'] != 406
            or proof['models_fitted'] != 0 or proof['qualifying_labels_read'] is not False
            or len(proof['records']) != 2
            or not all(r['original_complete_statistics_bitwise_equal'] and
                       r['every_actual_normalized_float32_bit_equal'] for r in proof['records'])):
        raise ValueError('Complete real-row exact transform proof required')
    cp = parity.REPORT / 'cache-verification.json'
    cache = u.read(cp)
    original_plan = u.read(parity.REPORT / 'plan.json')
    for name, digest in original_plan['bindings'].items():
        if u.sha(u.ROOT / name) != digest:
            raise ValueError('Reference parity implementation or input changed')
    if proof['cache_verification_sha256'] != u.sha(cp) or not cache['every_float32_input_bit_equal']:
        raise ValueError('Complete original virtual input proof differs')
    bindings = {str(p.relative_to(u.ROOT)): u.sha(p) for p in [proof_path, cp, parity.REPORT / 'plan.json']}
    for name, digest in cache['files_sha256'].items():
        p = parity.RUN / name
        if u.sha(p) != digest:
            raise ValueError('Exact complete input cache changed')
        bindings[str(p.relative_to(u.ROOT))] = digest
    return bindings


def prepare():
    configure()
    if (REPORT / 'manifest.json').exists():
        return n.verify_prepared()
    bindings = verified_cache()
    m = recipe.prepare()
    m['source_manifests_sha256'].update(bindings)
    m['memory_execution'] = dict(original_model_config_unchanged=True,
            original_rows_and_folds_unchanged=True, original_normalization_bits_verified=True,
            cached_complete_design_columns=191, cached_complete_raw_input_columns=406,
            policy='Read original complete matrices from exact caches. Assemble exact float32 ridge inputs without float64 stack copies; unmap source caches before scaling/training. Original eligible statistics/order; private in-place variance buffer and blockwise normalization only. Same MPS model/optimizer/seeds/stopping/batches; no approximation.')
    m['scope'] += ' Exact compact-buffer execution: no numerical recipe, data, capacity, seed or stopping change.'
    u.write(REPORT / 'manifest.json', m)
    return n.verify_prepared()


def fit_fold(y, groups, grams, rhs, fold, output):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    training, held = groups // 2 != fold, groups // 2 == fold
    fit, valid = training & (groups % 2 == 0), training & (groups % 2 == 1)
    if not all(np.any(v) for v in [training, held, fit, valid]):
        raise ValueError('Empty original outer/inner user partition')
    gi = np.arange(10);eligible = gi // 2 != fold
    iw = n.ridge_weights(grams, rhs, eligible & (gi % 2 == 0), CONFIG['ridge_lambda'])
    fw = n.ridge_weights(grams, rhs, eligible, CONFIG['ridge_lambda'])
    z = np.load(parity.RUN / 'design.npy', mmap_mode='r')
    inner_base, base = np.clip(z @ iw, 1.05, 4.95), np.clip(z @ fw, 1.05, 4.95)
    del z
    np.savez(output / 'ridge.npz', inner=iw, full=fw)
    cached = np.load(parity.RUN / 'inputs.npy', mmap_mode='r')
    inner_x = scaling.append_baseline(cached[:, :405], inner_base)
    full_x = scaling.append_baseline(cached[:, :405], base)
    del cached  # No source matrix remains mapped during fitting/scaling.
    inner_x, inner_scaler = scaling.scaled(inner_x, fit)
    full_x, full_scaler = scaling.scaled(full_x, training)
    np.savez(output / 'scalers.npz', **{'inner_' + k: v for k, v in inner_scaler.items()},
             **{'full_' + k: v for k, v in full_scaler.items()})
    correction = np.zeros(int(held.sum()), dtype=np.float64);chosen = []
    for seed in CONFIG['seeds']:
        actual_seed = seed + fold
        n.event(stage='select', family='nn', fold=fold, seed=seed)
        _, count = n.train_nn(inner_x[fit], y[fit] - inner_base[fit], CONFIG, actual_seed,
                CONFIG['nn_epochs'], valid=(inner_x[valid], y[valid] - inner_base[valid]))
        if count:
            import torch
            model, _ = n.train_nn(full_x[training], y[training] - base[training], CONFIG, actual_seed, count)
            correction += n.nn_predict(model, full_x[held]) / len(CONFIG['seeds'])
            torch.save(dict(state=model.state_dict(), config=CONFIG, feature_width=full_x.shape[1],
                       seed=actual_seed, epochs=count), output / f'nn-{seed}.pt')
            del model
        chosen.append(count)
        n.event(stage='refit_complete', family='nn', fold=fold, seed=seed, iterations=count)
    result = np.clip(base[held] + correction, 1.05, 4.95).astype('<f4')
    result.tofile(output / 'prediction.f32')
    return dict(family='nn', fold=fold, chosen_iterations=chosen, rows=int(held.sum()),
                rmse=n.x.metric(result, y[held]), baseline_rmse=n.x.metric(base[held], y[held]))


def worker():
    configure();n.verify_prepared();verified_cache()
    y = np.load(RUN / 'labels.npy');groups = np.load(RUN / 'groups.npy')
    with np.load(RUN / 'stats.npz') as saved:
        grams, rhs = saved['gram'], saved['rhs']
    for fold in range(5):
        path = REPORT / 'nn' / f'{fold}.json'
        if path.exists():
            raise ValueError('Existing full fold evidence preserved; no automatic restart')
        start = time.monotonic();out = RUN / 'nn' / str(fold)
        record = fit_fold(y, groups, grams, rhs, fold, out)
        record.update(seconds=time.monotonic() - start, manifest_sha256=u.sha(REPORT / 'manifest.json'),
                      outputs={p.name: u.sha(p) for p in out.iterdir() if p.is_file()})
        u.write(path, record);n.event(stage='fold_complete', **record)


def replay():
    configure();n.verify_prepared();verified_cache()
    groups = np.load(RUN / 'groups.npy');records = []
    for fold in range(5):
        root = RUN / 'nn' / str(fold);receipt = u.read(REPORT / 'nn' / f'{fold}.json')
        if receipt['manifest_sha256'] != u.sha(REPORT / 'manifest.json'):
            raise ValueError('Actual original fold manifest differs')
        for name, digest in receipt['outputs'].items():
            if u.sha(root / name) != digest:
                raise ValueError('Actual saved fold artifact changed')
        held = groups // 2 == fold
        z = np.load(parity.RUN / 'design.npy', mmap_mode='r')
        with np.load(root / 'ridge.npz') as weights:
            baseline = np.clip(z @ weights['full'], 1.05, 4.95)
        del z
        cached = np.load(parity.RUN / 'inputs.npy', mmap_mode='r')
        inputs = scaling.append_baseline(cached[held, :405], baseline[held]);del cached
        with np.load(root / 'scalers.npz') as saved:
            scaler = {key: saved['full_' + key] for key in ['lo', 'hi', 'mean', 'std']}
        # Reference normalization, rather than the new block implementation,
        # independently checks every actual held-row prediction path.
        inputs, _ = n.scaled(inputs, None, scaler=scaler)
        correction = np.zeros(int(held.sum()))
        for seed, count in zip(CONFIG['seeds'], receipt['chosen_iterations']):
            if count:
                fitted = recipe.neural_model(root / f'nn-{seed}.pt')
                correction += n.nn_predict(fitted, inputs) / len(CONFIG['seeds']);del fitted
        actual = np.clip(baseline[held] + correction, 1.05, 4.95).astype('<f4')
        expected = np.fromfile(root / 'prediction.f32', dtype='<f4')
        if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
            raise ValueError('Every actual saved model prediction must replay bitwise')
        records.append(dict(fold=fold, rows=len(actual), bitwise_equal=True))
    u.write(REPORT / 'nn-replay.json', dict(family='nn', records=records,
            verifier_sha256=u.sha(Path(__file__)), original_reference_normalization_used=True,
            complete_actual_cached_source_fingerprints_verified=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--worker', action='store_true');p.add_argument('--replay', action='store_true');a = p.parse_args()
    if a.worker:worker()
    elif a.replay:replay()
    else:prepare()
