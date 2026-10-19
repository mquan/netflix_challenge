#!/usr/bin/env python3
"""LC5: target-movie relevance on fixed local context, with uniform/shuffle controls."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from local_structure_train import fit_preprocessing, read_inputs, rmse
from round9_latent import read_model
from train_holdout import digest


def shuffle_relevance(similarity, valid, seed=20261019):
    """Shuffle nonpadding similarities, independently of all rating values."""
    rng = np.random.default_rng(seed)
    order = np.argsort(np.where(valid, rng.random(valid.shape), 2), axis=1, kind='stable')
    destination = np.argsort(~valid, axis=1, kind='stable')
    result = np.zeros_like(similarity)
    np.put_along_axis(result, destination, np.take_along_axis(similarity, order, axis=1), axis=1)
    return result


def local_features(context, similarity, uniform=False):
    cross = context['cross_rating'].astype(float)
    mask = cross > 0
    history = context['history_rating'].astype(float)
    target = context['neighbor_rating'].astype(float)
    weight = np.ones_like(similarity) if uniform else np.exp(4 * similarity)
    edge_weight = weight[:, None, :] * mask
    total = edge_weight.sum(2)
    norm = edge_weight / np.maximum(total[:, :, None], 1e-12)
    difference = history[:, None, :] - cross
    offset = (norm * difference).sum(2)
    disagreement = (norm * difference ** 2).sum(2)
    agreement = (norm * (difference == 0)).sum(2)
    support = total ** 2 / np.maximum((edge_weight ** 2).sum(2), 1e-12)
    reliability = support / (support + 5) * np.exp(-disagreement / 2)
    mass = reliability.sum(1)
    neighbor_weight = reliability / np.maximum(mass[:, None], 1e-12)
    average = lambda value: (neighbor_weight * value).sum(1)
    predicted = target + offset
    prediction = average(predicted)
    present = total > 0
    count = np.maximum(present.sum(1), 1)
    offset_mean = (offset * present).sum(1) / count
    cross_age = (norm * np.abs(context['cross_gap'].astype(float) / 365)).sum(2)
    knn = np.where(mass > 0, np.clip(prediction, 1, 5), context['pooled'][:, 0])
    return np.column_stack([
        knn, average(target), average(offset), average(disagreement), average(agreement),
        np.log1p(mass), mass ** 2 / np.maximum((reliability ** 2).sum(1), 1e-12),
        np.sqrt(average((predicted - prediction[:, None]) ** 2)), offset_mean,
        np.sqrt(((offset - offset_mean[:, None]) ** 2 * present).sum(1) / count),
        (disagreement * present).sum(1) / count, (support * present).sum(1) / count,
        average(cross_age), average(np.abs(context['neighbor_gap'].astype(float) / 365))
    ]).astype('f4')


def export_geometry(arrays, spec_path, context_path, output):
    spec = json.loads(Path(spec_path).read_text())
    provenance_path = Path('experiments/P0-A-K1.json')
    provenance = json.loads(provenance_path.read_text())
    model_path = provenance['model_artifact']['path']
    assert digest(model_path) == provenance['model_artifact']['sha256']
    assert list(provenance['inputs'].values())[0]['sha256'] == spec['data_sha256']
    d = load(spec['data'])
    model, info = read_model(model_path, d['users'], d['items'])
    factors = model['item_factors'].astype(float)
    factors -= factors.mean(0)
    factors /= np.maximum(np.linalg.norm(factors, axis=1, keepdims=True), 1e-12)
    queries = np.load(Path(context_path) / 'queries.npy')
    history_items = np.zeros((len(queries), 32), dtype=np.int32)
    targets = d['probe']['item'][queries].astype(np.int32)
    for k, q in enumerate(queries):
        row = d['probe'][q]
        u, day = int(row['user']), int(row['day'])
        lo, hi = d['train_offsets'][u:u + 2]
        rows = np.arange(int(lo), int(hi))
        gap = np.abs(d['train_days'][rows].astype(np.int32) - day)
        rows = rows[np.argsort(gap, kind='stable')[:32]]
        history_items[k, :len(rows)] = d['train_items'][rows]
        assert not np.any(history_items[k] == targets[k])
        np.testing.assert_array_equal(d['train_ratings'][rows], arrays['history_rating'][k, :len(rows)])
        np.testing.assert_array_equal(d['train_days'][rows].astype(int) - day, arrays['history_gap'][k, :len(rows)])
        if (k + 1) % 50000 == 0:
            print(json.dumps(dict(phase='history_alignment', rows=k + 1)), flush=True)
    valid = history_items > 0
    similarity = np.zeros(history_items.shape, dtype='f4')
    for start in range(0, len(queries), 512):
        sl = slice(start, start + 512)
        product = factors[np.maximum(history_items[sl] - 1, 0)] * factors[targets[sl] - 1, None, :]
        similarity[sl] = np.clip(product.sum(2), -1, 1) * valid[sl]
    shuffled = shuffle_relevance(similarity, valid)
    np.testing.assert_array_equal(np.sort(similarity, axis=1), np.sort(shuffled, axis=1))
    count = np.maximum(valid.sum(1), 1)
    mean = similarity.sum(1) / count
    common = np.column_stack([
        mean, np.sqrt((((similarity - mean[:, None]) ** 2) * valid).sum(1) / count),
        np.where(valid.any(1), np.where(valid, similarity, np.inf).min(1), 0),
        np.where(valid.any(1), np.where(valid, similarity, -np.inf).max(1), 0)
    ]).astype('f4')
    features = {arm: np.empty((len(queries), 14), dtype='f4')
                for arm in ('uniform', 'relevant', 'shuffled')}
    for start in range(0, len(queries), 512):
        sl = slice(start, start + 512)
        context = {key: np.asarray(value[sl]) for key, value in arrays.items()}
        for arm in features:
            values = shuffled[sl] if arm == 'shuffled' else similarity[sl]
            features[arm][sl] = local_features(context, values, uniform=(arm == 'uniform'))
    for name, values in dict(history_items=history_items, similarity=similarity,
                             shuffled_similarity=shuffled, geometry=common, **features).items():
        assert np.isfinite(values).all(), name
        np.save(output / (name + '.npy'), values)
    return common, features, dict(model_path=model_path, model_sha256=digest(model_path),
        model_info=info, provenance_sha256=digest(provenance_path),
        changed_similarity_fraction=float(np.mean(similarity[valid] != shuffled[valid])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', required=True)
    a = parser.parse_args()
    output = Path(a.output)
    if Path(a.summary).exists():
        parser.error('summary already exists')
    output.mkdir(parents=True, exist_ok=False)
    spec, context = 'experiments/LC2-spec.json', 'artifacts/local-context/LC2/context'
    arrays, columns, raw, y, users, _ = read_inputs(spec, context)
    geometry, features, provenance = export_geometry(arrays, spec, context, output)
    folds = blend_fold(users)
    fit, select, held, refit = [np.flatnonzero(mask) for mask in
                              (folds >= 2, folds == 1, folds == 0, folds != 0)]
    _, select_base, _ = fit_preprocessing(columns, raw, y, fit)
    _, refit_base, _ = fit_preprocessing(columns, raw, y, refit)
    params = dict(objective='regression', metric='None', learning_rate=.03,
                  num_leaves=15, min_data_in_leaf=300, lambda_l2=10, max_bin=127,
                  feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
                  num_threads=2, verbosity=-1, deterministic=True, force_col_wise=True)
    all_predictions, records = {}, {}
    print(json.dumps(dict(phase='features_ready', provenance=provenance, params=params)), flush=True)
    for arm in ('uniform', 'relevant', 'shuffled'):
        x = np.column_stack([raw, geometry, features[arm]]).astype('f4')
        all_predictions[arm], records[arm] = [], []
        for seed in (2026, 7):
            cfg = dict(params, seed=seed, feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed)
            def metric(prediction, dataset):
                return 'clipped_rmse', rmse(y[select], select_base[select] + prediction), False
            model = lgb.train(cfg, lgb.Dataset(x[fit], label=y[fit] - select_base[fit]),
                num_boost_round=2000,
                valid_sets=[lgb.Dataset(x[select], label=y[select] - select_base[select])],
                feval=metric, callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds = model.best_iteration
            score = rmse(y[select], select_base[select] + model.predict(x[select], num_threads=2))
            if score >= rmse(y[select], select_base[select]):
                rounds = 0
            if rounds:
                model = lgb.train(cfg, lgb.Dataset(x[refit], label=y[refit] - refit_base[refit]),
                                  num_boost_round=rounds)
                value = refit_base[held].astype(float) + model.predict(x[held], num_threads=2)
                model.save_model(str(output / f'{arm}-seed{seed}.txt'))
            else:
                value = refit_base[held].astype(float)
            value.astype('<f4').tofile(output / f'{arm}-seed{seed}.f32')
            all_predictions[arm].append(value)
            records[arm].append(dict(seed=seed, rounds=rounds, selection_rmse=score))
            print(json.dumps(dict(phase='arm_complete', arm=arm, **records[arm][-1])), flush=True)
    average = {arm: np.mean(values, axis=0) for arm, values in all_predictions.items()}
    comparisons = {arm: paired(y[held], average[arm], average['relevant'], users[held])
                   for arm in ('uniform', 'shuffled')}
    per_seed = {str(seed): paired(y[held], all_predictions['uniform'][i],
                                 all_predictions['relevant'][i], users[held])
                for i, seed in enumerate((2026, 7))}
    passed = (comparisons['uniform']['delta_rmse'] <= -.0001 and
              comparisons['uniform']['cluster_normal_95_mse_interval'][1] < 0 and
              comparisons['shuffled']['delta_rmse'] < 0 and
              all(item['delta_rmse'] < 0 for item in per_seed.values()))
    for arm, value in average.items():
        value.astype('<f4').tofile(output / (arm + '.f32'))
    np.save(output / 'held_indices.npy', held)
    report = dict(config=vars(a), params=params, provenance=provenance, training=records,
        spec_sha256=digest(spec), context_manifest_sha256=digest(Path(context) / 'manifest.json'),
        rmse={arm: rmse(y[held], value) for arm, value in average.items()},
        comparisons=comparisons, per_seed=per_seed, confirmation_gate_passed=passed,
        outputs={p.name: digest(p) for p in output.iterdir()},
        limitation='Reused P2 development rows. Positive results need fresh confirmation; no ensemble claim.')
    Path(a.summary).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('rmse', 'comparisons', 'confirmation_gate_passed')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
