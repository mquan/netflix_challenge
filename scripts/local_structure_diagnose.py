#!/usr/bin/env python3
"""Post-hoc LC2 mechanism audit. No model selection or promotion scores.

Cache the head inputs from frozen checkpoints, replace local representations
by their refit-user mean, and switch aggregation with fixed weights. These are
inference interventions, not estimates of retrained-model performance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gbdt_blend import blend_fold
from local_structure_train import NeighborEncoder, batch, read_inputs, rmse
from train_holdout import digest


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def describe(x):
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                    map(float, np.quantile(x, [0, .1, .5, .9, 1]))))


def capture(model, arrays, features, baseline, rows, a, shuffled=False):
    chunks, predictions = [], []
    hook = model.head.register_forward_pre_hook(
        lambda module, inputs: chunks.append(inputs[0][:, features.shape[1]:].detach().cpu().numpy()))
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), a.batch):
                ix = rows[start:start + a.batch]
                p = model(batch(arrays, ix, a.device, shuffled),
                          torch.from_numpy(features[ix]).to(a.device),
                          torch.from_numpy(baseline[ix]).to(a.device))
                predictions.append(p.cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(chunks), np.concatenate(predictions)


def head_predict(model, features, local, baseline, a):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), a.batch):
            sl = slice(start, start + a.batch)
            x = np.column_stack([features[sl], local[sl]]).astype('f4')
            correction = model.head(torch.from_numpy(x).to(a.device)).squeeze(1).cpu().numpy()
            values.append(baseline[sl] + correction)
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'])
    parser.add_argument('--batch', type=int, default=256)
    a = parser.parse_args()
    if Path(a.output).exists():
        parser.error('output already exists')
    torch.set_num_threads(2)
    pilot = Path('artifacts/local-context/LC2/pilot')
    original = json.loads((pilot / 'report.json').read_text())
    arrays, columns, raw, y, users, context_report = read_inputs(
        original['config']['spec'], original['config']['context'])
    prep = np.load(pilot / 'preprocessing.npz')
    features = np.clip((raw - prep['mean']) / prep['scale'], -8, 8).astype('f4')
    design = np.column_stack([np.ones(len(y)), columns - 3.6])
    baseline = (3.6 + design @ prep['weights']).astype('f4')
    held = np.load(pilot / 'held_indices.npy')
    fit = np.flatnonzero(blend_fold(users) != 0)
    # Fixed label-independent refit sample keeps the audit small and repeatable.
    fit = np.sort(np.random.default_rng(2718).choice(fit, 8192, replace=False))
    rows = np.r_[fit, held]
    nf = len(fit)
    report = dict(kind='Post-hoc mechanism diagnosis; no promotion or tuning',
                  config=vars(a), fit_sample=len(fit), evaluation_rows=len(held),
                  original_report_sha256=digest(pilot / 'report.json'),
                  script_sha256=digest(__file__), models={})
    mask = np.asarray(arrays['cross_rating'][rows]) > 0
    report['context'] = dict(edges=describe(mask.sum((1, 2))),
                             neighbors=describe(mask.any(2).sum(1)),
                             history=describe((arrays['history_rating'][rows] > 0).sum(1)),
                             empty_fraction=float(np.mean(~mask.any((1, 2)))))
    for arm in ('structured', 'pooled'):
        for seed in (2026, 7):
            torch.manual_seed(seed)
            initial = NeighborEncoder(features.shape[1], 16, pooled=(arm == 'pooled')).to(a.device)
            initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
            initial_local, _ = capture(initial, arrays, features, baseline, fit, a)
            model = NeighborEncoder(features.shape[1], 16, pooled=(arm == 'pooled')).to(a.device)
            path = pilot / f'{arm}-seed{seed}.pt'
            assert digest(path) == original['outputs'][path.name]
            model.load_state_dict(torch.load(path, map_location=a.device, weights_only=True))
            local, prediction = capture(model, arrays, features, baseline, rows, a)
            saved = np.fromfile(pilot / f'{arm}-seed{seed}.f32', dtype='<f4')
            replay_error = float(np.max(np.abs(prediction[nf:] - saved)))
            assert replay_error < 1e-5, replay_error
            cached = head_predict(model, features[rows], local, baseline[rows], a)
            assert np.max(np.abs(cached - prediction)) < 1e-5
            mean_local = np.broadcast_to(local[:nf].mean(0), local.shape).copy()
            constant = head_predict(model, features[rows], mean_local, baseline[rows], a)
            perm = np.r_[np.random.default_rng(17).permutation(nf),
                         nf + np.random.default_rng(17).permutation(len(held))]
            permuted = head_predict(model, features[rows], local[perm], baseline[rows], a)
            fixed_global = np.broadcast_to(features[fit].mean(0), features[rows].shape).copy()
            constant_global = head_predict(model, fixed_global, local, baseline[rows], a)
            w = model.head[0].weight.detach().cpu().numpy()
            local_affine = (local[:nf] - local[:nf].mean(0)) @ w[:, features.shape[1]:].T
            global_affine = (features[fit] - features[fit].mean(0)) @ w[:, :features.shape[1]].T
            result = dict(checkpoint_sha256=digest(path), replay_max_abs_error=replay_error,
                          initial_local_std=describe(initial_local.std(0)),
                          trained_local_std=describe(local[:nf].std(0)),
                          standardized_global_std=describe(features[fit].std(0)),
                          local_affine_rms=rms(local_affine), global_affine_rms=rms(global_affine),
                          local_to_global_affine_rms=rms(local_affine) / rms(global_affine),
                          initial_to_trained_local_rms=rms(local[:nf] - initial_local),
                          weight_relative_change={}, partitions={})
            centered = local[:nf].astype(float) - local[:nf].mean(0, dtype=float)
            eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2
            result['local_variance_top_components'] = {
                str(k): float(eigenvalues[:k].sum() / eigenvalues.sum()) for k in (1, 2, 4, 8)}
            # Label-free redundancy probe. Fixed regularization; fit only refit
            # users and predict the frozen representation on evaluation users.
            # The 27 pooled columns follow 11 base columns and one support.
            redundant = {}
            for feature_name, feature_slice in [('common_pooled', slice(12, None)),
                                                 ('all_global', slice(None))]:
                x = features[rows, feature_slice].astype(float)
                x = np.column_stack([np.ones(len(x)), x])
                penalty = np.eye(x.shape[1]) * nf * 1e-4
                penalty[0, 0] = 0
                weights = np.linalg.solve(x[:nf].T @ x[:nf] + penalty, x[:nf].T @ local[:nf])
                reconstructed = (x @ weights).astype('f4')
                reconstruction_prediction = head_predict(model, features[rows], reconstructed, baseline[rows], a)
                redundant[feature_name] = dict(
                    evaluation_local_variance_explained=1 - rms(local[nf:] - reconstructed[nf:]) ** 2 /
                    rms(local[nf:] - local[:nf].mean(0)) ** 2,
                    evaluation_prediction_change_rms=rms(reconstruction_prediction[nf:] - prediction[nf:]),
                    evaluation_reconstructed_rmse=rmse(y[held], reconstruction_prediction[nf:]))
            result['label_free_redundancy_probe'] = redundant
            state = model.state_dict()
            for prefix in ('edge', 'relation', 'neighbor', 'head'):
                before = np.concatenate([v.numpy().ravel() for k, v in initial_state.items() if k.startswith(prefix)])
                after = np.concatenate([v.detach().cpu().numpy().ravel() for k, v in state.items() if k.startswith(prefix)])
                result['weight_relative_change'][prefix] = rms(after - before) / rms(before)
            if arm == 'structured':
                model.pooled = True
                pooled_local, pooled_prediction = capture(model, arrays, features, baseline, rows, a)
                model.pooled = False
                shuffled_local, shuffled_prediction = capture(model, arrays, features, baseline, rows, a, True)
                result['fixed_weight_pooling_local_change_rms'] = rms(pooled_local - local)
                result['fixed_weight_shuffle_local_change_rms'] = rms(shuffled_local - local)
            for name, sl in [('refit_sample', slice(0, nf)), ('evaluation', slice(nf, None))]:
                actual_y = y[rows[sl]]
                p = prediction[sl]
                stats = dict(baseline_rmse=rmse(actual_y, baseline[rows[sl]]), model_rmse=rmse(actual_y, p),
                             constant_local_rmse=rmse(actual_y, constant[sl]),
                             permuted_local_rmse=rmse(actual_y, permuted[sl]),
                             constant_global_rmse=rmse(actual_y, constant_global[sl]),
                             correction_rms=rms(p - baseline[rows[sl]]),
                             constant_local_prediction_change_rms=rms(constant[sl] - p),
                             permuted_local_prediction_change_rms=rms(permuted[sl] - p),
                             constant_global_prediction_change_rms=rms(constant_global[sl] - p))
                if arm == 'structured':
                    stats.update(fixed_weight_pooling_rmse=rmse(actual_y, pooled_prediction[sl]),
                                 fixed_weight_pooling_prediction_change_rms=rms(pooled_prediction[sl] - p),
                                 fixed_weight_shuffle_rmse=rmse(actual_y, shuffled_prediction[sl]),
                                 fixed_weight_shuffle_prediction_change_rms=rms(shuffled_prediction[sl] - p))
                result['partitions'][name] = stats
            report['models'][f'{arm}-seed{seed}'] = result
            print(json.dumps(dict(model=f'{arm}-seed{seed}', **result)), flush=True)
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
