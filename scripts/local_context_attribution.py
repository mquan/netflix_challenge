#!/usr/bin/env python3
"""Retrain the LC2 common-feature head without a query-specific local vector."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gbdt_blend import blend_fold
from local_context_chain import paired
from local_structure_train import fit_preprocessing, predict, read_inputs, rmse, train
from train_holdout import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', required=True)
    cli = parser.parse_args()
    output = Path(cli.output)
    if Path(cli.summary).exists():
        parser.error('summary already exists')
    output.mkdir(parents=True, exist_ok=False)
    reference_path = Path('experiments/LC2-comparison.json')
    reference = json.loads(reference_path.read_text())
    a = argparse.Namespace(**reference['config'])
    a.disable_local, a.local_normalization = True, 'none'
    torch.set_num_threads(a.threads)
    arrays, columns, raw, y, users, _ = read_inputs(a.spec, a.context)
    folds = blend_fold(users)
    fit, selection, held, refit = [np.flatnonzero(mask) for mask in
                                  (folds >= 2, folds == 1, folds == 0, folds != 0)]
    sx, sb, _ = fit_preprocessing(columns, raw, y, fit)
    rx, rb, prep = fit_preprocessing(columns, raw, y, refit)
    old = Path(reference['config']['output'])
    np.testing.assert_array_equal(held, np.load(old / 'held_indices.npy'))
    np.testing.assert_array_equal(rb[held], np.fromfile(old / 'baseline.f32', dtype='<f4'))
    np.savez(output / 'preprocessing.npz', **prep)
    predictions, records, pooled_seeds = [], [], []
    for seed in a.seeds:
        path = old / f'pooled-seed{seed}.f32'
        assert digest(path) == reference['outputs'][path.name]
        pooled_seeds.append(np.fromfile(path, dtype='<f4').astype(float))
        _, epochs, history = train(arrays, sx, sb, y, fit, a=a, arm='common',
                                   seed=seed, epochs=a.epochs, validation=selection)
        model, _, refit_history = train(arrays, rx, rb, y, refit, a=a, arm='common',
                                        seed=seed, epochs=epochs)
        value = predict(model, arrays, rx, rb, held, a, False)
        value.astype('<f4').tofile(output / f'common-seed{seed}.f32')
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output / f'common-seed{seed}.pt')
        predictions.append(value.astype(float))
        records.append(dict(seed=seed, chosen_epochs=epochs, selection=history, refit=refit_history))
    # Score all fixed results only after both seeds have finished.
    average = np.mean(predictions, axis=0)
    pooled = np.mean(pooled_seeds, axis=0)
    comparison = paired(y[held], average, pooled, users[held])
    per_seed = {str(seed): paired(y[held], common, local, users[held])
                for seed, common, local in zip(a.seeds, predictions, pooled_seeds)}
    passes = (comparison['delta_rmse'] <= -.0001 and
              comparison['cluster_normal_95_mse_interval'][1] < 0 and
              all(item['delta_rmse'] < 0 for item in per_seed.values()))
    average.astype('<f4').tofile(output / 'common.f32')
    report = dict(config=vars(a), output=str(output), reference_sha256=digest(reference_path),
                  training=records, rmse=dict(ridge=rmse(y[held], rb[held]),
                                             common=rmse(y[held], average), pooled=rmse(y[held], pooled)),
                  pooled_vs_common=comparison, per_seed=per_seed, integration_gate_passed=passes,
                  outputs={p.name: digest(p) for p in sorted(output.iterdir())},
                  limitation='Post-hoc attribution on reused P2 development users; not independent confirmation. '
                             'Encoder parameters are allocated but unused in common-only control.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    Path(cli.summary).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('rmse', 'pooled_vs_common', 'per_seed', 'integration_gate_passed')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
