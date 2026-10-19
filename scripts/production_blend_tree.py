#!/usr/bin/env python3
"""Isolated LightGBM worker: never import PyTorch or the neural-blend module."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['fit', 'predict'])
    p.add_argument('--features', required=True)
    p.add_argument('--residual')
    p.add_argument('--settings', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output')
    p.add_argument('--batch', type=int, default=65536)
    a = p.parse_args()
    settings = json.loads(Path(a.settings).read_text())
    if lgb.__version__ != settings['lightgbm_version']:
        raise ValueError('LightGBM version changed')
    x = np.load(a.features, mmap_mode='r', allow_pickle=False)
    models = [Path(a.model_dir) / f'tree-{i}.txt' for i in range(len(settings['budgets']['tree']))]
    if not models or a.batch <= 0:
        raise ValueError('tree arms and positive batch size required')
    if a.action == 'fit':
        if not a.residual:
            p.error('--residual required')
        if any(path.exists() for path in models):
            raise FileExistsError('tree artifact already exists')
        residual = np.load(a.residual, allow_pickle=False)
        cfg = settings['tree_config']
        for path, arm in zip(models, settings['budgets']['tree']):
            params = dict(objective='regression', learning_rate=cfg['learning_rate'], num_leaves=cfg['leaves'],
                min_data_in_leaf=cfg['min_leaf'], feature_fraction=.8, bagging_fraction=.8,
                bagging_freq=1, lambda_l2=10, num_threads=1, verbose=-1, seed=arm['seed'])
            model = lgb.train(params, lgb.Dataset(x, residual), num_boost_round=arm['steps'])
            model.save_model(str(path))
            print(json.dumps(dict(stage='tree', **arm)), flush=True)
    else:
        if not a.output:
            p.error('--output required')
        if Path(a.output).exists():
            raise FileExistsError(a.output)
        estimators = [lgb.Booster(model_file=str(path)) for path in models]
        out = np.lib.format.open_memmap(a.output, mode='w+', dtype='<f8', shape=(len(x),))
        for start in range(0, len(x), a.batch):
            stop = min(start+a.batch, len(x))
            values = np.zeros(stop-start)
            for model in estimators:
                values += model.predict(x[start:stop], num_threads=1)/len(estimators)
            out[start:stop] = values
        out.flush()


if __name__ == '__main__':
    main()
