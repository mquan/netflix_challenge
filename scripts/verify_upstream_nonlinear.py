#!/usr/bin/env python3
"""Replay saved nonlinear models and independently score frozen float32 outputs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import upstream_nonlinear as n


def neural_model(checkpoint):
    import torch
    torch.set_num_threads(n.CONFIG['threads'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['config'] != n.CONFIG: raise ValueError('Saved neural recipe changed')
    width, hidden = saved['feature_width'], saved['config']['nn_hidden']
    model = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.GELU(),
        torch.nn.Dropout(n.CONFIG['nn_dropout']), torch.nn.Linear(hidden, hidden),
        torch.nn.GELU(), torch.nn.Dropout(n.CONFIG['nn_dropout']), torch.nn.Linear(hidden, 1))
    model.load_state_dict(saved['state'])
    return model


def replay(family):
    n.verify_prepared()
    features = np.load(n.RUN/'features.npy', mmap_mode='r')
    z = np.load(n.RUN/'design.npy', mmap_mode='r')
    groups = np.load(n.RUN/'groups.npy'); records = []
    for fold in range(5):
        root = n.RUN/family/str(fold)
        receipt = n.u.read(n.REPORT/family/(str(fold)+'.json'))
        if receipt['manifest_sha256'] != n.u.sha(n.REPORT/'manifest.json'):
            raise ValueError('Model recipe changed')
        for filename, digest in receipt['outputs'].items():
            if n.u.sha(root/filename) != digest: raise ValueError('Model output changed: '+filename)
        held = groups//2 == fold
        with np.load(root/'ridge.npz') as weights:
            base = np.clip(z@weights['full'], 1.05, 4.95)
        # Match float32 conversion before applying the saved training-only scaler.
        inputs = np.column_stack([features, base]).astype(np.float32)[held]
        if family == 'nn':
            with np.load(root/'scalers.npz') as saved:
                scaler = {key:saved['full_'+key] for key in ['lo','hi','mean','std']}
            inputs, _ = n.scaled(inputs, None, scaler=scaler)
        correction = np.zeros(held.sum())
        for seed, iterations in zip(n.CONFIG['seeds'], receipt['chosen_iterations']):
            if not iterations: continue
            if family == 'tree':
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(root/f'tree-{seed}.txt'))
                correction += model.predict(inputs, num_threads=n.CONFIG['threads'])/len(n.CONFIG['seeds'])
            else:
                model = neural_model(root/f'nn-{seed}.pt')
                correction += n.nn_predict(model, inputs)/len(n.CONFIG['seeds'])
            del model
        actual = np.clip(base[held]+correction, 1.05, 4.95).astype('<f4')
        expected = np.fromfile(root/'prediction.f32', dtype='<f4')
        if not np.array_equal(actual, expected):
            raise ValueError(f'Saved model replay differs for {family} fold {fold}: {np.max(np.abs(actual-expected))}')
        records.append(dict(fold=fold, rows=len(actual), bitwise_equal=True))
    n.u.write(n.REPORT/(family+'-replay.json'), dict(family=family, records=records,
              verifier_sha256=n.u.sha(Path(__file__))))


def score():
    n.verify_prepared()
    report = n.u.read(n.REPORT/'evaluation.json')
    y = np.load(n.RUN/'labels.npy'); users = np.load(n.RUN/'users.npy')
    baseline = np.clip(np.load(n.RUN/'baseline.npy'), 1.05, 4.95)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse); clusters = len(counts)
    records = []; predictions = {}
    for row in report['records']:
        path = n.RUN/(row['arm']+'.f32')
        if n.u.sha(path) != row['prediction_sha256']: raise ValueError('Final prediction hash changed')
        p = np.fromfile(path, dtype='<f4').astype(np.float64)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p<1.05-1e-6)|(p>4.95+1e-6)):
            raise ValueError('Malformed final prediction')
        predictions[row['arm']] = p
        p = np.clip(p, 1.05, 4.95)
        rmse = math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(p, y))/len(y))
        if abs(rmse-row['rmse']) > 1e-12: raise ValueError('Independent scorer differs')
        paired = (p-y)**2-(baseline-y)**2
        mean = float(paired.mean())
        # Cluster-robust SE of the row-weighted paired MSE, conditional on this bank.
        total = np.bincount(inverse, weights=paired)
        centered = total-counts*mean
        se = math.sqrt(clusters/(clusters-1)*float(centered@centered))/len(y)
        records.append(dict(arm=row['arm'], independent_rmse=rmse, delta_mse=mean,
            user_cluster_normal_95_mse_interval=[mean-1.96*se, mean+1.96*se],
            prediction_sha256=row['prediction_sha256']))
    expected = ((predictions['tree']+predictions['nn'])/2).astype('<f4')
    actual = np.fromfile(n.RUN/'fixed-half-tree-nn.f32', dtype='<f4')
    if not np.array_equal(expected, actual): raise ValueError('Fixed average differs')
    n.u.write(n.REPORT/'independent-verification.json', dict(state='complete',
        evaluation_sha256=n.u.sha(n.REPORT/'evaluation.json'), verifier_sha256=n.u.sha(Path(__file__)),
        fixed_average_bitwise_equal=True, scored_users=clusters, records=records,
        limitation='Intervals condition on this historically selected bank and fixed fitted predictions; no selection correction or official Test claim.'))
    print(records, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=['tree','nn']); args = ap.parse_args()
    if args.family: replay(args.family)
    else:
        for family in ['tree', 'nn']:
            subprocess.run([sys.executable, str(Path(__file__)), '--family', family], check=True)
        score()
