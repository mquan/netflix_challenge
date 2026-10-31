#!/usr/bin/env python3
"""Score the A7 sibling-split Stage C rehearsal (protocol in HEADROOM.md).

Arms on probe half B; every fitted coefficient comes from half A except the
labelled diagnostic arm 5:
  1 dev        ridge blend of dev columns (trained on T), fit on A
  2 dev+pool   arm 1 plus pooled half-A sibling residuals, coefficients fit on A (LOO)
  3 stageC     the arm-1 blender applied to columns retrained on T + A
  4 stageC+pool  arm 3 plus pooling with arm 2's A-fitted coefficients (deployable)
  5 diagnostic   arm 3 plus pooling, coefficients refit on B by user folds (selects nothing)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from sibling_pooling import design
from train_holdout import digest

DEV = ['artifacts/rehearsal/B-K1.f32', 'artifacts/rehearsal/B-knn256td.netflix.f32',
       'artifacts/rehearsal/B-knn64.netflix.f32', 'artifacts/rehearsal/B-knn256td.netflix.support.f32']
STAGE_C = ['artifacts/sibling/K1.f32', 'artifacts/sibling/knn256td.f32', 'artifacts/sibling/knn64.f32',
           'artifacts/sibling/knn256td.support.f32']
PENALTY = 10.


def ridge(x, y):
    x = np.column_stack([np.ones(len(y)), x])
    return np.linalg.solve(x.T @ x + PENALTY * np.eye(x.shape[1]), x.T @ y)


def apply(w, x):
    return np.column_stack([np.ones(len(x)), x]) @ w


def cols(paths):
    return np.column_stack([np.fromfile(p, dtype='<f4').astype(float) for p in paths])


def run(output, data_path='data/processed/netflix.bin', amask='artifacts/sibling/T-plus-A.bin.amask'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    d = load(data_path)
    p = d['probe']
    y = p['rating'].astype(float)
    a = np.fromfile(amask, dtype='u1').astype(bool)
    b = ~a
    if len(a) != len(y):
        raise ValueError('mask does not match probe')
    dev = cols(DEV)
    w_blend = ridge(dev[a], y[a])                    # blender fit on A only
    base_dev = apply(w_blend, dev)                   # full length; A part used for residuals
    e = y - base_dev                                 # honest: dev columns never saw probe labels
    xa = design(p, base_dev, e, a, a, leave_out=True)
    w_pool = np.linalg.solve(xa.T @ xa + PENALTY * np.eye(xa.shape[1]), xa.T @ y[a])
    xb_dev = design(p, base_dev, e, a, b, leave_out=False)
    arms = {'1_dev': base_dev[b], '2_dev_pool': xb_dev @ w_pool}
    have_c = all(Path(c).exists() for c in STAGE_C)
    if have_c:
        stage = cols(STAGE_C)
        if len(stage) != b.sum():
            raise ValueError('stage C columns do not match half B')
        base_c = np.zeros(len(y)); base_c[b] = apply(w_blend, stage)
        xb_c = design(p, np.where(b, base_c, base_dev), e, a, b, leave_out=False)
        arms['3_stageC'] = base_c[b]
        arms['4_stageC_pool'] = xb_c @ w_pool
        folds = blend_fold(p['user'][b])
        diag = np.empty(b.sum())
        for f in range(5):
            fit = folds != f
            w = np.linalg.solve(xb_c[fit].T @ xb_c[fit] + PENALTY * np.eye(xb_c.shape[1]), xb_c[fit].T @ y[b][fit])
            diag[~fit] = xb_c[~fit] @ w
        arms['5_diag_stageC_pool_refit_on_B'] = diag
    yb, ub = y[b], p['user'][b]
    rmse = {k: float(np.sqrt(np.mean((np.clip(v, 1, 5) - yb) ** 2))) for k, v in arms.items()}
    comparisons = {'pool_on_dev(1->2)': paired(yb, arms['1_dev'], arms['2_dev_pool'], ub)}
    if have_c:
        comparisons['sibling_stageC_offset(1->3)'] = paired(yb, arms['1_dev'], arms['3_stageC'], ub)
        comparisons['pool_on_stageC(3->4)'] = paired(yb, arms['3_stageC'], arms['4_stageC_pool'], ub)
        comparisons['diag_ideal_pool_on_stageC(3->5)'] = paired(yb, arms['3_stageC'],
                                                               arms['5_diag_stageC_pool_refit_on_B'], ub)
    report = dict(dev=DEV, stage_c=STAGE_C if have_c else None, amask_sha256=digest(amask),
                  stage_c_sha256={c: digest(c) for c in STAGE_C} if have_c else None,
                  rmse=rmse, comparisons=comparisons, blend_weights=w_blend.tolist(), pool_weights=w_pool.tolist(),
                  metric='Probe half B; blender and pooling coefficients fit on half A except arm 5 (diagnostic).')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rmse=rmse, deltas={k: (v['delta_rmse'], v['fold_deltas']) for k, v in comparisons.items()}),
                     indent=1))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    run(ap.parse_args().output)


if __name__ == '__main__':
    main()
