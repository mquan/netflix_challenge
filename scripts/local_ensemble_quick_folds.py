#!/usr/bin/env python3
"""Finish all five fixed ridge comparisons without changing the full trial."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
import datetime as dt
import json
from pathlib import Path
import time
import numpy as np
from local_ensemble_quick_screen import accumulate, design_block, solve
from local_ensemble_run import ROOT, RUN, DATA, META, OLD, digest, read, write, verify
from gbdt_blend import load, blend_fold


def main():
    os.chdir(ROOT)
    os.nice(10)
    destination = RUN/'quick-five-fold.json'
    if destination.exists():
        raise ValueError('Five-fold report exists; refuse overwrite')
    plan = read(RUN/'plan.json')
    verify(plan)
    write(RUN/'quick-five-fold-plan.json', dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
           full_plan_sha256=digest(RUN/'plan.json'), code_sha256=digest(__file__),
           purpose='Complete the same fixed ridge diagnostic on all five existing user folds; no arm selection or cancellation',
           alpha=1e-5, folds=list(range(5)), conditions=['baseline', 'snapshots', 'calibration', 'combined']))
    source = read(RUN/'bank/experiments/UNION-round-manifest.json')['reports']['union']['inputs']
    extras = read(RUN/'conditions/combined.json')['extra']
    paths = [RUN/'bank'/p for p in source]+[Path(p) for p in extras]
    predictions = np.column_stack([np.memmap(p, mode='r', dtype='<f4') for p in paths])
    context = np.load(META)['context']
    d = load(DATA)
    y = d['probe']['rating'].astype(np.float64)
    folds = blend_fold(d['probe']['user'])
    nsnapshot = len(read(RUN/'conditions/snapshots.json')['extra'])
    stride = predictions.shape[1]+1
    layouts = dict(baseline=np.arange(80), snapshots=np.arange(80+nsnapshot),
                   calibration=np.r_[np.arange(80), np.arange(80+nsnapshot, stride)],
                   combined=np.arange(stride))
    indices = {name: np.concatenate([layout+i*stride for i in range(context.shape[1])])
               for name, layout in layouts.items()}
    out = {name: np.full(len(y), np.nan) for name in layouts}
    old = np.fromfile(str(OLD)+'-s3.ridge.f32', dtype='<f4')
    parity = []
    start = time.monotonic()
    for fold in range(5):
        held = folds == fold
        gram, rhs, count = accumulate(predictions, context, y, ~held)
        held_rows = np.flatnonzero(held)
        for name, selected in indices.items():
            weights = solve(gram, rhs, count, selected)
            for lo in range(0, len(held_rows), 8192):
                take = held_rows[lo:lo+8192]
                out[name][take] = 3.6+design_block(predictions[take], context[take])[:, selected]@weights
        parity.append(float(np.max(np.abs(out['baseline'][held]-old[held]))))
        if parity[-1] > 2e-6:
            raise ValueError(f'Historical ridge parity failed on fold {fold}')
        status = dict(fold_complete=fold, elapsed_seconds=time.monotonic()-start,
                      rmse={name: float(np.sqrt(np.mean((np.clip(p[held], 1, 5)-y[held])**2)))
                            for name, p in out.items()})
        write(RUN/'quick-five-fold-status.json', status)
        print(json.dumps(status), flush=True)
    def rmse(p, mask=None):
        mask = np.ones(len(y), dtype=bool) if mask is None else mask
        return float(np.sqrt(np.mean((np.clip(p[mask], 1, 5)-y[mask])**2)))
    base_error = (np.clip(out['baseline'], 1, 5)-y)**2
    records = []
    for name, p in out.items():
        delta = (np.clip(p, 1, 5)-y)**2-base_error
        users = d['probe']['user']
        counts = np.bincount(users)
        active = counts > 0
        influence = np.bincount(users, delta)-delta.mean()*counts
        k = int(active.sum())
        se = float(np.sqrt(k/(k-1)*np.sum(influence[active]**2))/len(y))
        dest = RUN/f'quick-five-fold-{name}.f32'
        p.astype('<f4').tofile(dest)
        records.append(dict(name=name, rmse=rmse(p), delta_rmse=rmse(p)-rmse(out['baseline']),
                            fold_delta=[rmse(p, folds == f)-rmse(out['baseline'], folds == f) for f in range(5)],
                            paired_user_cluster_mse_interval=[float(delta.mean()-1.96*se), float(delta.mean()+1.96*se)],
                            sha256=digest(dest)))
    verify(plan)
    report = dict(state='complete', elapsed_seconds=time.monotonic()-start,
                  rows=len(y), folds=5, alpha=1e-5,
                  historical_ridge_max_difference=max(parity), results=records,
                  limitation='Fixed five-user-fold contextual ridge development diagnostic, not the stronger tree/neural control or an untouched Test; full trial unchanged')
    write(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
