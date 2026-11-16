#!/usr/bin/env python3
"""Low-memory one-user-fold ridge diagnostic alongside the unchanged full trial."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
import datetime as dt
import json
from pathlib import Path
import sys
import time
import numpy as np
from gbdt_blend import blend_fold, load
from local_ensemble_run import ROOT, RUN, DATA, META, OLD, digest, read, write, verify


def design_block(predictions, context):
    z = np.column_stack([np.ones(len(predictions)), predictions.astype(np.float64)-3.6])
    return (context[:, :, None].astype(np.float64)*z[:, None, :]).reshape(len(z), -1)


def accumulate(predictions, context, y, fit, chunk=8192, progress=None):
    width = (predictions.shape[1]+1)*context.shape[1]
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros(width, dtype=np.float64)
    count = 0
    for lo in range(0, len(y), chunk):
        hi = min(lo+chunk, len(y))
        take = np.flatnonzero(fit[lo:hi])+lo
        x = design_block(predictions[take], context[take])
        gram += x.T@x
        rhs += x.T@(y[take]-3.6)
        count += len(take)
        if progress:
            progress(hi, len(y))
    return gram, rhs, count


def solve(gram, rhs, count, indices):
    a = gram[np.ix_(indices, indices)].copy()
    diagonal = np.arange(len(indices))
    a[diagonal, diagonal] += count*1e-5
    a[0, 0] -= count*1e-5
    return np.linalg.solve(a, rhs[indices])


def main():
    os.chdir(ROOT)
    os.nice(10)
    destination = RUN/'quick-screen.json'
    if destination.exists():
        raise ValueError('Quick-screen report exists; refuse overwrite')
    plan = read(RUN/'plan.json')
    verify(plan)
    source = read(RUN/'bank/experiments/UNION-round-manifest.json')['reports']['union']['inputs']
    extras = read(RUN/'conditions/combined.json')['extra']
    paths = [RUN/'bank'/p for p in source]+[Path(p) for p in extras]
    # This table is ~550 MiB; the expanded interaction matrix exists only in blocks.
    columns = [np.memmap(p, mode='r', dtype='<f4') for p in paths]
    predictions = np.column_stack(columns)
    context = np.load(META)['context']
    d = load(DATA)
    y = d['probe']['rating'].astype(np.float64)
    users = d['probe']['user']
    held = blend_fold(users) == 0
    start = time.monotonic()
    def progress(done, total):
        if done == total or (done//8192) % 16 == 0:
            status = dict(state='accumulating', rows_processed=done, total_rows=total,
                          elapsed_seconds=time.monotonic()-start, pid=os.getpid())
            write(RUN/'quick-screen-status.json', status)
            print(json.dumps(status), flush=True)
    gram, rhs, count = accumulate(predictions, context, y, ~held, progress=progress)
    nsnapshot = len(read(RUN/'conditions/snapshots.json')['extra'])
    stride = predictions.shape[1]+1
    layouts = {
        'baseline': np.arange(80),
        'snapshots': np.arange(80+nsnapshot),
        'calibration': np.r_[np.arange(80), np.arange(80+nsnapshot, stride)],
        'combined': np.arange(stride),
    }
    held_rows = np.flatnonzero(held)
    results = {}
    for name, layout in layouts.items():
        indices = np.concatenate([layout+i*stride for i in range(context.shape[1])])
        weights = solve(gram, rhs, count, indices)
        out = np.empty(len(held_rows))
        for lo in range(0, len(held_rows), 8192):
            hi = min(lo+8192, len(held_rows))
            take = held_rows[lo:hi]
            out[lo:hi] = 3.6+design_block(predictions[take], context[take])[:, indices]@weights
        results[name] = out
    old = np.fromfile(str(OLD)+'-s3.ridge.f32', dtype='<f4')[held]
    parity = float(np.max(np.abs(results['baseline']-old)))
    if parity > 2e-6:
        raise ValueError(f'Quick-screen baseline does not match historical ridge: {parity}')
    base_error = (np.clip(results['baseline'], 1, 5)-y[held])**2
    records = []
    for name, pred in results.items():
        error = (np.clip(pred, 1, 5)-y[held])**2
        records.append(dict(name=name, rmse=float(np.sqrt(error.mean())),
                            delta_rmse=float(np.sqrt(error.mean())-np.sqrt(base_error.mean()))))
    report = dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-start, outer_fold=0,
                  train_rows=count, held_rows=int(held.sum()), alpha=1e-5,
                  historical_ridge_max_difference=parity,
                  fitting_labels='Outer-training users only',
                  limitation='One-fold contextual ridge screen; not the tree/neural ensemble, not full CV or independent Test; does not select or cancel full-trial arms',
                  plan_sha256=digest(RUN/'plan.json'), results=records)
    verify(plan)
    write(destination, report)
    write(RUN/'quick-screen-status.json', dict(state='complete', report=str(destination)))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
