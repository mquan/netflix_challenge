#!/usr/bin/env python3
"""Run matched local-bank experiments with one CPU lane and one CPU/GPU lane."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import datetime as dt
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from gbdt_blend import load, blend_fold
from local_ensemble_run import ROOT, RUN, DATA, SEEDS, read, write, digest, verify
from local_ensemble_device import choose_device


def build_stages(arms, workers, threads, nn_threads, device, destination):
    stages = []
    for name in ['baseline', *arms]:
        config = read(RUN/'conditions'/f'{name}.json')
        config['output'] = str(destination/'results'/name)
        config['processes'] = workers
        command = ['scripts/local_ensemble_device.py', 'tree', '--tree-threads', str(threads)]
        for key, value in config.items():
            flag = '--'+key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, list):
                command += [flag, *map(str, value)]
            else:
                command += [flag, str(value)]
        prefix = config['output']
        stages.append(dict(name=name+'-tree', family='tree', deps=[], command=command, report=prefix+'.json'))
        command = ['scripts/local_ensemble_device.py', 'nn', '--device', device,
                   '--data', str(DATA), '--like', prefix+'.json', '--output', prefix+'-NN2',
                   '--hidden', '128', '--layers', '2', '--dropout', '.2', '--lr', '.0003',
                   '--weight-decay', '.01', '--batch', '4096', '--epochs', '60',
                   '--patience', '5', '--seeds', *map(str, SEEDS), '--threads', str(nn_threads)]
        stages.append(dict(name=name+'-nn', family='neural', deps=[name+'-tree'], command=command, report=prefix+'-NN2.json'))
    for name in ['baseline', *(['combined'] if 'combined' in arms else [])]:
        prefix = str(destination/'results'/name)
        for seed in SEEDS:
            out = prefix+f'-ENN-{seed}'
            command = ['scripts/local_ensemble_device.py', 'enn', '--device', device,
                       '--data', str(DATA), '--like', prefix+'.json', '--nn2', prefix+'-NN2',
                       '--output', out, '--nets', '400', '--subset', '4', '--hidden', '8',
                       '--epochs', '40', '--patience', '3', '--threads', str(nn_threads),
                       '--seed', str(seed), '--subset-seed', '2026', '--verbose']
            stages.append(dict(name=f'{name}-enn-{seed}', family='neural', deps=[name+'-nn'],
                               command=command, report=out+'.json'))
    return stages


def summary(destination, arms):
    baseline = destination/'results/baseline-NN2.avg.f32'
    if not baseline.exists():
        return
    d = load(DATA)
    y = d['probe']['rating'].astype(np.float64)
    folds = blend_fold(d['probe']['user'])
    base = np.fromfile(baseline, dtype='<f4').astype(np.float64)
    candidates = [(name, destination/'results'/f'{name}-NN2.avg.f32') for name in ['baseline', *arms]]
    for name in ['baseline', 'combined']:
        paths = [destination/'results'/f'{name}-ENN-{s}.avg3.f32' for s in SEEDS]
        if all(p.exists() for p in paths):
            pred = np.mean([np.fromfile(p, dtype='<f4').astype(np.float64) for p in paths], axis=0)
            out = destination/'results'/f'{name}-ENN-three-seed.avg3.f32'
            pred.astype('<f4').tofile(out)
            candidates.append((name+'-ENN-three-seed', out))
    def rmse(p, mask=None):
        mask = np.ones(len(y), dtype=bool) if mask is None else mask
        return float(np.sqrt(np.mean((np.clip(p[mask], 1, 5)-y[mask])**2)))
    records = []
    for name, path in candidates:
        if not path.exists():
            continue
        p = np.fromfile(path, dtype='<f4').astype(np.float64)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f'Invalid output {path}')
        records.append(dict(name=name, rmse=rmse(p), delta_rmse=rmse(p)-rmse(base),
                            fold_delta=[rmse(p, folds == f)-rmse(base, folds == f) for f in range(5)],
                            sha256=digest(path)))
    write(destination/'comparison.json', dict(results=records,
          limitation='Matched same-machine/backend original-Probe development comparisons; base checkpoints were historically Probe-selected; no qualifying result'))


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', choices=['snapshots', 'calibration', 'combined'], nargs='+', default=['combined'])
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    ap.add_argument('--fold-workers', type=int, default=min(5, max(1, (os.cpu_count() or 4)//2)))
    ap.add_argument('--tree-threads', type=int, default=3)
    ap.add_argument('--nn-threads', type=int, default=4)
    ap.add_argument('--output', type=Path, default=RUN/'remote')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if min(a.fold_workers, a.tree_threads, a.nn_threads) < 1 or len(set(a.arms)) != len(a.arms):
        ap.error('Positive worker/thread counts and unique arms required')
    device = str(choose_device(a.device))
    destination = a.output.resolve()
    stages = build_stages(a.arms, a.fold_workers, a.tree_threads, a.nn_threads, device, destination)
    if a.dry_run:
        print(json.dumps(dict(device=device, stages=stages), indent=2))
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'results').mkdir(exist_ok=True)
    (destination/'logs').mkdir(exist_ok=True)
    lock = destination/'runner.lock'
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f'Remote runner already active: {pid}')
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    active = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        local_plan = read(RUN/'plan.json')
        verify(local_plan)
        plan = dict(device=device, arms=a.arms, stages=stages,
                    local_plan_sha256=digest(RUN/'plan.json'),
                    code={str(p): digest(p) for p in (ROOT/'scripts').glob('*.py')},
                    package_versions=local_plan['package_versions'])
        plan_path = destination/'plan.json'
        if plan_path.exists() and read(plan_path) != plan:
            raise ValueError('Remote plan differs; resume with identical arguments or a new --output')
        write(plan_path, plan)
        done = set()
        for stage in stages:
            receipt = destination/'logs'/f'{stage["name"]}.receipt.json'
            if receipt.exists():
                old = read(receipt)
                if old['report_sha256'] != digest(stage['report']):
                    raise ValueError(f'Completed report changed: {stage["name"]}')
                done.add(stage['name'])
        while len(done) < len(stages):
            for name, (stage, process, log, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                if code:
                    raise RuntimeError(f'{name} exited {code}; inspect {log.name}')
                write(destination/'logs'/f'{name}.receipt.json',
                      dict(report_sha256=digest(stage['report']), elapsed_seconds=time.monotonic()-started))
                done.add(name)
                del active[name]
                summary(destination, a.arms)
            ready = [s for s in stages if s['name'] not in done and s['name'] not in active
                     and set(s['deps']) <= done]
            for stage in ready:
                if any(other[0]['family'] == stage['family'] for other in active.values()):
                    continue
                if Path(stage['report']).exists():
                    raise ValueError(f'Unreceipted report exists: {stage["report"]}')
                verify(local_plan)
                log = (destination/'logs'/f'{stage["name"]}.log').open('a', buffering=1)
                log.write(json.dumps(dict(command=stage['command']))+'\n')
                process = subprocess.Popen([sys.executable, '-u', *stage['command']], cwd=ROOT,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[stage['name']] = (stage, process, log, time.monotonic())
            write(destination/'status.json', dict(state='running', pid=os.getpid(), device=device,
                   completed=sorted(done), total=len(stages),
                   active={name: p[1].pid for name, p in active.items()},
                   updated_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            time.sleep(2)
        summary(destination, a.arms)
        write(destination/'status.json', dict(state='complete', device=device, total=len(stages)))
    except BaseException as error:
        write(destination/'status.json', dict(state='failed', error=repr(error)))
        raise
    finally:
        for stage, process, log, started in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            log.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
