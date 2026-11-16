#!/usr/bin/env python3
"""Run the predeclared local-only R9 continuation, with durable resumable stages."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
from gbdt_blend import blend_fold, load

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'artifacts/independent-ensemble'
OLD = RUN/'control/R9-LAT-A2h-20261010'
DATA = ROOT/'data/processed/netflix.bin'
META = ROOT/'artifacts/r4/meta-context.npz'
SEEDS = [2026, 7, 13]
TARGETS = ['A2h-refit.refit.e12.f32', 'B4-full-r128.s0160.f32',
           'MT3-refit.refit.e7.f32', 'MH3-categorical-mse-refit.f32',
           'R4D-rbm200-cond.f32', 'joint256-reg01.f32']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def score(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 1, 5)-y)**2)))


def stage_commands():
    stages = []
    for name in ['baseline', 'snapshots', 'calibration', 'combined']:
        config = read(RUN/'conditions'/f'{name}.json')
        output = RUN/'results'/name
        args = ['scripts/round4_boost.py']
        for k, v in config.items():
            flag = '--'+k.replace('_', '-')
            if isinstance(v, bool):
                if v:
                    args.append(flag)
            elif isinstance(v, list):
                args.extend([flag, *map(str, v)])
            else:
                args.extend([flag, str(v)])
        stages.append(dict(name=name+'-tree', command=args, report=str(output)+'.json'))
        args = ['scripts/round8_nn_blend.py', '--data', str(DATA),
                '--like', str(output)+'.json', '--output', str(output)+'-NN2',
                '--hidden', '128', '--layers', '2', '--dropout', '.2',
                '--lr', '.0003', '--weight-decay', '.01', '--batch', '4096',
                '--epochs', '60', '--patience', '5', '--seeds', *map(str, SEEDS),
                '--threads', '2']
        stages.append(dict(name=name+'-nn', command=args, report=str(output)+'-NN2.json'))
    for name in ['baseline', 'combined']:
        for seed in SEEDS:
            output = RUN/'results'/f'{name}-ENN-{seed}'
            args = ['scripts/round9_enn_blend.py', '--data', str(DATA),
                    '--like', str(RUN/'results'/name)+'.json',
                    '--nn2', str(RUN/'results'/name)+'-NN2', '--output', str(output),
                    '--nets', '400', '--subset', '4', '--hidden', '8',
                    '--epochs', '40', '--patience', '3', '--threads', '2',
                    '--seed', str(seed), '--subset-seed', '2026', '--verbose']
            stages.append(dict(name=f'{name}-enn-{seed}', command=args,
                               report=str(output)+'.json'))
    return stages


def prepare():
    if (RUN/'plan.json').exists():
        raise ValueError('Plan already frozen; use --run to resume')
    tree = read(str(OLD)+'-s3.json')
    bank = RUN/'bank'
    frozen = {}
    for line in (bank/'MANIFEST.sha256').read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        p = bank/rel.strip()
        if digest(p) != expected:
            raise ValueError(f'Bank hash mismatch: {p}')
        frozen[str(p)] = expected
    names = read(bank/'experiments/UNION-round-manifest.json')['reports']['union']['inputs']
    if len(names) != 79:
        raise ValueError('Expected exactly 79 local R9 predictors')
    expected = set(tree['inputs'].values())
    for p in bank.rglob('*.f32'):
        if digest(p) not in expected:
            raise ValueError(f'Input is not in original local R9 report: {p}')
    d = load(DATA)
    y = d['probe']['rating'].astype(np.float64)
    folds = blend_fold(d['probe']['user'])
    saved = np.fromfile(str(OLD)+'-NN2.avg.f32', dtype='<f4')
    original = read(str(OLD)+'-NN2.json')
    measured = score(saved, y)
    if abs(measured-original['avg_rmse']) > 1e-7:
        raise ValueError('Saved R9 score does not reproduce')
    audit = dict(state='verified', rows=len(y), saved_rmse=measured,
                 recorded_rmse=original['avg_rmse'], prediction_columns=79,
                 fold_rmse=[score(saved[folds == f], y[folds == f]) for f in range(5)],
                 limitation='Probe-selected historical base models; development evidence only')
    for p in (RUN/'control').iterdir():
        if p.is_file():
            frozen[str(p)] = digest(p)
    frozen[str(DATA)] = digest(DATA)
    frozen[str(META)] = digest(META)
    nbytes = len(y)*4
    snapshots = sorted((RUN/'snapshots').glob('A2h-e*.f32'))
    if len(snapshots) != 12:
        raise ValueError('Expected twelve original locally trained snapshots')
    for p in snapshots:
        if p.stat().st_size != nbytes:
            raise ValueError(f'Wrong snapshot rows: {p}')
        frozen[str(p)] = digest(p)
    if digest(snapshots[-1]) not in names.values():
        raise ValueError('Epoch 12 snapshot is not the frozen R9 input')
    features = RUN/'calibration'
    features.mkdir(exist_ok=True)
    generated = []
    for name in TARGETS:
        matches = {v for k, v in tree['inputs'].items() if Path(k).name == name}
        if len(matches) != 1:
            raise ValueError(f'Missing or ambiguous predeclared local predictor {name}')
        h = matches.pop()
        candidates = [bank/k for k, v in names.items() if v == h]
        if not candidates:
            raise ValueError(f'Predictor missing from restored bank: {name}')
        p = np.fromfile(candidates[0], dtype='<f4').astype(np.float64)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError('Bad local predictor')
        for degree in [2, 3]:
            dest = features/f'{name}.power{degree}.f32'
            (3.6+(p-3.6)**degree).astype('<f4').tofile(dest)
            generated.append(str(dest))
            frozen[str(dest)] = digest(dest)
    (RUN/'conditions').mkdir(exist_ok=True)
    (RUN/'results').mkdir(exist_ok=True)
    (RUN/'logs').mkdir(exist_ok=True)
    for name in ['baseline', 'snapshots', 'calibration', 'combined']:
        extras = []
        if name in ['snapshots', 'combined']:
            extras += [str(p) for p in snapshots[:-1]]
        if name in ['calibration', 'combined']:
            extras += generated
        config = dict(data=str(DATA), output=str(RUN/'results'/name),
                      bank_root=str(bank), no_union=False, extra=extras, extra_support=[],
                      rounds=8000, learning_rate=.02, leaves=63, min_leaf=200,
                      processes=1, seeds=SEEDS, extra_meta=True)
        path = RUN/'conditions'/f'{name}.json'
        write(path, config)
        frozen[str(path)] = digest(path)
    code_paths = list((ROOT/'scripts').glob('*.py'))
    code_hashes = {str(p): digest(p) for p in code_paths}
    write(RUN/'baseline-audit.json', audit)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'lightgbm']}
    plan = dict(created_utc=now(), purpose='Predeclared local-only R9 ensemble continuation',
                package_versions=versions,
                branches=dict(local='research/independent-ensemble'),
                fitting_labels='Original Probe development ratings only',
                excluded='Public predictions, pretrained upstream weights, recovered qualifying labels',
                limitation=audit['limitation'], inputs=frozen, code=code_hashes,
                stages=stage_commands(), historical_baseline=audit,
                calibration_targets=TARGETS, seeds=SEEDS)
    write(RUN/'plan.json', plan)
    write(RUN/'status.json', dict(state='prepared', updated_utc=now(), next_stage=0,
                                 total_stages=len(plan['stages'])))
    print(json.dumps(audit), flush=True)


def verify(plan):
    for name, version in plan['package_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Frozen package version changed: {name}')
    for p, expected in {**plan['inputs'], **plan['code']}.items():
        if digest(p) != expected:
            raise ValueError(f'Frozen input/code changed: {p}')


def summarize():
    d = load(DATA)
    y = d['probe']['rating'].astype(np.float64)
    user = d['probe']['user']
    folds = blend_fold(user)
    base = np.fromfile(RUN/'results/baseline-NN2.avg.f32', dtype='<f4').astype(np.float64)
    records = []
    candidates = [(name, RUN/'results'/f'{name}-NN2.avg.f32')
                  for name in ['baseline', 'snapshots', 'calibration', 'combined']]
    for name in ['baseline', 'combined']:
        for kind in ['enn', 'avg3']:
            paths = [RUN/'results'/f'{name}-ENN-{s}.{kind}.f32' for s in SEEDS]
            if all(p.exists() for p in paths):
                value = np.mean([np.fromfile(p, dtype='<f4').astype(np.float64) for p in paths], axis=0)
                dest = RUN/'results'/f'{name}-{kind}-three-seed.f32'
                value.astype('<f4').tofile(dest)
                candidates.append((name+'-'+kind+'-three-seed', dest))
    for name, path in candidates:
        if not path.exists():
            continue
        p = np.fromfile(path, dtype='<f4').astype(np.float64)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f'Invalid completed prediction: {path}')
        difference = (np.clip(p, 1, 5)-y)**2-(np.clip(base, 1, 5)-y)**2
        sums = np.bincount(user, difference, minlength=d['users'])
        counts = np.bincount(user, minlength=d['users'])
        active = counts > 0
        delta_mse = float(difference.mean())
        influence = sums[active]-delta_mse*counts[active]
        clusters = int(active.sum())
        se = float(np.sqrt(clusters/(clusters-1)*np.dot(influence, influence))/len(y))
        records.append(dict(name=name, rmse=score(p, y), delta_rmse=score(p, y)-score(base, y),
                            fold_delta=[score(p[folds == f], y[folds == f])-score(base[folds == f], y[folds == f]) for f in range(5)],
                            delta_mse=delta_mse, paired_user_cluster_mse_interval=[delta_mse-1.96*se, delta_mse+1.96*se],
                            sha256=digest(path)))
    historical = read(RUN/'baseline-audit.json')['saved_rmse']
    seed_scores = {}
    for name in ['baseline', 'combined']:
        seed_scores[name] = {str(s): read(RUN/'results'/f'{name}-ENN-{s}.json')['avg3_rmse']
                             for s in SEEDS if (RUN/'results'/f'{name}-ENN-{s}.json').exists()}
    write(RUN/'comparison.json', dict(updated_utc=now(), metric='Original-Probe development RMSE',
                                     historical_baseline_rmse=historical,
                                     fresh_baseline_delta=score(base, y)-historical,
                                     enn_seed_scores=seed_scores,
                                     limitation='Conditional comparisons on historically Probe-selected models, adaptive research; not independent Test',
                                     results=records))


def run():
    lock = RUN/'runner.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
            return run()
        raise RuntimeError(f'Runner already active, PID {pid}')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    inhibitor = None
    try:
        if Path('/usr/bin/caffeinate').exists():
            inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        os.nice(10)
        plan = read(RUN/'plan.json')
        verify(plan)
        for index, stage in enumerate(plan['stages']):
            receipt = RUN/'logs'/f'{stage["name"]}.receipt.json'
            if receipt.exists():
                done = read(receipt)
                if done['returncode'] != 0 or digest(stage['report']) != done['report_sha256']:
                    raise ValueError(f'Invalid completed stage: {stage["name"]}')
                continue
            if Path(stage['report']).exists():
                raise ValueError(f'Output exists without a completion receipt: {stage["report"]}')
            verify(plan)
            status = dict(state='running', pid=os.getpid(), stage=stage['name'],
                          next_stage=index, total_stages=len(plan['stages']), updated_utc=now())
            write(RUN/'status.json', status)
            print(json.dumps(status), flush=True)
            started = time.monotonic()
            path = RUN/'logs'/f'{stage["name"]}.log'
            with path.open('a') as stream:
                stream.write(json.dumps(dict(started_utc=now(), command=stage['command']))+'\n')
                stream.flush()
                process = subprocess.Popen([sys.executable, '-u', *stage['command']], cwd=ROOT,
                                           stdout=stream, stderr=subprocess.STDOUT)
                status['child_pid'] = process.pid
                write(RUN/'status.json', status)
                code = process.wait()
            if code:
                raise RuntimeError(f'{stage["name"]} exited {code}; inspect {path}')
            if stage['name'] == 'baseline-tree':
                old = np.fromfile(str(OLD)+'-s3.ridge.f32', dtype='<f4')
                fresh = np.fromfile(RUN/'results/baseline.ridge.f32', dtype='<f4')
                delta = float(np.max(np.abs(old.astype(np.float64)-fresh)))
                write(RUN/'fresh-ridge-parity.json', dict(max_absolute_difference=delta,
                                                         old_sha256=digest(str(OLD)+'-s3.ridge.f32'),
                                                         new_sha256=digest(RUN/'results/baseline.ridge.f32')))
                if delta > 2e-6:
                    raise ValueError(f'Fresh ridge does not reproduce preserved R9 control: {delta}')
            write(receipt, dict(name=stage['name'], returncode=code, finished_utc=now(),
                                elapsed_seconds=time.monotonic()-started,
                                report_sha256=digest(stage['report'])))
            if (RUN/'results/baseline-NN2.avg.f32').exists():
                summarize()
        verify(plan)
        summarize()
        write(RUN/'status.json', dict(state='complete', updated_utc=now(), total_stages=len(plan['stages'])))
    except BaseException as e:
        write(RUN/'status.json', dict(state='failed', updated_utc=now(), error=repr(e)))
        raise
    finally:
        lock.unlink(missing_ok=True)
        if inhibitor:
            inhibitor.terminate()


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.run:
        run()
    else:
        summarize()


if __name__ == '__main__':
    main()
