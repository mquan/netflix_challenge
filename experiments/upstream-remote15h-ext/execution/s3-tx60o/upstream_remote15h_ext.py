#!/usr/bin/env python3
"""Remote15h extension: identical to upstream_remote15h.py except PLAN path, epoch-line regex
(also accepts unscored qual-phase lines "Epoch NN" from lib.rs report_rmse) and the terminal==last-epoch
check applying only when the last epoch snapshot exists (jobs without EPSAVE). Never reads qual labels."""
import argparse
import datetime as dt
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from upstream_split3_pilot import arrays, projection, assert_disjoint, keys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/upstream-remote15h-ext/plan.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def environment(knobs=None):
    keep = {'PATH', 'HOME', 'TMPDIR', 'USER', 'LOGNAME', 'SHELL', 'LANG',
            'LC_ALL', 'CARGO_HOME', 'RUSTUP_HOME', 'SDKROOT',
            'MACOSX_DEPLOYMENT_TARGET', 'DYLD_LIBRARY_PATH'}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(RAYON_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               VECLIB_MAXIMUM_THREADS='1')
    env.update(knobs or {})
    return env


def check_files(root, files):
    for rel, expected in files.items():
        if sha(root / rel) != expected:
            raise ValueError('Immutable file differs: ' + str(root / rel))


def vector(path, count):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.shape != (count,) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Invalid float32 prediction: ' + str(path))
    return a


def trace_check(log, epochs):
    trace = [int(n) for n in re.findall(r'^Epoch\s+(\d+)\b', log, re.M)]
    if trace != list(range(1, epochs + 1)) * 2:
        raise ValueError('Both complete original epoch traces required')


def verify_inputs(tree):
    plan = read(PLAN)
    receipt = read(tree.parent / 'preparation.json')
    if receipt['plan_sha256'] != sha(PLAN):
        raise ValueError('Frozen plan changed')
    check_files(tree, plan['source_files_sha256'])
    check_files(tree / 'data', plan['data_sha256'])
    check_files(tree / 'data', plan['split3_sha256'])
    check_files(tree, receipt['binary_sha256'])
    if np.any(np.load(tree / 'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying targets must be zero')
    return plan


def prepare(source, data, tree):
    plan = read(PLAN)
    if tree.exists() or (tree.parent / 'preparation.json').exists():
        raise ValueError('Existing preparation preserved; use verify-inputs')
    if shutil.disk_usage(tree.parent).free < 20 * 2**30:
        raise ValueError('Need at least 20 GiB free before preparation')
    check_files(source, plan['source_files_sha256'])
    check_files(data, plan['data_sha256'])
    if np.any(np.load(data / 'qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying targets must be zero')
    tree.mkdir()
    for rel in plan['source_files_sha256']:
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
    (tree / 'data').mkdir()
    for split in ['train', 'probe', 'fulltrain', 'qual', 'probex']:
        (tree / 'data' / split).symlink_to(data / split, target_is_directory=True)
    with (tree.parent / 'build.log').open('x') as log:
        subprocess.run(['cargo', 'build', '--release', '--locked', '--features',
                        'blas', '--bin', 'fit', '--bin', 'split_3'], cwd=tree,
                       env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    with (tree.parent / 'split3.log').open('x') as log:
        subprocess.run(['./target/release/split_3'], cwd=tree, env=environment(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    check_files(tree / 'data', plan['split3_sha256'])
    source_rows = arrays(tree / 'data/probe_3', labels=True)
    known = arrays(tree / 'data/probex', labels=True)
    mapping = projection(source_rows, known)
    assert_disjoint(np.sort(keys(*known[:3])), arrays(tree / 'data/train_3'))
    rowmap = tree.parent / 'probex-row-map.npy'
    np.save(rowmap, mapping)
    if sha(rowmap) != plan['row_map_sha256']:
        raise ValueError('Exact coordinator row map differs')
    write(tree.parent / 'preparation.json', dict(state='prepared_own_data_only',
          observed=stamp(), plan_sha256=sha(PLAN), source_pin=plan['source_pin'],
          binary_sha256={f'target/release/{n}': sha(tree / 'target/release' / n)
                         for n in ['fit', 'split_3']},
          row_map_sha256=sha(rowmap), known_rows_absent_from_training=True,
          qualifying_targets_zero=True, qualifying_labels_read=False,
          public_prediction_values_read=False))
    verify_inputs(tree)


def run_job(tree, name, deadline):
    plan = verify_inputs(tree)
    job = next(j for j in plan['jobs'] if j['name'] == name)
    remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining < job['minimum_remaining_training_hours'] * 3600:
        raise ValueError('Insufficient reserved time for a complete producer')
    if shutil.disk_usage(tree).free < 20 * 2**30:
        raise ValueError('Need 20 GiB disk headroom before next native job')
    out = tree / 'preds_3'
    if out.exists() and list(out.glob(name + '*')):
        raise ValueError('Existing native outputs preserved; do not automatically restart')
    logpath = tree.parent / (name + '.log')
    with logpath.open('x') as log:
        child = subprocess.Popen(job['command'], cwd=tree,
                                 env=environment(job['environment']),
                                 stdout=log, stderr=subprocess.STDOUT)
        write(tree.parent / (name + '-launch.json'), dict(job=name, pid=child.pid,
              started=stamp(), deadline=deadline.isoformat(), command=job['command'],
              environment=job['environment'], plan_sha256=sha(PLAN),
              preparation_sha256=sha(tree.parent / 'preparation.json')))
        while child.poll() is None:
            if dt.datetime.now(dt.timezone.utc) >= deadline:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                write(tree.parent / (name + '-incomplete.json'),
                      dict(state='deadline_interrupted_not_eligible', observed=stamp()))
                raise RuntimeError('Deadline reached: incomplete native evidence preserved')
            time.sleep(10)
        if child.returncode:
            raise RuntimeError('Native job failed; original evidence preserved')
    trace_check(logpath.read_text(), job['epochs_per_phase'])
    shutil.copyfile(logpath, out / (name + '.out'))
    outputs = {str(p.relative_to(tree)): sha(p) for p in sorted(out.glob(name + '*')) if p.is_file()}
    if not set(job['original_registry_job']['outputs']).issubset(outputs):
        raise ValueError('Original declared native outputs incomplete')
    write(tree.parent / (name + '-receipt.json'), dict(state='native_complete_pending_export',
          observed=stamp(), job=name, outputs=outputs, log_sha256=sha(logpath),
          launch_sha256=sha(tree.parent / (name + '-launch.json'))))


def export(tree, name, dest):
    # Recheck source/data/binary, both traces, all outputs, then independently
    # reconstruct the known-row projection rather than trusting the saved map.
    plan = verify_inputs(tree)
    job = next(j for j in plan['jobs'] if j['name'] == name)
    receipt_path = tree.parent / (name + '-receipt.json')
    receipt = read(receipt_path)
    log = tree.parent / (name + '.log')
    launch_path = tree.parent / (name + '-launch.json')
    launch = read(launch_path)
    if (receipt['job'] != name or receipt['state'] != 'native_complete_pending_export'
            or receipt['log_sha256'] != sha(log)
            or receipt['launch_sha256'] != sha(launch_path)
            or launch['plan_sha256'] != sha(PLAN)
            or launch['preparation_sha256'] != sha(tree.parent / 'preparation.json')
            or launch['command'] != job['command'] or launch['environment'] != job['environment']):
        raise ValueError('Native receipt/launch binding differs')
    trace_check(log.read_text(), job['epochs_per_phase'])
    check_files(tree, receipt['outputs'])
    if not set(job['original_registry_job']['outputs']).issubset(receipt['outputs']):
        raise ValueError('Original outputs incomplete')
    for rel in receipt['outputs']:
        if rel.endswith('.probe_3.npy'):
            vector(tree / rel, 1760494)
        elif rel.endswith('.qual.npy'):
            vector(tree / rel, 2817131)
    mapping = projection(arrays(tree / 'data/probe_3', labels=True),
                         arrays(tree / 'data/probex', labels=True))
    if not np.array_equal(mapping, np.load(tree.parent / 'probex-row-map.npy')):
        raise ValueError('Independent row projection differs')
    for split in ['probe_3', 'qual']:
        final = vector(tree / f'preds_3/{name}.{split}.npy', 1760494 if split == 'probe_3' else 2817131)
        last_path = tree / f"preds_3/{name}_ep{job['epochs_per_phase']:02}.{split}.npy"
        if not last_path.exists():
            continue
        last = vector(last_path, len(final))
        if not np.array_equal(final.view(np.uint32), last.view(np.uint32)):
            raise ValueError('Terminal and last full epoch differ')
    if dest.exists():
        raise ValueError('Existing export preserved')
    dest.mkdir(parents=True)
    records = []
    for column in job['published_columns']:
        known = vector(tree / f'preds_3/{column}.probe_3.npy', 1760494)[mapping]
        qual = vector(tree / f'preds_3/{column}.qual.npy', 2817131)
        target = dest / (column + '.npz')
        np.savez_compressed(target, probex=known, qual=qual)
        with np.load(target, allow_pickle=False) as saved:
            for key, expected in [('probex', known), ('qual', qual)]:
                if not np.array_equal(saved[key].view(np.uint32), expected.view(np.uint32)):
                    raise ValueError('Lossless transfer differs')
        records.append(dict(column=column, file=target.name, sha256=sha(target),
                            native_probe_sha256=sha(tree / f'preds_3/{column}.probe_3.npy'),
                            native_qual_sha256=sha(tree / f'preds_3/{column}.qual.npy')))
    for path in [PLAN, receipt_path, launch_path, log, tree.parent / 'preparation.json',
                 tree.parent / 'build.log', tree.parent / 'split3.log', Path(__file__)]:
        shutil.copyfile(path, dest / path.name)
    write(dest / 'verification.json', dict(state='verified_complete_original_remote15h_producer',
          observed=stamp(), job=name, source_pin=plan['source_pin'], records=records,
          source_data_binary_rehashed=True, both_full_original_epoch_traces=True,
          independent_known_projection=True, qualifying_labels_read=False,
          public_prediction_values_read=False, files_sha256={p.name: sha(p) for p in dest.iterdir()},
          scope='Own numerical training outputs only. No blend or goal-success claim.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'verify-inputs', 'run-job', 'export'])
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--source-tree', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--job', choices=[j['name'] for j in read(PLAN)['jobs']])
    p.add_argument('--deadline-utc')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    tree = a.tree.expanduser().resolve()
    tree.parent.mkdir(parents=True, exist_ok=True)
    operation_lock = (tree.parent / 'operation.lock').open('a')
    fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.action == 'prepare':
        if not a.source_tree or not a.data_root:
            p.error('prepare requires --source-tree and --data-root')
        tree.parent.mkdir(parents=True, exist_ok=True)
        prepare(a.source_tree.expanduser().resolve(), a.data_root.expanduser().resolve(), tree)
    elif a.action == 'verify-inputs':
        verify_inputs(tree)
    elif a.action == 'run-job':
        if not a.job or not a.deadline_utc:
            p.error('run-job requires --job and --deadline-utc')
        deadline = dt.datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
        if deadline.utcoffset() != dt.timedelta(0):
            p.error('deadline must explicitly use UTC')
        run_job(tree, a.job, deadline)
    else:
        if not a.job or not a.output:
            p.error('export requires --job and --output')
        export(tree, a.job, a.output.expanduser().resolve())
    print(a.action + ': complete', flush=True)


if __name__ == '__main__':
    main()
