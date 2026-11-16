#!/usr/bin/env python3
"""Publish fresh upstream results and independently verify them on the first Mac."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np
import upstream_repro as upstream

ROOT = upstream.ROOT
OUT = ROOT/'experiments/upstream-results'
LOCAL = ROOT/'artifacts/upstream-coordinator'
REPORT = ROOT/'experiments/reproduction-coordination'
WORKER_BRANCH = 'research/upstream-worker'
COORDINATOR_BRANCH = 'research/upstream-reproduction'
JOBS = ['tsvdx4-60', 'tsvdx4-60__knns', 'rbmx2-40', 'rbmx2-40__knn3',
        'dnn-24', 'dnn-24__knn3', 'dnn-24__dnn-16p50', 'dnn-24__dnn-16p50__knn3']
ARMS = {'base-only': ['tsvdx4-60', 'rbmx2-40', 'dnn-24'],
        'base-plus-chains': JOBS,
        'temporal-base': ['tsvdx4-60'], 'temporal-chain': JOBS[:2],
        'rbm-base': ['rbmx2-40'], 'rbm-chain': JOBS[2:4],
        'dnn-base': ['dnn-24'], 'dnn-chain': JOBS[4:]}
PLAN = dict(version=1, source_commit=upstream.PIN, jobs=JOBS, arms=ARMS,
            folds=5, fold_seed=20261116, grouping='native upstream user index',
            ridge_lambda=1000.0, input_clip=[0, 6], output_clip=[1.05, 4.95],
            intercept='unpenalized', tuning='none; all arms/lambda/folds fixed before fresh results',
            metric='ProbeX development RMSE',
            limitation='Meta-level user-fold cross-fitting only; published base recipes were historically development-selected. Not untouched test evidence.',
            artifact_scope='Fresh ProbeX predictions, known development labels and user indices only; no qualifying predictions or outcomes')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(command, **kwargs):
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def git(*args, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check, timeout=180,
                          env={**os.environ, 'GIT_TERMINAL_PROMPT':'0'})


def branch_guard(expected):
    if git('branch', '--show-current').stdout.strip() != expected:
        raise ValueError('This role requires branch '+expected)


def publish(expected, paths, message, enabled):
    if not enabled:
        return
    branch_guard(expected)
    git('add', '--', *paths)
    if git('diff', '--cached', '--quiet', '--', *paths, check=False).returncode:
        git('-c', 'user.email=mquannie@gmail.com', 'commit', '--only', '-m', message, '--', *paths)
    # Never merge, rebase, force-push, or change another role's branch here.
    result = git('push', 'origin', 'HEAD:refs/heads/'+expected, check=False)
    if result.returncode:
        raise RuntimeError('Progress push failed; local artifacts retained: '+result.stderr[-2000:])


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metric(prediction, labels):
    return float(np.sqrt(np.mean((np.clip(np.asarray(prediction, dtype=np.float64), 1.05, 4.95)-np.asarray(labels, dtype=np.float64))**2)))


def user_folds(users):
    # Arithmetic on arrays wraps modulo 2**64, independently of row order.
    x = np.asarray(users, dtype=np.uint64) + np.uint64(PLAN['fold_seed'])
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(PLAN['folds'])).astype(np.int8)


def crossfit(matrix, labels, folds):
    """Each scored user's labels are excluded from that fold's ridge solve."""
    x = np.column_stack([np.clip(np.asarray(matrix, dtype=np.float64), 0, 6), np.ones(len(labels))])
    labels = np.asarray(labels, dtype=np.float64)
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    ridge = np.eye(x.shape[1])*PLAN['ridge_lambda']
    ridge[-1, -1] = 0
    for fold in range(PLAN['folds']):
        test = folds == fold
        train = ~test
        if not test.any() or not train.any():
            raise ValueError('Empty development training/scoring fold')
        a, y = x[train], labels[train]
        weights = np.linalg.solve(a.T@a+ridge, a.T@y)
        prediction[test] = np.clip(x[test]@weights, 1.05, 4.95)
    return prediction


def evaluate(predictions, labels, users, output_dir):
    if set(predictions) != set(JOBS):
        raise ValueError('Fresh-only evaluation requires exactly the eight planned jobs')
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = user_folds(users)
    records = []
    for name, columns in ARMS.items():
        value = crossfit(np.column_stack([predictions[c] for c in columns]), labels, folds)
        destination = output_dir/(name+'.f32')
        value.astype('<f4').tofile(destination)
        records.append(dict(arm=name, columns=columns, rmse=metric(value, labels),
                            fold_rmse=[metric(value[folds == f], labels[folds == f]) for f in range(5)],
                            prediction_sha256=upstream.sha(destination)))
    scores = {r['arm']: r for r in records}
    pairs = [('base-only', 'base-plus-chains'), ('temporal-base', 'temporal-chain'),
             ('rbm-base', 'rbm-chain'), ('dnn-base', 'dnn-chain')]
    contrasts = [dict(base=a, treatment=b, delta_rmse=scores[b]['rmse']-scores[a]['rmse'],
                      fold_delta=[b-a for a,b in zip(scores[a]['fold_rmse'], scores[b]['fold_rmse'])])
                 for a,b in pairs]
    return dict(plan=PLAN, rows=len(labels), records=records, contrasts=contrasts,
                input_provenance='Only fresh worker prediction artifacts; released reference predictions excluded from every fit')


def export_job(name):
    upstream.validate_fresh()
    receipt = upstream.read(upstream.BASE/'fresh-receipts'/(name+'.json'))
    file = upstream.FRESH/'preds_new'/(name+'.probex.npy')
    if upstream.sha(file) != receipt['outputs'][str(file.relative_to(upstream.FRESH))]:
        raise ValueError('Fresh output changed before export')
    value = np.load(file, allow_pickle=False)
    if value.dtype != np.dtype('float32') or value.shape != (1128164,) or not np.isfinite(value).all():
        raise ValueError('Unexpected native prediction type, row count or values')
    destination = OUT/(name+'.npz')
    np.savez_compressed(destination, prediction=value)
    comparison = upstream.read(upstream.BASE/'comparisons'/(name+'.json'))
    upstream.write(OUT/(name+'.receipt.json'), receipt)
    shutil.copyfile(upstream.BASE/'fresh-receipts'/(name+'.log'), OUT/(name+'.log'))
    return dict(job=name, artifact=destination.name, artifact_sha256=upstream.sha(destination),
                prediction_bytes_sha256=array_sha(value), source_npy_sha256=upstream.sha(file),
                comparison=comparison)


def ensure_cargo():
    """Install a verified official installer into this experiment only if needed."""
    env = upstream.environment()
    if shutil.which('cargo', path=env['PATH']):
        return
    system, machine = platform.system(), platform.machine().lower()
    arch = {'arm64': 'aarch64', 'aarch64': 'aarch64', 'x86_64': 'x86_64', 'amd64': 'x86_64'}.get(machine)
    suffix = {'Darwin': 'apple-darwin', 'Linux': 'unknown-linux-gnu'}.get(system)
    if not arch or not suffix:
        raise ValueError('Install Rust manually on this platform, then resume')
    url = f'https://static.rust-lang.org/rustup/dist/{arch}-{suffix}/rustup-init'
    toolchain = upstream.BASE/'toolchain'
    toolchain.mkdir(parents=True, exist_ok=True)
    installer = toolchain/'rustup-init'
    with urllib.request.urlopen(url+'.sha256', timeout=60) as response:
        expected = response.read().decode().split()[0]
    with urllib.request.urlopen(url, timeout=120) as response, installer.open('wb') as destination:
        shutil.copyfileobj(response, destination)
    if len(expected) != 64 or upstream.sha(installer) != expected:
        raise ValueError('Official Rust installer checksum mismatch')
    installer.chmod(0o700)
    env.update(CARGO_HOME=str(toolchain/'cargo'), RUSTUP_HOME=str(toolchain/'rustup'))
    subprocess.run([str(installer), '-y', '--no-modify-path', '--profile', 'minimal',
                    '--default-toolchain', '1.100.0'], env=env, check=True)


def worker(pushing):
    branch_guard(WORKER_BRANCH)
    OUT.mkdir(parents=True, exist_ok=True)
    upstream.write(OUT/'meta-plan.json', PLAN)
    manifest = dict(version=1, source_commit=upstream.PIN, state='preparing', jobs={},
                    plan_sha256=upstream.sha(OUT/'meta-plan.json'), started=now(),
                    python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform())
    if (OUT/'manifest.json').exists():
        previous = upstream.read(OUT/'manifest.json')
        if previous['source_commit'] != upstream.PIN or previous['plan_sha256'] != manifest['plan_sha256']:
            raise ValueError('Existing worker output uses a different frozen plan')
        manifest = previous
    def progress(state, active=None, error=None):
        manifest.update(state=state, active=active, updated=now(), error=error)
        upstream.write(OUT/'manifest.json', manifest)
        (ROOT/'UPSTREAM_WORKER_STATUS.md').write_text(
            '# Upstream worker progress\n\n'+f'Updated: {manifest["updated"]}\n\n'
            +f'State: **{state}**. Completed: {len(manifest["jobs"])}/8. Active: {active or "none"}.\n\n'
            +'Qualifying outcomes are zero sentinels. Exported predictions are fresh ProbeX only.\n'
            +(f'\nError: {error}\n' if error else ''))
        publish(WORKER_BRANCH, ['experiments/upstream-results', 'UPSTREAM_WORKER_STATUS.md'],
                'upstream worker: '+state+' '+(active or ''), pushing)
    try:
        progress('preparing')
        upstream.bootstrap()
        ensure_cargo()
        if not (upstream.BASE/'build-receipt.json').exists():
            upstream.build()
        if not upstream.FRESH.exists():
            candidates = [ROOT/'data/raw/nf_prize_dataset.tar.gz', ROOT.parent/'netflix-independent/data/raw/nf_prize_dataset.tar.gz']
            archive = next((p for p in candidates if p.exists()), None)
            upstream.training_layout(archive)
        if not (upstream.BASE/'fresh-data-receipt.json').exists():
            upstream.setup_data()
        reference_manifest = upstream.BASE/'reference-manifest.json'
        if (not reference_manifest.exists() or upstream.read(reference_manifest).get('index_sha256')
                != upstream.sha(ROOT/'experiments/upstream-repro/reference-index.toml')):
            upstream.prepare_reference(ROOT/'experiments/upstream-repro/reference-index.toml',
                ratings_path=upstream.FRESH/'data/probex/ratings.npy')
        upstream.python_reference()
        upstream.native_reference()
        labels = np.load(upstream.FRESH/'data/probex/ratings.npy', allow_pickle=False)
        users = np.load(upstream.FRESH/'data/probex/user_idxs.npy', allow_pickle=False)
        if not np.array_equal(labels, np.load(upstream.REFERENCE/'data/probex/ratings.npy', allow_pickle=False)):
            raise ValueError('Fresh/reference development label order differs')
        np.savez_compressed(OUT/'development.npz', labels=labels, users=users)
        manifest.update(development_sha256=upstream.sha(OUT/'development.npz'),
                        labels_bytes_sha256=array_sha(labels), users_bytes_sha256=array_sha(users))
        for name in JOBS:
            progress('running', name)
            upstream.model_job(name, execute=True)
            upstream.compare_model(name)
            manifest['jobs'][name] = export_job(name)
            progress('running')
        progress('complete')
    except BaseException as error:
        # Preserve the native log and partial artifacts; no automatic deletion/retraining.
        try:
            progress('failed', manifest.get('active'), repr(error))
        except Exception:
            traceback.print_exc()
        raise


def worker_blob(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def ingest_worker(ref):
    prefix = 'experiments/upstream-results/'
    manifest = json.loads(worker_blob(ref, prefix+'manifest.json'))
    if manifest['source_commit'] != upstream.PIN:
        raise ValueError('Worker source commit differs')
    plan_bytes = worker_blob(ref, prefix+'meta-plan.json')
    if hashlib.sha256(plan_bytes).hexdigest() != manifest['plan_sha256'] or json.loads(plan_bytes) != PLAN:
        raise ValueError('Worker/meta-evaluation plan differs')
    incoming = LOCAL/'incoming'
    incoming.mkdir(parents=True, exist_ok=True)
    if not manifest.get('development_sha256'):
        return manifest, None, None, {}, []
    data = worker_blob(ref, prefix+'development.npz')
    if hashlib.sha256(data).hexdigest() != manifest['development_sha256']:
        raise ValueError('Worker development artifact hash differs')
    (incoming/'development.npz').write_bytes(data)
    with np.load(incoming/'development.npz', allow_pickle=False) as archive:
        labels, users = archive['labels'], archive['users']
    expected = np.load(upstream.REFERENCE/'data/probex/ratings.npy', allow_pickle=False)
    if not np.array_equal(labels, expected) or users.shape != labels.shape:
        raise ValueError('Worker labels/row order differ from the independently reproduced reference')
    if array_sha(labels) != manifest['labels_bytes_sha256'] or array_sha(users) != manifest['users_bytes_sha256']:
        raise ValueError('Worker label/user content digest differs')
    predictions, checks = {}, []
    reference_manifest = upstream.read(upstream.BASE/'reference-manifest.json')
    pinned = {e['name']:e for e in reference_manifest['entries']+reference_manifest.get('auxiliary_entries', [])}
    for name, entry in manifest['jobs'].items():
        if name not in JOBS or entry['artifact'] != name+'.npz':
            raise ValueError('Unexpected worker model/artifact name')
        destination = incoming/entry['artifact']
        if not destination.exists() or upstream.sha(destination) != entry['artifact_sha256']:
            destination.write_bytes(worker_blob(ref, prefix+entry['artifact']))
        if upstream.sha(destination) != entry['artifact_sha256']:
            raise ValueError('Prediction artifact hash mismatch')
        receipt = json.loads(worker_blob(ref, prefix+name+'.receipt.json'))
        if receipt['source_commit'] != upstream.PIN or receipt['job'] != name:
            raise ValueError('Wrong source/job in execution receipt')
        if receipt['outputs'].get('preds_new/'+name+'.probex.npy') != entry['source_npy_sha256']:
            raise ValueError('Exported output is not the receipted native prediction')
        with np.load(destination, allow_pickle=False) as archive:
            value = archive['prediction']
        if value.dtype != np.dtype('float32') or value.shape != labels.shape or not np.isfinite(value).all():
            raise ValueError('Malformed worker predictions')
        if array_sha(value) != entry['prediction_bytes_sha256']:
            raise ValueError('Decoded prediction digest mismatch')
        measured = metric(value.astype(np.float64), labels)
        if abs(measured-entry['comparison']['fresh_rmse']) > 1e-8:
            raise ValueError('Independent fresh score differs from worker score')
        check = dict(job=name, fresh_rmse=measured, artifact_verified=True,
                     source_execution='Worker receipt and log; execution itself is not remotely attested')
        if name in pinned:
            fixture_path = upstream.REFERENCE/pinned[name]['path']
            if upstream.sha(fixture_path) != pinned[name]['sha256']:
                raise ValueError('Pinned local reference fixture changed')
            fixture = np.load(fixture_path, allow_pickle=False).astype(np.float64)
            check.update(reference_available=True, released_rmse=metric(fixture, labels),
                         prediction_rmse_difference=float(np.sqrt(np.mean((value-fixture)**2))))
        else:
            if name != 'rbmx2-40':
                raise ValueError('Expected reference fixture unavailable')
            check.update(reference_available=False, note='No rbmx2-40 base fixture in pinned upstream store index')
        predictions[name] = value
        checks.append(check)
    return manifest, labels, users, predictions, checks


def local_summary(local_root):
    run_dir = local_root/'artifacts/independent-ensemble'
    status = upstream.read(run_dir/'status.json')
    complete = (run_dir/'assignment-complete.json').exists()
    if complete:
        integrity = REPORT/'local-ensemble/integrity.json'
        if not integrity.exists():
            assignment = upstream.read(run_dir/'assignment-complete.json')
            if upstream.sha(run_dir/'plan.json') != assignment['policy']['plan_sha256']:
                raise ValueError('Local frozen plan changed')
            plan = upstream.read(run_dir/'plan.json')
            for path, digest in {**plan['inputs'], **plan['code']}.items():
                if upstream.sha(path) != digest:
                    raise ValueError('Local frozen input/code changed: '+path)
            upstream.write(integrity, dict(plan_sha256=upstream.sha(run_dir/'plan.json'),
                           frozen_inputs_and_code_verified=True, files=len(plan['inputs'])+len(plan['code'])))
        if not (REPORT/'local-ensemble/comparison.json').exists():
            code = ['.venv/bin/python', 'scripts/local_ensemble_run.py', '--summarize']
            subprocess.run(code, cwd=local_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for file in ['comparison.json', 'assignment-complete.json', 'status.json']:
            destination = REPORT/'local-ensemble'/file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir/file, destination)
        # Recheck each assigned report against its original successful receipt.
        for name in ['baseline-tree','baseline-nn','snapshots-tree','snapshots-nn','calibration-tree','calibration-nn']:
            receipt = upstream.read(run_dir/'logs'/(name+'.receipt.json'))
            stage = next(s for s in upstream.read(run_dir/'plan.json')['stages'] if s['name'] == name)
            if receipt['returncode'] != 0 or upstream.sha(stage['report']) != receipt['report_sha256']:
                raise ValueError('Completed local stage receipt differs: '+name)
            upstream.write(REPORT/'local-ensemble'/(name+'.receipt.json'), receipt)
    return dict(complete=complete, state=status['state'], active=status.get('stage'))


def coordinate_once(local_root, pushing):
    branch_guard(COORDINATOR_BRANCH)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'failure.json').unlink(missing_ok=True)
    upstream.write(REPORT/'meta-plan.json', PLAN)
    local = local_summary(local_root)
    ref = 'refs/remotes/origin/'+WORKER_BRANCH
    fetched = git('fetch', 'origin', 'refs/heads/'+WORKER_BRANCH+':'+ref, check=False)
    if fetched.returncode:
        remote = dict(state='waiting-for-worker-branch', error=fetched.stderr[-1500:])
        checks = []
    else:
        remote, labels, users, predictions, checks = ingest_worker(ref)
        remote = dict(state=remote['state'], active=remote.get('active'), completed=len(predictions),
                      commit=git('rev-parse', ref).stdout.strip(), error=remote.get('error'))
        upstream.write(REPORT/'model-checks.json', checks)
        if set(predictions) == set(JOBS) and (REPORT/'fresh-only-blends.json').exists():
            frozen = upstream.read(REPORT/'fresh-only-blends.json')['input_artifact_sha256']
            if any(upstream.sha(LOCAL/'incoming'/(n+'.npz')) != frozen[n] for n in JOBS):
                raise ValueError('Fresh-only inputs changed after evaluation; investigate before refitting')
        if local['complete'] and set(predictions) == set(JOBS) and not (REPORT/'fresh-only-blends.json').exists():
            evaluation = evaluate(predictions, labels, users, LOCAL/'predictions')
            evaluation['worker_commit'] = remote['commit']
            evaluation['input_artifact_sha256'] = {n: upstream.sha(LOCAL/'incoming'/(n+'.npz')) for n in JOBS}
            upstream.write(REPORT/'fresh-only-blends.json', evaluation)
    state = dict(local=local, remote=remote, fresh_only_blends_complete=(REPORT/'fresh-only-blends.json').exists())
    # Exclude wall-clock/heartbeat timestamps from tracked status to avoid idle commits.
    upstream.write(REPORT/'status.json', state)
    lines = ['# Reproduction coordination', '',
             'Local machine: '+json.dumps(local)+'.', '', 'Upstream worker: '+json.dumps(remote)+'.', '',
             'Fresh-only development blends: '+('complete' if state['fresh_only_blends_complete'] else 'waiting for all eight fresh model artifacts')+'.', '',
             'The 0.856167 released-prediction reference is not a fresh-training result.']
    if checks:
        lines += ['', '| Model | Fresh development RMSE | Released fixture RMSE |', '| --- | ---: | ---: |']
        for check in checks:
            reference = f'{check["released_rmse"]:.9f}' if check['reference_available'] else 'unavailable'
            lines += [f'| {check["job"]} | {check["fresh_rmse"]:.9f} | {reference} |']
    (ROOT/'COORDINATION_STATUS.md').write_text('\n'.join(lines)+'\n')
    publish(COORDINATOR_BRANCH, ['experiments/reproduction-coordination','COORDINATION_STATUS.md'],
            'Record reproduction coordination and independent checks', pushing)
    return state


def coordinate(local_root, pushing, once, interval):
    LOCAL.mkdir(parents=True, exist_ok=True)
    while True:
        branch_guard(COORDINATOR_BRANCH)
        try:
            state = coordinate_once(local_root, pushing)
            upstream.write(LOCAL/'heartbeat.json', dict(updated=now(), pid=os.getpid(), **state))
            if state['local']['complete'] and state['fresh_only_blends_complete']:
                return
        except Exception as error:
            traceback.print_exc()
            upstream.write(LOCAL/'heartbeat.json', dict(updated=now(), pid=os.getpid(), error=repr(error)))
            upstream.write(REPORT/'failure.json', dict(error=repr(error), action='Preserve artifacts; verification will retry, no silent overwrite or fallback'))
            (ROOT/'COORDINATION_STATUS.md').write_text('# Reproduction coordination\n\nVerification error: '+repr(error)+'\n\nArtifacts are preserved; coordinator will retry.\n')
            try:
                publish(COORDINATOR_BRANCH, ['experiments/reproduction-coordination','COORDINATION_STATUS.md'],
                        'Report reproduction verification failure', pushing)
            except Exception:
                traceback.print_exc()
            if once:
                raise
        if once:
            return
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    role = ap.add_mutually_exclusive_group(required=True)
    role.add_argument('--worker', action='store_true')
    role.add_argument('--coordinator', action='store_true')
    ap.add_argument('--publish', action='store_true', help='Commit/push role-owned outputs to its designated branch')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=int, default=120)
    ap.add_argument('--local-root', type=Path, default=ROOT.parent/'netflix_independent_ensemble')
    a = ap.parse_args()
    branch_guard(WORKER_BRANCH if a.worker else COORDINATOR_BRANCH)
    lock_dir = upstream.BASE if a.worker else LOCAL
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir/('worker.lock' if a.worker else 'coordinator.lock')
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError('Role already running; inspect '+str(lock))
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    inhibitor = None
    try:
        if Path('/usr/bin/caffeinate').exists():
            inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        os.nice(10)
        if a.worker:
            worker(a.publish)
        else:
            coordinate(a.local_root.resolve(), a.publish, a.once, max(a.interval, 30))
    finally:
        lock.unlink(missing_ok=True)
        if inhibitor:
            inhibitor.terminate()


if __name__ == '__main__':
    main()
