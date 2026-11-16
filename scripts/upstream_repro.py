#!/usr/bin/env python3
"""Prepare an exact upstream development reference and isolated fresh-training tree."""
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import tomli
from gbdt_blend import load

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/upstream-repro'
SOURCE = BASE/'source'
REFERENCE = BASE/'reference'
FRESH = BASE/'fresh'
PIN = '49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c'
URL = 'https://huggingface.co/buckets/gtakacs/netflix_prize/resolve/'


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    env = dict(os.environ)
    cargo = BASE/'toolchain/cargo'
    if (cargo/'bin/cargo').exists():
        env.update(CARGO_HOME=str(cargo), RUSTUP_HOME=str(BASE/'toolchain/rustup'))
        env['PATH'] = str(cargo/'bin')+os.pathsep+env.get('PATH', '')
    env.update(CARGO_TARGET_DIR=str(BASE/'target'), CARGO_BUILD_JOBS='2',
               RAYON_NUM_THREADS=os.environ.get('UPSTREAM_REPRO_THREADS', '2'),
               VECLIB_MAXIMUM_THREADS=os.environ.get('UPSTREAM_REPRO_BLAS_THREADS', '1'),
               OPENBLAS_NUM_THREADS=os.environ.get('UPSTREAM_REPRO_BLAS_THREADS', '1'))
    if sys.platform == 'darwin':
        # Upstream fit's blas-enabled ndarray calls need Accelerate linked
        # explicitly on macOS. This changes linking, not model source/math.
        env['RUSTFLAGS'] = (env.get('RUSTFLAGS', '')+' -C link-arg=-framework -C link-arg=Accelerate').strip()
    return env


def bootstrap(local_source=None):
    BASE.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        subprocess.run(['git', 'clone', str(local_source or 'https://github.com/gtakacs/netflix_prize.git'), str(SOURCE)], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', PIN], check=True)
    if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('Upstream checkout is not at the pinned commit')
    changed = subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--name-only', 'HEAD'], text=True)
    if changed.strip():
        raise ValueError('Upstream tracked source/configuration was modified')
    shutil.copyfile(ROOT/'scripts/upstream_registry.rs', SOURCE/'src/bin/repro_registry.rs')


def build():
    bootstrap()
    env = environment()
    cargo = shutil.which('cargo', path=env['PATH'])
    if not cargo:
        raise RuntimeError('Install Rust using rustup, then rerun --build; no global settings are modified by this script')
    command = [cargo, 'build', '--release', '--locked', '--features', 'blas']
    for binary in ['ridge', 'preds', 'run', 'fit', 'ingest', 'newsplit', 'download', 'dnn', 'tsvdx4-new', 'rbmx2-new', 'repro_registry']:
        command += ['--bin', binary]
    with (BASE/'build-verified.log').open('w') as log:
        subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    registry = subprocess.check_output([str(BASE/'target/release/repro_registry')], cwd=SOURCE, env=env)
    (BASE/'registry.json').write_bytes(registry)
    write(BASE/'build-receipt.json', dict(source_commit=PIN,
          cargo=subprocess.check_output([cargo, '--version'], env=env, text=True).strip(),
          rustflags=env.get('RUSTFLAGS', ''), command=command,
          binaries={p.name: sha(p) for p in (BASE/'target/release').iterdir() if p.is_file() and os.access(p, os.X_OK)}))


def prepare_reference(index_path, dataset_path=None, mapping_path=None, cache_root=None, jobs=4, ratings_path=None):
    registry = read(BASE/'registry.json')
    with open(index_path, 'rb') as stream:
        index = {e['path']: e for e in tomli.load(stream)['file']}
    if ratings_path:
        y = np.load(ratings_path, allow_pickle=False)
        users = np.load(ratings_path.parent/'user_idxs.npy', allow_pickle=False)
        provenance = dict(ratings_sha256=sha(ratings_path), users_sha256=sha(ratings_path.parent/'user_idxs.npy'))
    else:
        d = load(dataset_path)
        mapping = np.fromfile(mapping_path, dtype='<u4')
        n = len(d['probe'])
        if len(mapping) != n or not np.array_equal(np.sort(mapping), np.arange(n)):
            raise ValueError('ProbeX source-order mapping is not a bijection')
        y, users = np.empty(n, dtype=np.int8), np.empty(n, dtype=np.int32)
        y[mapping], users[mapping] = d['probe']['rating'], d['probe']['user']
        provenance = dict(dataset_sha256=sha(dataset_path), mapping_sha256=sha(mapping_path))
    n = len(y)
    if n != 1128164 or users.shape != y.shape or not np.isin(y, [1, 2, 3, 4, 5]).all():
        raise ValueError('Expected exact upstream ProbeX known-rating rows')
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for name in ['models-new.toml', 'pipeline-new.toml', 'ensembles.toml']:
        shutil.copyfile(SOURCE/name, REFERENCE/name)
    (REFERENCE/'data/probex').mkdir(parents=True, exist_ok=True)
    np.save(REFERENCE/'data/probex/ratings.npy', y, allow_pickle=False)
    np.save(REFERENCE/'data/probex/user_idxs.npy', users, allow_pickle=False)
    entries = []
    def fetch(name):
        key = f'preds_new/{name}.probex.npy'
        entry = index[key]
        destination = REFERENCE/key
        destination.parent.mkdir(parents=True, exist_ok=True)
        def valid(path):
            return path.is_file() and path.stat().st_size == entry['size'] and sha(path, 'md5') == entry['md5']
        if not valid(destination):
            if destination.exists():
                raise ValueError(f'Existing fixture differs from pinned index: {destination}')
            candidates = []
            if cache_root:
                candidates = [cache_root/'artifacts/public-new/columns'/f'{name}.probex.npy',
                    cache_root/'artifacts/public-new/support'/f'{Path(name).name}.probex.npy',
                    cache_root/'artifacts/public-probe-blenders-expanded/source'/key,
                    cache_root/'artifacts/public-probe-blenders/source'/key]
            cached = next((p for p in candidates if valid(p)), None)
            if cached:
                link_or_copy(cached, destination)
            else:
                temp = destination.with_name(destination.name+'.download')
                with urllib.request.urlopen(URL+key, timeout=90) as source, temp.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                if not valid(temp):
                    raise ValueError(f'Download checksum mismatch: {key}')
                temp.replace(destination)
        value = np.load(destination, mmap_mode='r', allow_pickle=False)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid source fixture {key}')
        return dict(name=name, path=str(destination.relative_to(REFERENCE)),
                    rows=n, md5=entry['md5'], sha256=sha(destination), provenance='released reference fixture, not trained locally')
    auxiliary = ['dnn-24', 'dnn-24__dnn-16p50']
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        entries = list(pool.map(fetch, registry['names']))
        auxiliary_entries = list(pool.map(fetch, auxiliary))
    if list(REFERENCE.glob('preds_new/**/*.qual.npy')):
        raise ValueError('Qualifying predictions must stay outside the development reference directory')
    manifest = dict(source_commit=PIN, index_sha256=sha(index_path), **provenance,
                    rows=n, columns=len(entries),
                    labels='ProbeX development labels only', qualifying_labels_read=False,
                    entries=entries, auxiliary_entries=auxiliary_entries,
                    unavailable_reference_jobs=['rbmx2-40'],
                    groups=registry['groups'], clip=registry['clip'])
    write(BASE/'reference-manifest.json', manifest)
    print(json.dumps(dict(stage='reference-prepared', rows=n, columns=len(entries))), flush=True)


def python_reference():
    manifest = read(BASE/'reference-manifest.json')
    y = np.load(REFERENCE/'data/probex/ratings.npy').astype(np.float64)
    arrays = [np.load(REFERENCE/e['path'], mmap_mode='r', allow_pickle=False) for e in manifest['entries']]
    width = len(arrays)+1
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    for lo in range(0, len(y), 8192):
        hi = min(lo+8192, len(y))
        x = np.column_stack([a[lo:hi] for a in arrays]).astype(np.float64)
        for j, clip in enumerate(manifest['clip']):
            if clip:
                x[:, j] = np.clip(x[:, j], 0, 6)
        z = np.column_stack([x, np.ones(len(x))])
        gram += z.T@z
        rhs += z.T@y[lo:hi]
    with (REFERENCE/'ensembles.toml').open('rb') as f:
        definition = next(e for e in tomli.load(f)['ensemble'] if e['name'] == 'new')
    records = []
    for name, columns in [*manifest['groups'].items(), ('ensemble', list(range(len(arrays))))]:
        selected = [*columns, len(arrays)]
        g = gram[np.ix_(selected, selected)].copy()
        g[np.arange(len(columns)), np.arange(len(columns))] += definition['lambda']
        w = np.linalg.solve(g, rhs[selected])
        pred = np.empty(len(y))
        for lo in range(0, len(y), 8192):
            hi = min(lo+8192, len(y))
            x = np.column_stack([arrays[j][lo:hi] for j in columns]).astype(np.float64)
            for k, j in enumerate(columns):
                if manifest['clip'][j]:
                    x[:, k] = np.clip(x[:, k], 0, 6)
            pred[lo:hi] = np.clip(np.column_stack([x, np.ones(len(x))])@w, 1.05, 4.95)
        rmse = float(np.sqrt(np.mean((pred-y)**2)))
        expected = definition['expected'][name]['probe']
        records.append(dict(group=name, columns=len(columns), rmse=rmse,
                            expected=expected, delta=rmse-expected,
                            reproduced=abs(rmse-expected) <= definition['tolerance']))
        if name == 'ensemble':
            pred.astype('<f4').tofile(BASE/'reference-probe.f32')
            np.save(BASE/'reference-weights.npy', w, allow_pickle=False)
    result = dict(state='complete', all_reference_rows_reproduced=all(r['reproduced'] for r in records),
                  implementation='Independent NumPy implementation of pinned upstream ridge/clipping/regularization',
                  qualifying_labels_read=False, records=records,
                  limitation='In-sample development-fitted blend reference, not independent CV or qualifying performance')
    write(BASE/'python-reference.json', result)
    if not result['all_reference_rows_reproduced']:
        raise ValueError('Independent reference does not match all stored development rows')
    print(json.dumps(result, indent=2), flush=True)


def native_reference():
    env = environment()
    command = [str(BASE/'target/release/ridge'), '--ensemble', 'new']
    result = subprocess.run(command, cwd=REFERENCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (BASE/'native-reference.log').write_text(result.stdout)
    if result.returncode or 'Reference check: PASS' not in result.stdout or 'Quiz set:  skipped' not in result.stdout:
        raise ValueError('Native reference failed or unexpectedly accessed qualifying evaluation; inspect native-reference.log')
    write(BASE/'native-reference.json', dict(state='verified', source_commit=PIN,
          command=command, binary_sha256=sha(command[0]), log_sha256=sha(BASE/'native-reference.log'),
          qualifying_labels_read=False, log=result.stdout))
    print(result.stdout, flush=True)


def training_layout(archive):
    if FRESH.exists():
        raise ValueError('Fresh-training checkout already exists; preserve it rather than overwrite')
    subprocess.run(['git', 'clone', '--local', str(SOURCE), str(FRESH)], check=True)
    subprocess.run(['git', '-C', str(FRESH), 'checkout', '--detach', PIN], check=True)
    # Replace the bundled recovered-label payload before any upstream ingestion.
    path = FRESH/'data/qual_ratings/qual_ratings.csv.gz'
    with path.open('wb') as binary, gzip.GzipFile(fileobj=binary, mode='wb', mtime=0) as f:
        f.write(b'rating,is_test\n')
        remaining = 2817131
        while remaining:
            count = min(remaining, 65536)
            f.write(b'0,0\n'*count)
            remaining -= count
    if archive:
        if sha(archive, 'md5') != 'a8f23d2d76461211c6b4c0ca6df2547d':
            raise ValueError('Raw Netflix archive checksum differs')
        destination = FRESH/'data/raw/nf_prize_dataset.tar.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(archive, destination)
    (FRESH/'target').symlink_to(BASE/'target', target_is_directory=True)
    registry = read(BASE/'registry.json')
    initial = ['tsvdx4-60', 'tsvdx4-60__knns', 'rbmx2-40', 'rbmx2-40__knn3',
               'dnn-24', 'dnn-24__knn3', 'dnn-24__dnn-16p50', 'dnn-24__dnn-16p50__knn3']
    for name in initial:
        if name not in registry['jobs']:
            raise ValueError(f'Unresolved pinned upstream recipe: {name}')
    write(BASE/'fresh-training-plan.json', dict(source_commit=PIN, stage='prepared-not-launched',
          labels='Original known training/development ratings; all qualifying ratings replaced by zero sentinels',
          synthetic_qualifying_payload_sha256=sha(path),
          source_model_code_modified=False, reference_predictions_accessible_as_training_inputs=False,
          initial_jobs=[dict(name=n, **registry['jobs'][n]) for n in initial],
          all_jobs=registry['jobs'], environment={k:v for k,v in environment().items()
          if k in ['CARGO_TARGET_DIR','CARGO_BUILD_JOBS','RAYON_NUM_THREADS','RUSTFLAGS']},
          caveats=['Legacy epoch-blend jobs embed upstream development-fitted coefficients; reproduce as reference, refit on local development predictions for strict fresh-fit attribution',
                   'Released fixtures remain in a separate directory and must not satisfy fresh model dependencies',
                   'The pinned single-split reference score is fitted on development labels; matching it is numerical reproduction, not held-out performance']))
    print('Fresh-training directory and exact eight-job recipe plan prepared; no base-model training launched.', flush=True)


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def validate_fresh():
    plan = read(BASE/'fresh-training-plan.json')
    head = subprocess.check_output(['git', '-C', str(FRESH), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(FRESH), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
    if head != PIN or set(changes) - {'data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Fresh source or configuration differs from the pinned upstream recipe')
    if sha(FRESH/'data/qual_ratings/qual_ratings.csv.gz') != plan['synthetic_qualifying_payload_sha256']:
        raise ValueError('Fresh qualifying payload is not the frozen zero sentinel')
    ratings = FRESH/'data/qual/ratings.npy'
    if ratings.exists() and np.any(np.load(ratings, mmap_mode='r', allow_pickle=False)):
        raise ValueError('Fresh qualifying ratings contain outcomes')
    return plan


def setup_data():
    validate_fresh()
    command = [str(BASE/'target/release/run'), '-n', '--setup']
    with (BASE/'fresh-setup.log').open('w') as log:
        subprocess.run(command, cwd=FRESH, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    validate_fresh()
    y = np.load(FRESH/'data/probex/ratings.npy', allow_pickle=False)
    if (REFERENCE/'data/probex/ratings.npy').exists():
        if not np.array_equal(y, np.load(REFERENCE/'data/probex/ratings.npy', allow_pickle=False)):
            raise ValueError('Native ProbeX row order/labels differ from the reference')
    write(BASE/'fresh-data-receipt.json', dict(source_commit=PIN, rows=len(y),
          ratings_sha256=sha(FRESH/'data/probex/ratings.npy'), qualifying_outcomes_zero=True))


def dependency_order(jobs, name):
    """Include implicit file producers as well as upstream inputs_from edges."""
    producers = {output: key for key, job in jobs.items() for output in job['outputs']}
    done, active, ordered = set(), set(), []
    def visit(key):
        if key in active:
            raise ValueError('Cycle in upstream job dependencies')
        if key in done:
            return
        if key not in jobs:
            raise ValueError(f'Unknown upstream job: {key}')
        active.add(key)
        job = jobs[key]
        parents = list(job['inputs_from'])
        parents += [producers[f] for f in job['inputs'] if f in producers and producers[f] != key]
        for parent in parents:
            visit(parent)
        active.remove(key)
        done.add(key)
        ordered.append(key)
    visit(name)
    return ordered


def model_job(name, execute=False):
    plan = validate_fresh()
    jobs = plan['all_jobs']
    order = [n for n in dependency_order(jobs, name) if n not in {'download', 'ingest', 'newsplit'}]
    print(json.dumps(dict(requested=name, order=order, commands=[jobs[n]['command'] for n in order]), indent=2), flush=True)
    if not execute:
        return
    if not (BASE/'fresh-data-receipt.json').exists():
        raise ValueError('Run --setup-data before model training')
    receipts = BASE/'fresh-receipts'
    receipts.mkdir(exist_ok=True)
    for key in order:
        validate_fresh()
        job = jobs[key]
        receipt = receipts/(key+'.json')
        outputs = [FRESH/f for f in job['outputs']]
        if receipt.exists():
            previous = read(receipt)
            if not all((FRESH/path).is_file() and sha(FRESH/path) == digest
                       for path, digest in previous['outputs'].items()):
                raise ValueError(f'Previously trained outputs changed: {key}')
            continue
        if any(p.exists() for p in outputs):
            raise ValueError(f'Unreceipted outputs already exist for {key}; preserve and inspect them before proceeding')
        predictions = FRESH/'preds_new'
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in predictions.rglob('*') if p.is_file()} if predictions.exists() else {}
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        with (receipts/(key+'.log')).open('w') as log:
            subprocess.run([str(BASE/'target/release/run'), '-n', key], cwd=FRESH,
                           env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
        generated = [p for p in predictions.rglob('*') if p.is_file()
                     and before.get(str(p)) != (p.stat().st_size, p.stat().st_mtime_ns)]
        outputs = sorted(set(outputs + generated))
        write(receipt, dict(source_commit=PIN, job=key, command=job['command'], started=started,
              finished=dt.datetime.now(dt.timezone.utc).isoformat(),
              outputs={str(p.relative_to(FRESH)): sha(p) for p in outputs},
              provenance='Fresh execution of pinned upstream source; no released prediction inputs'))
        print('Finished '+key, flush=True)


def compare_model(name):
    validate_fresh()
    receipt = BASE/'fresh-receipts'/(name+'.json')
    if not receipt.exists():
        raise ValueError('Comparison requires a fresh execution receipt')
    output = FRESH/'preds_new'/(name+'.probex.npy')
    if sha(output) != read(receipt)['outputs'][str(output.relative_to(FRESH))]:
        raise ValueError('Fresh output changed since its execution receipt')
    fresh = np.load(output, allow_pickle=False).astype(np.float64)
    reference_path = REFERENCE/'preds_new'/(name+'.probex.npy')
    reference = np.load(reference_path, allow_pickle=False).astype(np.float64) if reference_path.exists() else None
    y = np.load(FRESH/'data/probex/ratings.npy', allow_pickle=False)
    if not np.array_equal(y, np.load(REFERENCE/'data/probex/ratings.npy', allow_pickle=False)):
        raise ValueError('Development label order differs')
    if fresh.shape != y.shape or not np.isfinite(fresh).all():
        raise ValueError('Invalid fresh predictions')
    score = lambda p: float(np.sqrt(np.mean((np.clip(p, 1.05, 4.95)-y)**2)))
    result = dict(job=name, rows=len(y), fresh_rmse=score(fresh),
                  released_rmse=score(reference) if reference is not None else None,
                  reference_available=reference is not None,
                  prediction_rmse_difference=float(np.sqrt(np.mean((fresh-reference)**2))) if reference is not None else None,
                  max_absolute_difference=float(np.max(np.abs(fresh-reference))) if reference is not None else None,
                  fresh_sha256=sha(output), reference_sha256=sha(reference_path) if reference is not None else None,
                  scope='ProbeX diagnostic only; model training is fresh, reference vectors are imported')
    write(BASE/'comparisons'/(name+'.json'), result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bootstrap', action='store_true')
    ap.add_argument('--local-source', type=Path)
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--prepare-reference', action='store_true')
    ap.add_argument('--index', type=Path)
    ap.add_argument('--dataset', type=Path)
    ap.add_argument('--mapping', type=Path)
    ap.add_argument('--ratings-npy', type=Path, help='Native fresh/data/probex/ratings.npy; no mapping required')
    ap.add_argument('--cache-root', type=Path)
    ap.add_argument('--python-reference', action='store_true')
    ap.add_argument('--native-reference', action='store_true')
    ap.add_argument('--prepare-training', action='store_true')
    ap.add_argument('--archive', type=Path)
    ap.add_argument('--setup-data', action='store_true')
    ap.add_argument('--plan-job')
    ap.add_argument('--run-job')
    ap.add_argument('--compare-model')
    a = ap.parse_args()
    if a.bootstrap:
        bootstrap(a.local_source)
    if a.build:
        build()
    if a.prepare_reference:
        if not a.index or not (a.ratings_npy or (a.dataset and a.mapping)):
            ap.error('Reference preparation needs --index and either --ratings-npy or both --dataset and --mapping')
        prepare_reference(a.index, a.dataset, a.mapping, a.cache_root, ratings_path=a.ratings_npy)
    if a.python_reference:
        python_reference()
    if a.native_reference:
        native_reference()
    if a.prepare_training:
        training_layout(a.archive)
    if a.setup_data:
        setup_data()
    if a.plan_job:
        model_job(a.plan_job)
    if a.run_job:
        model_job(a.run_job, execute=True)
    if a.compare_model:
        compare_model(a.compare_model)


if __name__ == '__main__':
    main()
