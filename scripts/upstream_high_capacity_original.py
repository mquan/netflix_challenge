#!/usr/bin/env python3
"""Prepare isolated exact high-capacity presets; never launch training here."""
import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np
import upstream_repro as u

RUN = u.ROOT / 'artifacts/upstream-high-capacity-original'
TREE = RUN / 'source'
REPORT = u.ROOT / 'experiments/upstream-high-capacity-original'
JOBS = {'tsvdx4-1600lm': ('tsvdx4-new', 1600, 2),
        'tsvdx5-1200lm': ('tsvdx5-new', 1200, 23)}
COUNTS = {'trainx': 99352343, 'probex': 1128164,
          'fulltrain': 100480507, 'qual': 2817131}
ZERO_TARGET = 'data/qual_ratings/qual_ratings.csv.gz'


def tracked_files(tree):
    names = subprocess.check_output(['git', '-C', str(tree), 'ls-files', '-z'])
    return [tree / n.decode() for n in names.split(b'\0') if n]


def disjoint_training_queries(tree, train='trainx', query='probex'):
    """Check every training identity against every held query, bounded memory."""
    qu = np.load(tree / f'data/{query}/user_idxs.npy', mmap_mode='r')
    qi = np.load(tree / f'data/{query}/item_idxs.npy', mmap_mode='r')
    tu = np.load(tree / f'data/{train}/user_idxs.npy', mmap_mode='r')
    ti = np.load(tree / f'data/{train}/item_idxs.npy', mmap_mode='r')
    keys = np.sort(qu.astype(np.uint64) * np.uint64(17770) + qi.astype(np.uint64))
    if len(keys) != COUNTS[query] or np.any(keys[1:] == keys[:-1]):
        raise ValueError('Query identities are not unique/full')
    if len(tu) != COUNTS[train] or len(ti) != len(tu):
        raise ValueError('Complete training population required')
    for lo in range(0, len(tu), 262144):
        k = tu[lo:lo+262144].astype(np.uint64) * np.uint64(17770) + ti[lo:lo+262144].astype(np.uint64)
        at = np.searchsorted(keys, k)
        eligible = at < len(keys)
        if np.any(keys[at[eligible]] == k[eligible]):
            raise ValueError('A held query identity occurs in training')
    return dict(training_rows=len(tu), query_rows=len(keys),
                all_training_identities_scanned=True, overlap=0)


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    destination = REPORT / 'execution-plan.json'
    if destination.exists():
        plan = u.read(destination)
        for rel, digest in plan['bindings'].items():
            if u.sha(u.ROOT / rel) != digest:
                raise ValueError('Prepared original input changed: ' + rel)
        return plan
    u.validate_fresh()
    if subprocess.check_output(['git', '-C', str(u.SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != u.PIN:
        raise ValueError('Original source pin differs')
    if subprocess.check_output(['git', '-C', str(u.SOURCE), 'diff', 'HEAD', '--name-only'], text=True).strip():
        raise ValueError('Original tracked source differs')
    RUN.mkdir(exist_ok=True)
    if not TREE.exists():
        subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(u.SOURCE), str(TREE)], check=True)
        subprocess.run(['git', '-C', str(TREE), 'checkout', '--detach', u.PIN], check=True)
    if subprocess.check_output(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], text=True).strip() != u.PIN:
        raise ValueError('Isolated source pin differs')
    changes = subprocess.check_output(['git', '-C', str(TREE), 'diff', 'HEAD', '--name-only'], text=True).splitlines()
    if set(changes) - {ZERO_TARGET}:
        raise ValueError('Isolated tracked source differs')
    # The source repository tracks its qualifying CSV. Replace it with the
    # already validated synthetic zero sentinel before admitting any inputs.
    shutil.copyfile(u.FRESH / ZERO_TARGET, TREE / ZERO_TARGET)
    bindings = {}
    def bind(p):
        bindings[str(p.relative_to(u.ROOT))] = u.sha(p)
    for path in tracked_files(TREE):
        # Git submodules, if any, require separate complete source treatment.
        if not path.is_file():
            raise ValueError('Non-file tracked source requires explicit admission')
        expected = u.FRESH / ZERO_TARGET if path.relative_to(TREE).as_posix() == ZERO_TARGET else u.SOURCE / path.relative_to(TREE)
        if u.sha(path) != u.sha(expected):
            raise ValueError('Exact original source file differs')
        bind(path)
    (TREE / 'data').mkdir(exist_ok=True)
    for split, count in COUNTS.items():
        parent = u.FRESH / 'data' / split
        target = TREE / 'data' / split
        if not target.exists():
            target.symlink_to(parent, target_is_directory=True)
        if target.resolve() != parent.resolve():
            raise ValueError('Isolated dataset link differs')
        for path in sorted(parent.glob('*.npy')):
            bind(path)
        values = np.load(parent / 'ratings.npy', mmap_mode='r')
        if values.shape != (count,):
            raise ValueError('Full original row count differs: ' + split)
        if split == 'qual' and np.any(values):
            raise ValueError('Qualifying fitting targets must be zero')
    # Existing compiled binaries are recorded, not silently rebuilt under a
    # shared target directory while another native experiment is running.
    registry = u.read(u.BASE / 'registry.json')['jobs']
    jobs = {}
    for name, (binary, factors, epochs) in JOBS.items():
        path = u.BASE / 'target/release' / binary
        if not path.is_file():
            raise ValueError('Original compiled binary absent: ' + binary)
        bind(path)
        jobs[name] = dict(original_registry_job=registry[name], factors=factors,
                          epochs_per_phase=epochs, original_low_memory_preset=True,
                          native_command=[str(path), name],
                          cwd=str(TREE), binary_sha256=u.sha(path))
    if (TREE / 'preds_new').exists():
        # Original tracked md5 catalogs are metadata, not prediction inputs.
        # Do not mistake their directory for an existing trained output.
        tracked = set(tracked_files(TREE))
        for path in (TREE / 'preds_new').rglob('*'):
            if path.is_file() and (path not in tracked or path.suffix != '.md5'):
                raise ValueError('Unexpected existing isolated numerical output')
    checks = [disjoint_training_queries(TREE),
              disjoint_training_queries(TREE, 'fulltrain', 'qual')]
    for path in [Path(__file__), u.ROOT / 'HIGH_CAPACITY_ORIGINAL_PLAN.md',
                 u.ROOT / 'scripts/upstream_repro.py',
                 u.BASE / 'registry.json', u.BASE / 'fresh-training-plan.json',
                 u.BASE / 'fresh-data-receipt.json']:
        bind(path)
    plan = dict(state='complete_isolated_original_preparation_no_training_launched',
                source_pin=u.PIN, jobs=jobs, original_rows=COUNTS,
                full_identity_disjointness=checks, bindings=bindings,
                qualifying_targets_zero=True, qualifying_labels_read=False,
                downloaded_prediction_inputs=False,
                execution_gate='Separate guarded execution and independent binary/source/output verification required. This preparer never launches training. Do not use isolated binaries without a resource owner.',
                scope='Exact source, complete original data hashes, all training/query identities and compiled binary hashes bound. Compiled binary provenance still requires an independent source/build receipt check before execution.')
    u.write(destination, plan)
    return plan


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(prepare()['state'], flush=True)
