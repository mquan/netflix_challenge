#!/usr/bin/env python3
"""Losslessly archive an explicit set of inactive legacy build caches."""
import argparse
from pathlib import Path
import shutil
import stat
import upstream_archive_shared_build_cache as archive

u = archive.u
REPORT = u.ROOT / 'experiments/upstream-legacy-build-cache-archive'
NAMES = (
    'full-upstream-mapped', 'upstream-aex-backfill',
    'upstream-cfnade-checkpoint', 'upstream-cfnade-fast',
    'upstream-cfnade-fast-v2', 'upstream-cfnade-sparse-rows',
    'upstream-cfnade96-sparse-resume', 'upstream-grouped-storage',
    'upstream-parallel-mapped', 'upstream-qualifying-backfill',
)
PROTECTED = (
    'upstream-own14-matched391-calibration/frozen.json',
    'upstream-own14-matched391-calibration/independent-preaudit-verification.json',
    'upstream-own14-matched391-post-native-audit/plan.json',
    'upstream-complete-cf-delta/plan.json',
    'upstream-original1200/execution-plan.json',
    'upstream-original1200-exact-build-recovery/plan.json',
    'upstream-complete-trajectory-data-queue/plan.json',
    'upstream-complete-trajectory-neural-queue/plan.json',
    'upstream-complete-trajectory-fold-bag-queue/plan.json',
    'upstream-own15-complete-trajectory-queue/plan.json',
    'upstream-weekend-trajectory-data-queue/plan.json',
    'upstream-weekend-trajectory-neural-queue/plan.json',
    'upstream-weekend-trajectory-fold-bag-queue/plan.json',
    'upstream-own16-weekend-trajectory-queue/plan.json',
    'upstream-remote-weekend-return-queue/s3-rbm400pf/plan.json',
    'upstream-remote-weekend-return-queue/s3-tx300r/plan.json',
    'upstream-remote-weekend-return-queue/s3-rbm800p/plan.json',
    'upstream-complete13-calibration/independent-canonical-verification.json',
    'upstream-own391-paired-h128-mps/independent-canonical-verification.json',
    'upstream-own391-fold-bag/independent-verification.json',
)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from strings(key)
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def inventory():
    protected = [u.ROOT / 'experiments' / name for name in PROTECTED]
    records = [u.read(path) for path in protected]
    roots = [(u.ROOT / 'artifacts' / name / 'target/release/build').resolve()
             for name in NAMES]
    if len(roots) != len(set(roots)):
        raise ValueError('Distinct canonical cache roots required')
    # Resolve aliases as well as canonical paths in every nested binding.
    for record in records:
        for text in strings(record):
            if not (text.startswith('artifacts/') or text.startswith(str(u.ROOT) + '/')):
                continue
            path = Path(text)
            resolved = (path if path.is_absolute() else u.ROOT / path).resolve()
            if any(resolved == root or root in resolved.parents for root in roots):
                raise ValueError('Frozen dependency references a proposed cache: ' + text)
    rows = []
    for name, root in zip(NAMES, roots):
        if root != u.ROOT / 'artifacts' / name / 'target/release/build':
            raise ValueError('Unexpected canonical root: ' + str(root))
        files = []
        for path in sorted(root.rglob('*')):
            if path.is_dir() and not path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Only regular single-link cache files: ' + str(path))
            files.append(path)
        if not files:
            raise ValueError('Nonempty original inventory required; never retry partial retirement')
        rows.append(dict(name=name, cache=str(root.relative_to(u.ROOT)),
                         files=len(files), bytes=sum(p.stat().st_size for p in files)))
    protected.extend([Path(__file__).resolve(), Path(archive.__file__).resolve()])
    return rows, {str(p.relative_to(u.ROOT)): u.sha(p) for p in protected}


def main(execute=False):
    if REPORT.exists():
        raise ValueError('Previous archive evidence preserved; no automatic retry')
    rows, protected = inventory()
    print('Explicit legacy caches:', len(rows), 'files:', sum(r['files'] for r in rows),
          'bytes:', sum(r['bytes'] for r in rows), flush=True)
    if not execute:
        return
    REPORT.mkdir()
    free_before = shutil.disk_usage(u.ROOT).free
    u.write(REPORT / 'selection.json', dict(caches=rows, protected_sha256=protected,
        procedure_sha256=u.sha(Path(archive.__file__)),
        rule='Reuse unchanged shared-cache archive procedure per explicit legacy cache. '
             'All payload bytes independently replay before unlink; preserve source, '
             'models, data, predictions, executables, original metadata and archives.'))
    archive.durable(REPORT / 'selection.json')
    archive.durable(REPORT)
    archive.durable(REPORT.parent)
    completed = []
    for row in rows:
        for relative, expected in protected.items():
            if u.sha(u.ROOT / relative) != expected:
                raise ValueError('Original protected evidence changed')
        archive.CACHE = u.ROOT / row['cache']
        archive.ARCHIVE = u.ROOT / 'artifacts' / (row['name'] + '-legacy-build-cache.tar.gz')
        archive.REPORT = REPORT / row['name']
        archive.main()
        proof = archive.REPORT / 'verification.json'
        completed.append(dict(name=row['name'], verification_sha256=u.sha(proof),
                              verification=u.read(proof)))
        print('Archived with full payload replay:', row['name'], flush=True)
    for relative, expected in protected.items():
        if u.sha(u.ROOT / relative) != expected:
            raise ValueError('Protected evidence changed after archive')
    u.write(REPORT / 'verification.json', dict(
        state='complete_all10_lossless_legacy_build_cache_archives_and_protected_evidence',
        selection_sha256=u.sha(REPORT / 'selection.json'), caches=completed,
        protected_sha256=protected, free_before_bytes=free_before,
        free_after_bytes=shutil.disk_usage(u.ROOT).free,
        numerical_files_removed=False, original_source_files_removed=False,
        same_native_training_preserved=True))
    archive.durable(REPORT / 'verification.json')
    archive.durable(REPORT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
