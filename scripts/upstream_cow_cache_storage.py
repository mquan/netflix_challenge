#!/usr/bin/env python3
"""Share exact inactive cache bytes with APFS clones, preserving logical files."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'experiments/upstream-cow-cache-storage'
CANDIDATES = ROOT / 'experiments/upstream-old-cache-storage/full-hash-candidates.json'
ALLOWED = {'upstream-phase-bank', 'upstream-recovered-temporal-bank',
           'upstream-remote-batch-pretraining-audit-format', 'upstream-remote-batch'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def identity(path):
    s = path.lstat()
    return dict(ino=s.st_ino, dev=s.st_dev, bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                mode=s.st_mode, uid=s.st_uid, gid=s.st_gid, nlink=s.st_nlink, flags=s.st_flags)


def validate_path(path, record):
    rel = path.relative_to(ROOT / 'artifacts')
    if len(rel.parts) != 2 or rel.parts[0] not in ALLOWED or path.suffix != '.npy':
        raise ValueError('Only the declared inactive cache copies may change')
    current = identity(path)
    if current != {k: record[k] for k in current}:
        raise ValueError('Candidate identity changed: ' + str(path))
    if not stat.S_ISREG(current['mode']) or current['nlink'] != 1 or current['flags']:
        raise ValueError('Need an unflagged regular file with one logical link')
    if subprocess.check_output(['/usr/bin/xattr', str(path)]):
        raise ValueError('Extended attributes require separate handling')
    acl = subprocess.check_output(['/bin/ls', '-lde', str(path)], text=True)
    if len(acl.splitlines()) != 1:
        raise ValueError('ACL requires separate handling')
    return current


def assert_closed(paths):
    result = subprocess.run(['/usr/sbin/lsof', '-Fpn', '--', *map(str, paths)],
                            capture_output=True, text=True)
    if result.returncode not in (0, 1) or result.stdout.strip() or result.stderr.strip():
        raise ValueError('Open handles or uncertain lsof observation: ' + result.stdout + result.stderr)


def same_bytes(a, b):
    with a.open('rb') as x, b.open('rb') as y:
        while True:
            left, right = x.read(4 * 1024 * 1024), y.read(4 * 1024 * 1024)
            if left != right:
                raise ValueError('Complete byte comparison differs')
            if not left:
                return


def main():
    if REPORT.exists():
        raise ValueError('Existing storage evidence preserved; no automatic repeat')
    candidates = json.loads(CANDIDATES.read_text())
    fixture_path = ROOT / 'experiments/upstream-storage-compression-fixture/clone-verification.json'
    fixture = json.loads(fixture_path.read_text())
    if not (fixture['numpy_mmap_write_isolation'] and fixture['separate_inodes_and_link_counts']
            and fixture['original_research_files_modified'] is False):
        raise ValueError('Prior same-filesystem CoW write-isolation fixture required')
    REPORT.mkdir()
    plan = dict(state='declared_exact_inactive_cache_cow_sharing', observed=stamp(),
                script_sha256=sha(Path(__file__)), candidates_sha256=sha(CANDIDATES),
                fixture_sha256=sha(fixture_path), records=candidates['records'],
                policy='Full SHA equality, full byte comparison, unchanged identities, no handles, no xattrs/ACL/flags, separate clone inode; preserve destination permissions/owner/mtime, atomic replacement, complete post-SHA and NumPy mmap readability. No logical numerical file deleted or modified.')
    write(REPORT / 'plan.json', plan)
    before = shutil.disk_usage(ROOT).free
    done = []
    for index, group in enumerate(plan['records']):
        if not group['full_sha256_equal'] or len(group['files']) != 2:
            raise ValueError('Two fully hash-equal candidates required')
        a, b = group['files']
        source, target = Path(a['path']), Path(b['path'])
        sa, sb = validate_path(source, a), validate_path(target, b)
        if (sa['dev'] != sb['dev'] or sa['ino'] == sb['ino'] or
                any(sa[k] != sb[k] for k in ['mode', 'uid', 'gid', 'bytes'])):
            raise ValueError('Metadata or filesystem incompatible')
        assert_closed([source, target])
        if sha(source) != a['sha256'] or sha(target) != a['sha256']:
            raise ValueError('Complete original bytes changed')
        same_bytes(source, target)
        validate_path(source, a)
        validate_path(target, b)
        original_time = target.stat()
        stage = target.with_name(target.name + '.cow-stage')
        if stage.exists():
            raise ValueError('Existing staging file preserved')
        subprocess.run(['/bin/cp', '-c', '-p', str(source), str(stage)], check=True)
        os.utime(stage, ns=(original_time.st_atime_ns, original_time.st_mtime_ns))
        ss = identity(stage)
        if (ss['ino'] in [sa['ino'], sb['ino']] or ss['nlink'] != 1 or
                any(ss[k] != sb[k] for k in ['dev', 'bytes', 'mtime_ns', 'mode', 'uid', 'gid', 'flags']) or
                sha(stage) != a['sha256']):
            raise ValueError('Clone bytes/independent inode/metadata differ; original preserved')
        assert_closed([source, target])
        validate_path(source, a)
        validate_path(target, b)
        write(REPORT / f'{index}-ready.json', dict(source=a, target_before=b, clone=ss,
              original_both_complete_hashes_verified=True, original_full_byte_comparison=True,
              no_open_handles=True, clone_sha256=a['sha256'], observed=stamp()))
        os.replace(stage, target)
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        after_target = identity(target)
        if after_target != ss or sha(target) != a['sha256'] or sha(source) != a['sha256']:
            raise ValueError('Post-clone logical content or metadata mismatch')
        validate_path(source, a)
        # Inspect the actual .npy header/mmap through each retained logical path.
        import numpy as np
        left = np.load(source, mmap_mode='r', allow_pickle=False)
        right = np.load(target, mmap_mode='r', allow_pickle=False)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Post-clone numerical array readability differs')
        shape = list(left.shape)
        del left, right
        row = dict(index=index, source=str(source.relative_to(ROOT)),
                   target=str(target.relative_to(ROOT)), logical_bytes=sa['bytes'],
                   sha256=a['sha256'], original_target_inode=sb['ino'],
                   source_inode=sa['ino'], cloned_target_inode=ss['ino'],
                   metadata_preserved=True, separate_inodes_and_single_links=True,
                   complete_before_and_after_hashes_verified=True, full_byte_comparison=True,
                   numpy_mmap_shape=shape, observed=stamp())
        done.append(row)
        write(REPORT / f'{index}-complete.json', row)
        print(row['target'], row['logical_bytes'], 'complete', flush=True)
    after = shutil.disk_usage(ROOT).free
    write(REPORT / 'verification.json', dict(state='complete_exact_inactive_cache_cow_sharing',
          observed=stamp(), plan_sha256=sha(REPORT / 'plan.json'), records=done,
          logical_duplicate_bytes_shared=sum(r['logical_bytes'] for r in done),
          disk_free_before_bytes=before, disk_free_after_bytes=after,
          observed_free_space_change_bytes=after-before,
          concurrent_native_writes_may_affect_free_space_delta=True,
          datasets_models_checkpoints_and_active_files_changed=False,
          logical_paths_retained=True, complete_numerical_bytes_preserved=True,
          future_writes_independent_by_verified_cow_fixture=True))


if __name__ == '__main__':
    main()
