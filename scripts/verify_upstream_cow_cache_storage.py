#!/usr/bin/env python3
"""Separate complete-byte and metadata check of retained CoW cache paths."""
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'experiments/upstream-cow-cache-storage'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = REPORT / 'independent-verification.json'
    if out.exists():
        raise ValueError('Existing independent evidence preserved')
    result = json.loads((REPORT / 'verification.json').read_text())
    plan = json.loads((REPORT / 'plan.json').read_text())
    if (result['state'] != 'complete_exact_inactive_cache_cow_sharing'
            or result['plan_sha256'] != sha(REPORT / 'plan.json')
            or plan['script_sha256'] != sha(ROOT / 'scripts/upstream_cow_cache_storage.py')
            or len(result['records']) != len(plan['records'])):
        raise ValueError('Original complete execution and immutable plan required')
    checked = []
    for i, row in enumerate(result['records']):
        original = plan['records'][i]['files']
        a, b = ROOT / row['source'], ROOT / row['target']
        sa, sb = a.stat(), b.stat()
        for path, s, before in [(a, sa, original[0]), (b, sb, original[1])]:
            if (not path.is_file() or path.is_symlink() or s.st_nlink != 1 or
                    s.st_size != before['bytes'] or s.st_mtime_ns != before['mtime_ns'] or
                    s.st_mode != before['mode'] or s.st_uid != before['uid'] or
                    s.st_gid != before['gid'] or s.st_flags != before['flags'] or
                    sha(path) != before['sha256']):
                raise ValueError('Complete logical file or original metadata differs')
            if subprocess.check_output(['/usr/bin/xattr', str(path)]):
                raise ValueError('Unexpected extended attributes')
        if sa.st_dev != sb.st_dev or sa.st_ino == sb.st_ino or sa.st_ino != original[0]['ino']:
            raise ValueError('Unchanged source and independent clone inode required')
        subprocess.run(['/usr/bin/cmp', '-s', str(a), str(b)], check=True)
        x, y = np.load(a, mmap_mode='r'), np.load(b, mmap_mode='r')
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError('Both actual NumPy paths must remain readable')
        checked.append(dict(source=row['source'], target=row['target'],
                            sha256=sha(b), logical_bytes=sb.st_size,
                            independent_full_cmp_equal=True, metadata_preserved=True,
                            separate_inodes_and_links=True, numpy_shape=list(y.shape)))
        del x, y
    proof = dict(state='complete_independent_cow_logical_file_verification',
                 observed=dt.datetime.now(dt.timezone.utc).isoformat(), records=checked,
                 shared_duplicate_bytes=sum(r['logical_bytes'] for r in checked),
                 execution_sha256=sha(REPORT / 'verification.json'),
                 plan_sha256=sha(REPORT / 'plan.json'), verifier_sha256=sha(Path(__file__)),
                 scope='All original logical files retain exact complete hashes and metadata; separate cmp reads every pair, and NumPy mmaps both retained paths. The original fixture separately proves CoW write isolation.')
    out.write_text(json.dumps(proof, indent=2) + '\n')
    print(proof['state'], proof['shared_duplicate_bytes'], flush=True)


if __name__ == '__main__':
    main()
