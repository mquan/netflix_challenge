#!/usr/bin/env python3
"""Read-only independent replay of retained legacy-cache archive evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'experiments/upstream-legacy-build-cache-archive'
EXPECTED = {
    'full-upstream-mapped', 'upstream-aex-backfill',
    'upstream-cfnade-checkpoint', 'upstream-cfnade-fast',
    'upstream-cfnade-fast-v2', 'upstream-cfnade-sparse-rows',
    'upstream-cfnade96-sparse-resume', 'upstream-grouped-storage',
    'upstream-parallel-mapped', 'upstream-qualifying-backfill',
}


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1048576), b''):
        result.update(block)
    return result.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    completion = read(REPORT / 'verification.json')
    selection = read(REPORT / 'selection.json')
    if (completion['state'] != 'complete_all10_lossless_legacy_build_cache_archives_and_protected_evidence'
            or completion['selection_sha256'] != sha(REPORT / 'selection.json')
            or completion['numerical_files_removed'] is not False
            or completion['original_source_files_removed'] is not False
            or completion['protected_sha256'] != selection['protected_sha256']
            or len(completion['caches']) != 10
            or {r['name'] for r in completion['caches']} != EXPECTED
            or len(selection['caches']) != 10
            or {r['name'] for r in selection['caches']} != EXPECTED):
        raise ValueError('Exact declared ten-cache completion and protection chain required')
    protected = dict(selection['protected_sha256'])
    results = []
    total_files = total_bytes = 0
    for record in completion['caches']:
        name = record['name']
        directory = REPORT / name
        proof = read(directory / 'verification.json')
        if (sha(directory / 'verification.json') != record['verification_sha256']
                or proof != record['verification']
                or proof['state'] != 'complete_lossless_build_cache_archive_all_bytes_verified_and_protected_files_equal'):
            raise ValueError('Original per-cache completion binding differs')
        before = read(directory / 'archive-verification.json')
        plan = read(directory / 'plan.json')
        if (proof['archive_verification_sha256'] != sha(directory / 'archive-verification.json')
                or before['plan_sha256'] != sha(directory / 'plan.json')
                or before['state'] != 'complete_every_original_build_cache_member_rehashed_before_unlink'
                or proof['protected_sha256'] != plan['protected_sha256']):
            raise ValueError('Before-unlink payload/protection proof chain differs')
        for relative, expected in plan['protected_sha256'].items():
            if relative in protected and protected[relative] != expected:
                raise ValueError('Conflicting protected file expectation')
            protected[relative] = expected
        path = ROOT / 'artifacts' / (name + '-legacy-build-cache.tar.gz')
        if (before['archive'] != str(path.relative_to(ROOT))
                or before['archive_sha256'] != sha(path)
                or proof['archive_sha256'] != before['archive_sha256']
                or before['archive_bytes'] != path.stat().st_size):
            raise ValueError('Exact retained archive bytes required')
        # Read through the gzip trailer to verify the full CRC, separately
        # from tar's end-of-archive handling.
        with gzip.open(path, 'rb') as stream:
            digest(stream)
        inventory = plan['records']
        prefix = 'artifacts/' + name + '/target/release/build/'
        paths = [row['path'] for row in inventory]
        if len(paths) != len(set(paths)) or not paths:
            raise ValueError('Unique nonempty original inventory required')
        chosen = next(row for row in selection['caches'] if row['name'] == name)
        if (chosen['cache'] + '/' != prefix or chosen['files'] != len(inventory)
                or chosen['bytes'] != sum(row['identity']['bytes'] for row in inventory)
                or before['files'] != len(inventory)
                or before['original_bytes'] != chosen['bytes']):
            raise ValueError('Selection and original full inventory differ')
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            if [member.name for member in members] != paths:
                raise ValueError('Exact archive membership/order required')
            for member, row in zip(members, inventory):
                parts = Path(member.name).parts
                if (not member.isfile() or member.name.startswith('/')
                        or '..' in parts or not member.name.startswith(prefix)
                        or member.size != row['identity']['bytes']):
                    raise ValueError('Safe regular build-cache member required')
                with archive.extractfile(member) as stream:
                    if digest(stream) != row['sha256']:
                        raise ValueError('Every retained payload must replay original SHA256')
                if (ROOT / row['path']).exists():
                    raise ValueError('Retired original unexpectedly exists')
        total_files += len(inventory)
        total_bytes += chosen['bytes']
        results.append(dict(name=name, archive_sha256=before['archive_sha256'],
                            original_files=len(inventory), original_bytes=chosen['bytes']))
    for relative, expected in protected.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Protected executable/source/evidence differs: ' + relative)
    return dict(state='complete_independent_all10_retained_archives_every_original_payload_CRC_absence_and_protected_hashes',
                files=total_files, original_bytes=total_bytes, archives=results,
                protected_files=len(protected), protected_sha256=protected,
                completion_sha256=sha(REPORT / 'verification.json'),
                selection_sha256=sha(REPORT / 'selection.json'),
                verifier_sha256=sha(Path(__file__)), recovered_qualifying_labels_read=False)


if __name__ == '__main__':
    result = verify()
    output = REPORT / 'independent-verification.json'
    if output.exists():
        raise ValueError('Existing independent evidence retained')
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(result['state'], result['files'], result['original_bytes'],
          'protected', result['protected_files'])
