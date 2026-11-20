#!/usr/bin/env python3
"""Preserve unused original1600 build intermediates byte-for-byte before unlink."""
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import upstream_repro as u
import upstream_read_cache_policy as identity

CACHE=u.ROOT/'artifacts/upstream-high-capacity-original/target/release/build'
ARCHIVE=u.ROOT/'artifacts/upstream-original1600-build-cache.tar.gz'
REPORT=u.ROOT/'experiments/upstream-original1600-build-cache-archive'


def stream_sha(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def durable(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    if REPORT.exists() or ARCHIVE.exists():raise ValueError('Existing archive/evidence retained')
    if CACHE.is_symlink() or CACHE.resolve()!=CACHE:raise ValueError('Exact nonsymlink cache root required')
    complete=u.ROOT/'experiments/upstream-high-capacity-successor/completion.json'
    if u.read(complete)['state']!='complete_original1600_and_independent_full_output_verification':
        raise ValueError('Fully complete original1600 producer required')
    protected=[complete,u.ROOT/'artifacts/upstream-high-capacity-original/target/release/tsvdx4-new']
    for rel in ['experiments/upstream-own391-paired-data/manifest.json',
        'experiments/upstream-own352-bounded-replay/independent-canonical-verification.json',
        'experiments/upstream-high-capacity-original/execution-plan.json']:
        path=u.ROOT/rel
        if str(CACHE.relative_to(u.ROOT))+'/' in path.read_text():raise ValueError('Build cache occurs in protected source closure')
        protected.append(path)
    protected_hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in protected}
    observed=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(CACHE)],text=True,capture_output=True)
    if observed.returncode!=1 or observed.stdout.strip() or observed.stderr.strip():
        raise ValueError('Build intermediates must have no observed open users')
    records=[]
    for path in sorted(CACHE.rglob('*')):
        if path.is_dir() and not path.is_symlink():continue
        s=path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or path.is_symlink():
            raise ValueError('Only regular single-link build intermediates may be archived')
        records.append(dict(path=str(path.relative_to(u.ROOT)),identity=identity.identity(path),sha256=u.sha(path)))
    if not records:raise ValueError('Explicit nonempty build-cache inventory required')
    REPORT.mkdir();before=shutil.disk_usage(u.ROOT).free
    u.write(REPORT/'plan.json',dict(records=records,protected_sha256=protected_hashes,script_sha256=u.sha(Path(__file__)),
        rule='Only completed unused original1600 target/release/build regular single-link intermediates. No source, executable, arrays, models, predictions, checkpoints or numerical proof deleted. Archive all bytes and rehash every extracted member before any unlink. Preserve archive and exact metadata inventory. Reject observed open users or changed identities. Keep empty directory hierarchy.'))
    with tarfile.open(ARCHIVE,'x:gz',compresslevel=3,format=tarfile.PAX_FORMAT) as archive:
        for row in records:
            path=u.ROOT/row['path']
            if identity.identity(path)!=row['identity']:raise ValueError('Build cache changed before archive')
            archive.add(path,arcname=row['path'],recursive=False)
    durable(ARCHIVE);durable(ARCHIVE.parent)
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        members=archive.getmembers()
        if [m.name for m in members]!=[r['path'] for r in records]:raise ValueError('Exact archive inventory required')
        for member,row in zip(members,records):
            if not member.isfile() or member.size!=row['identity']['bytes']:raise ValueError('Archive type/size mismatch')
            with archive.extractfile(member) as stream:
                if stream_sha(stream)!=row['sha256']:raise ValueError('Every archived payload must match original bytes')
    archive_sha=u.sha(ARCHIVE)
    for row in records:
        if identity.identity(u.ROOT/row['path'])!=row['identity']:raise ValueError('Original cache changed before removal')
    for rel,digest in protected_hashes.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Protected executable/evidence changed')
    u.write(REPORT/'archive-verification.json',dict(state='complete_every_original_build_cache_member_rehashed_before_unlink',
        archive=str(ARCHIVE.relative_to(u.ROOT)),archive_sha256=archive_sha,files=len(records),
        original_bytes=sum(r['identity']['bytes'] for r in records),archive_bytes=ARCHIVE.stat().st_size,
        plan_sha256=u.sha(REPORT/'plan.json')))
    durable(REPORT/'plan.json');durable(REPORT/'archive-verification.json');durable(REPORT);durable(REPORT.parent)
    for row in records:
        path=u.ROOT/row['path']
        if identity.identity(path)!=row['identity']:raise ValueError('Cache identity changed during removal')
        path.unlink()
    if any((u.ROOT/r['path']).exists() for r in records):raise ValueError('Every archived original must be accounted')
    for rel,digest in protected_hashes.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Protected executable/evidence changed after removal')
    u.write(REPORT/'verification.json',dict(state='complete_lossless_build_cache_archive_all_bytes_verified_and_protected_files_equal',
        archive_sha256=archive_sha,archive_verification_sha256=u.sha(REPORT/'archive-verification.json'),
        files=len(records),protected_sha256=protected_hashes,free_before_bytes=before,
        free_after_bytes=shutil.disk_usage(u.ROOT).free,numerical_files_removed=False,original_source_files_removed=False,
        generated_build_cache_sources_losslessly_archived=True))


if __name__=='__main__':main()
