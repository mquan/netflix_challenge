#!/usr/bin/env python3
"""Discard only fully accounted, index-exact retired downloaded public inodes."""
import datetime as dt
from pathlib import Path
import re
import shutil
import stat
import subprocess
import numpy as np
import tomli
import upstream_repro as u

OLD=Path('/Users/quan/workspace/netflix_prize')
REF=u.ROOT/'artifacts/upstream-repro/reference/preds_new'
INDEX=OLD/'artifacts/public-source-audit-index-20261115.toml'
INVENTORY=u.ROOT/'experiments/upstream-remaining-public-hardlink-inventory.json'
REPORT=u.ROOT/'experiments/upstream-retire-reference-hardlinks'
ALLOWED=[REF,OLD/'artifacts/public-new/columns',
    OLD/'artifacts/public-probe-blenders/source/preds_new',
    OLD/'artifacts/public-probe-blenders-expanded/source/preds_new']
PROOFS=['upstream-own391-paired-data/manifest.json',
    'upstream-own391-paired-h128-queue/plan.json','upstream-complete-cf-delta-queue/plan.json',
    'upstream-cf96-complete-bank/manifest.json','upstream-high-capacity-original/execution-plan.json']


def closed(paths):
    result=subprocess.run(['/usr/sbin/lsof','-Fpn','--',*[str(p) for p in paths]],text=True,capture_output=True)
    return result.returncode==1 and not result.stdout.strip() and not result.stderr.strip()


def main():
    if REPORT.exists():raise ValueError('Existing retirement evidence preserved')
    for directory in ALLOWED:
        if directory.resolve()!=directory or directory.is_symlink():
            raise ValueError('Explicit public directories must not redirect')
    inventory=u.read(INVENTORY)
    if inventory['state']!='read_only_remaining_downloaded_reference_inode_inventory' or inventory['files_deleted']!=0:
        raise ValueError('Read-only full inode inventory required')
    index={r['path']:r for r in tomli.loads(INDEX.read_text())['file']}
    tracked={root:set(subprocess.check_output(['git','ls-files','-z'],cwd=root).split(b'\0')) for root in [u.ROOT,OLD]}
    protected=set();bindings={}
    for relative in PROOFS:
        p=u.ROOT/'experiments'/relative;proof=u.read(p);bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
        for key in proof.get('source_sha256',proof.get('bindings',{})):
            path=u.ROOT/key
            if path.is_file():s=path.stat();protected.add((s.st_dev,s.st_ino))
    for p in [Path(__file__),u.ROOT/'scripts/verify_upstream_reference_retirement.py',INDEX,INVENTORY]:bindings[str(p)]=u.sha(p)
    REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=bindings,
        allowed_directories=[str(p) for p in ALLOWED],
        authorization='User explicitly authorized cleanup and removal of downloaded public-bank results; they are not our own effort.',
        policy='Only complete known-ProbeX public float32 arrays with original download-index MD5/size. Every hardlink must be inventoried and be an untracked/unflagged/closed regular file in four exact public directories. Exclude every inode bound by current own391/delta/CF95/1600 source closures. Recheck all aliases and bytes before unlink. Keep pre-deletion SHA/metadata journal. Never remove source, archives, own outputs/models/checkpoints or change any process. Intentional public-cache discard, not lossless archival.'))
    before=shutil.disk_usage(u.ROOT).free;records=[];skipped=[]
    for item in inventory['records']:
        paths=[Path(s) for s in item['aliases']]
        if not paths or len(paths)!=item['nlink'] or len(set(paths))!=len(paths):
            skipped.append(dict(sample=item['sample'],reason='incomplete alias accounting'));continue
        if any(p.parent not in ALLOWED or not re.fullmatch(r'[A-Za-z0-9_.-]+\.probex\.npy',p.name) for p in paths):
            skipped.append(dict(sample=item['sample'],reason='outside exact retired-public allowlist'));continue
        if len({p.name for p in paths})!=1:raise ValueError('All hardlink names must match the same indexed public value')
        first=paths[0];s=first.lstat();key=(s.st_dev,s.st_ino)
        if key in protected:
            skipped.append(dict(sample=item['sample'],reason='protected own input inode'));continue
        snapshots=[]
        for p in paths:
            a=p.lstat();root=u.ROOT if p.is_relative_to(u.ROOT) else OLD
            if (not stat.S_ISREG(a.st_mode) or p.is_symlink() or p.parent.resolve()!=p.parent or
                a.st_flags or (a.st_dev,a.st_ino)!=key or a.st_nlink!=len(paths) or
                str(p.relative_to(root)).encode() in tracked[root]):
                raise ValueError('An alias is not an eligible unchanged untracked public file: '+str(p))
            snapshots.append((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_nlink,a.st_mode,a.st_uid,a.st_gid,a.st_flags))
        entry=index['preds_new/'+first.name]
        if s.st_size!=entry['size'] or s.st_size!=item['bytes'] or u.sha(first,'md5')!=entry['md5']:
            raise ValueError('Not the exact downloaded public value; preserve all aliases')
        array=np.load(first,mmap_mode='r',allow_pickle=False)
        if array.shape!=(1128164,) or array.dtype!=np.dtype('<f4'):raise ValueError('Original public shape differs')
        del array
        digest=u.sha(first)
        if not closed(paths):
            skipped.append(dict(sample=item['sample'],reason='open or unobservable public inode'));continue
        if u.sha(first)!=digest:raise ValueError('Public bytes changed during inventory')
        for p,expected in zip(paths,snapshots):
            a=p.lstat()
            if (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_nlink,a.st_mode,a.st_uid,a.st_gid,a.st_flags)!=expected:
                raise ValueError('Public alias metadata changed before discard')
        row=dict(paths=[str(p) for p in paths],sha256=digest,download_index_md5=entry['md5'],bytes=s.st_size,
            device=s.st_dev,inode=s.st_ino,original_nlink=s.st_nlink,mode=stat.S_IMODE(s.st_mode),
            uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns,removed_paths=[])
        records.append(row);u.write(REPORT/'partial.json',dict(records=records))
        for i,p in enumerate(paths):
            a=p.lstat()
            if (a.st_dev,a.st_ino)!=key or a.st_nlink!=len(paths)-i:raise ValueError('Unexpected new alias during discard')
            p.unlink();row['removed_paths'].append(str(p));u.write(REPORT/'partial.json',dict(records=records))
    for relative,digest in bindings.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Protected evidence changed during cleanup')
    if any(Path(p).exists() for r in records for p in r['removed_paths']):raise ValueError('Discarded alias remains')
    u.write(REPORT/'verification.json',dict(state='complete_authorized_index_exact_public_reference_hardlink_retirement',
        records=records,skipped=skipped,logical_bytes_discarded=sum(r['bytes'] for r in records),
        free_before=before,free_after=shutil.disk_usage(u.ROOT).free,own_inputs_models_checkpoints_modified=False,
        earlier_lossless_archives_modified=False,qualifying_outcomes_read=False))
    print('Retired public inode groups:',len(records),'bytes:',sum(r['bytes'] for r in records),flush=True)


if __name__=='__main__':main()
