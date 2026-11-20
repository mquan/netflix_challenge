#!/usr/bin/env python3
"""Losslessly archive closed retired public-bank Gram caches; keep exact restoration."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
import zipfile
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-retired-public-stats-archive'
OLD=Path('/Users/quan/workspace/netflix_prize')
DIRECTORIES=[OLD/'artifacts/public-new',OLD/'artifacts/public-bank']

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def closed(path):
    result=subprocess.run(['/usr/sbin/lsof','-Fpn','--',str(path)],text=True,capture_output=True)
    return result.returncode==1 and not result.stdout.strip() and not result.stderr.strip()

def decode_digest(archive):
    child=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    digest=hashlib.sha256();count=0
    for block in iter(lambda:child.stdout.read(4*2**20),b''):digest.update(block);count+=len(block)
    error=child.stderr.read();code=child.wait()
    if code:raise RuntimeError('Lossless archive decode failed: '+error.decode())
    return digest.hexdigest(),count

def archive():
    REPORT.mkdir(exist_ok=False)
    protected='\n'.join((u.ROOT/p).read_text() for p in [
        'experiments/upstream-own6-tight-calibration/calibration.json',
        'experiments/upstream-nn-pool6-assembly/calibration.json',
        'experiments/upstream-own167-context218-mps/manifest.json',
        'experiments/upstream-split3-full-raw-bank/manifest.json'])
    candidates=[]
    for directory in DIRECTORIES:
        for p in directory.rglob('*.stats.npz'):
            s=p.lstat()
            if p.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_flags or s.st_size<16*2**20:continue
            if str(p) in protected or p.name in protected:continue
            tracked=subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(OLD))],cwd=OLD,capture_output=True)
            if tracked.returncode==0 or not closed(p):continue
            with zipfile.ZipFile(p) as package:
                if set(package.namelist())!={'gram.npy','rhs.npy','count.npy'}:continue
            candidates.append(p)
    candidates.sort(key=lambda p:p.stat().st_size)
    u.write(REPORT/'plan.json',dict(created=stamp(),script_sha256=u.sha(Path(__file__)),candidates=[str(p) for p in candidates],
        rule='Only untracked, closed, unflagged single-link retired public-bank Gram/rhs/count caches. Preserve every original byte via checksum-verified zstd archive before retiring uncompressed cache. No data arrays, model weights, own study inputs, native checkpoints or source code. Exact metadata and restoration command retained. Cleanup temporary peak bounded by original size+1MiB above native16GiB floor+0.75GiB checkpoint writes+1GiB transaction margin. This is a bounded space-reducing transaction, not new model-cache admission.'))
    before=shutil.disk_usage(u.ROOT).free;records=[]
    for p in candidates:
        s=p.stat();minimum=16*2**30+int(.75*2**30)+2**30+s.st_size+2**20
        if shutil.disk_usage(u.ROOT).free<minimum:continue
        target=p.with_name(p.name+'.zst');temp=p.with_name(p.name+'.archive-in-progress')
        if target.exists() or temp.exists():raise ValueError('Existing archival files preserved')
        original=u.sha(p)
        metadata=dict(mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns,atime_ns=s.st_atime_ns,flags=s.st_flags)
        if not closed(p):continue
        try:
            with temp.open('xb') as output:
                child=subprocess.Popen(['zstd','-q','-3','-T1','-c',str(p)],stdout=output,stderr=subprocess.PIPE)
                while child.poll() is None:
                    if shutil.disk_usage(u.ROOT).free<16*2**30+int(.75*2**30)+2**30:
                        child.terminate();child.wait();raise RuntimeError('Cleanup checkpoint reserve reached; original cache preserved')
                    time.sleep(.2)
                error=child.stderr.read()
                if child.returncode:raise RuntimeError(error.decode())
            decoded,size=decode_digest(temp)
            if decoded!=original or size!=s.st_size or u.sha(p)!=original or p.stat().st_mtime_ns!=s.st_mtime_ns or not closed(p):
                raise ValueError('Exact original cache/archive verification failed')
            if temp.stat().st_size>=s.st_size:
                temp.unlink();continue
            os.chmod(temp,0o444);temp.replace(target)
            row=dict(original=str(p),archive=str(target),original_sha256=original,original_bytes=s.st_size,
                archive_sha256=u.sha(target),archive_bytes=target.stat().st_size,metadata=metadata,
                exact_full_decode_sha256_verified=True,uncompressed_cache_retired=False)
            records.append(row);u.write(REPORT/'partial.json',dict(records=records))
            p.unlink();row['uncompressed_cache_retired']=True
            u.write(REPORT/'partial.json',dict(records=records))
            print(p.name,s.st_size-target.stat().st_size,'bytes recovered; exact archive verified',flush=True)
        except BaseException:
            if temp.exists():temp.unlink()
            raise
    u.write(REPORT/'verification.json',dict(state='complete_lossless_retired_public_gram_cache_archival',observed=stamp(),records=records,
        logical_bytes_reclaimed=sum(r['original_bytes']-r['archive_bytes'] for r in records),free_before=before,free_after=shutil.disk_usage(u.ROOT).free,
        original_bytes_discarded_without_archive=0,own_datasets_or_models_or_checkpoints_changed=False,
        restore_command='.venv/bin/python scripts/upstream_public_stats_archive.py --restore-all'))

def restore():
    source=REPORT/('verification.json' if (REPORT/'verification.json').exists() else 'partial.json')
    for row in u.read(source)['records']:
        path,archive_path=Path(row['original']),Path(row['archive'])
        if path.exists():
            if u.sha(path)!=row['original_sha256']:raise ValueError('Existing historical path has different bytes')
            continue
        if u.sha(archive_path)!=row['archive_sha256']:raise ValueError('Exact archive changed')
        if shutil.disk_usage(u.ROOT).free<16*2**30+int(.75*2**30)+2*2**30+row['original_bytes']:raise RuntimeError('Insufficient original checkpoint reserve for restoration')
        temp=path.with_name(path.name+'.restore-in-progress')
        with temp.open('xb') as output:subprocess.run(['zstd','-q','-d','-c',str(archive_path)],stdout=output,check=True)
        if u.sha(temp)!=row['original_sha256']:raise ValueError('Original byte restoration differs')
        m=row['metadata'];os.chmod(temp,m['mode']);os.chown(temp,m['uid'],m['gid']);os.utime(temp,ns=(m['atime_ns'],m['mtime_ns']))
        temp.replace(path)
        print('Restored exact cache',path,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restore-all',action='store_true');args=parser.parse_args()
    restore() if args.restore_all else archive()
