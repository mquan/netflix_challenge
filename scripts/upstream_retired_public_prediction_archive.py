#!/usr/bin/env python3
"""Recover native-job disk headroom by losslessly archiving retired public predictions."""
import datetime as dt
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
import numpy as np
import upstream_repro as u
import upstream_public_stats_archive as original

REPORT=u.ROOT/'experiments/upstream-retired-public-prediction-archive'
OLD=original.OLD
DIRECTORIES=[OLD/'artifacts/public-new/qual/preds_new',OLD/'artifacts/public-bank/split3-columns',OLD/'artifacts/public-new/columns',OLD/'artifacts/public-bank/columns']
PROOFS=['upstream-own6-tight-calibration/calibration.json','upstream-nn-pool6-assembly/calibration.json',
    'upstream-own167-context218-mps/manifest.json','upstream-split3-full-raw-bank/manifest.json',
    'upstream-split3-voting-stream/plan.json','upstream-split3-matched-mps/manifest.json']

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    if REPORT.exists():raise ValueError('Existing lossless cleanup transaction preserved')
    REPORT.mkdir();protected=set();proofs={}
    for name in PROOFS:
        p=u.ROOT/'experiments'/name;proof=u.read(p);proofs[str(p.relative_to(u.ROOT))]=u.sha(p)
        for key in proof.get('source_sha256',{}):protected.add((u.ROOT/key).resolve())
    paths=[]
    for root in DIRECTORIES:
        for p in sorted(root.glob('*.npy')):
            s=p.lstat()
            if p.resolve() in protected or p.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_flags:continue
            tracked=subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(OLD))],cwd=OLD,capture_output=True)
            if tracked.returncode==0 or not original.closed(p):continue
            a=np.load(p,mmap_mode='r',allow_pickle=False)
            if a.ndim!=1 or a.dtype!=np.float32 or len(a) not in [1128164,1760494,2817131]:raise ValueError('Unexpected retired public feature schema')
            del a
            paths.append(p)
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256={**proofs,str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),str(Path(original.__file__).relative_to(u.ROOT)):u.sha(Path(original.__file__))},
        candidates=[str(p) for p in paths],target_free_gib=20.5,
        rule='Only closed untracked single-link retired public-bank *.npy feature caches in four explicit public directories, excluded from own active source bindings. Retain every original NPY byte and metadata through full decode-SHA verified zstd archives before unlink. Own numerical inputs, datasets, model weights, checkpoints and source untouched. Stop at20.5GiB native admission headroom; preserve16GiB floor+0.75GiB checkpoint writes+1GiB temporary cleanup margin. No new training allocation or lowered job guard.'))
    before=shutil.disk_usage(u.ROOT).free;records=[]
    for p in paths:
        if shutil.disk_usage(u.ROOT).free>20.5*2**30:break
        s=p.stat()
        if shutil.disk_usage(u.ROOT).free<17.75*2**30+s.st_size+2**20:raise RuntimeError('Original checkpoint cleanup reserve unavailable')
        target=p.with_name(p.name+'.zst');temp=p.with_name(p.name+'.archive-in-progress')
        if target.exists() or temp.exists():raise ValueError('Existing archive transaction preserved')
        h=u.sha(p)
        if not original.closed(p):continue
        try:
            with temp.open('xb') as output:
                child=subprocess.Popen(['zstd','-q','-3','-T1','-c',str(p)],stdout=output,stderr=subprocess.PIPE)
                while child.poll() is None:
                    if shutil.disk_usage(u.ROOT).free<17.75*2**30:
                        child.terminate();child.wait();raise RuntimeError('Original checkpoint reserve reached; original public cache preserved')
                    time.sleep(.2)
                error=child.stderr.read()
                if child.returncode:raise RuntimeError(error.decode())
            decoded,size=original.decode_digest(temp)
            current=p.stat()
            if (decoded!=h or size!=s.st_size or u.sha(p)!=h or current.st_mtime_ns!=s.st_mtime_ns or
                current.st_ino!=s.st_ino or current.st_nlink!=1 or not original.closed(p)):raise ValueError('Exact full-byte archive/source verification failed')
            if temp.stat().st_size>=s.st_size:temp.unlink();continue
            os.chmod(temp,0o444);temp.replace(target)
            row=dict(original=str(p),archive=str(target),original_sha256=h,original_bytes=s.st_size,archive_sha256=u.sha(target),archive_bytes=target.stat().st_size,
                metadata=dict(mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns,atime_ns=s.st_atime_ns,flags=s.st_flags),
                exact_full_decode_sha256_verified=True,uncompressed_cache_retired=False)
            records.append(row);u.write(REPORT/'partial.json',dict(records=records))
            p.unlink();row['uncompressed_cache_retired']=True;u.write(REPORT/'partial.json',dict(records=records))
            print(p.name,'exact archive verified; net',s.st_size-target.stat().st_size,'bytes recovered',flush=True)
        except BaseException:
            if temp.exists():temp.unlink()
            raise
    for key,h in proofs.items():
        if u.sha(u.ROOT/key)!=h:raise ValueError('Own source binding changed during cleanup')
    u.write(REPORT/'verification.json',dict(state='complete_lossless_retired_public_prediction_archival',observed=stamp(),records=records,
        logical_bytes_reclaimed=sum(r['original_bytes']-r['archive_bytes'] for r in records),free_before=before,free_after=shutil.disk_usage(u.ROOT).free,
        own_datasets_models_checkpoints_and_prediction_values_changed=False,original_bytes_discarded_without_archive=0,
        restore_command='.venv/bin/python scripts/upstream_retired_public_prediction_archive.py --restore-all'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--restore-all',action='store_true');args=p.parse_args()
    if args.restore_all:original.REPORT=REPORT;original.restore()
    else:main()
