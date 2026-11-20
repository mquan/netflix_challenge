#!/usr/bin/env python3
"""Discard explicitly retired downloaded public prediction caches, preserving own work."""
import datetime as dt
import os
from pathlib import Path
import shutil
import stat
import subprocess
import numpy as np
import upstream_repro as u
import upstream_public_stats_archive as old

REPORT=u.ROOT/'experiments/upstream-retire-downloaded-public-qual-cache'
ROOT=old.OLD/'artifacts/public-qual'
PAUSED={40876:'upstream_cf96_disk_resume.py',40924:'cfnade-stage x',61826:'upstream_split3_next3.py',87902:'fit tx s3-tx100r'}

def paused():
    for pid,command in PAUSED.items():
        value=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],capture_output=True,text=True).stdout.strip()
        if not value or not value.startswith('T') or command not in value:raise ValueError('Original native state/owner must remain paused during disk reserve recovery')

def main():
    REPORT.mkdir(exist_ok=False);paused()
    if ROOT.is_symlink():raise ValueError('Explicit retired public cache directory required')
    own_proofs=[u.ROOT/'experiments'/p for p in ['upstream-split3-matched-mps/manifest.json','upstream-split3-full-raw-bank/manifest.json',
        'upstream-own6-tight-calibration/calibration.json','upstream-nn-pool6-assembly/calibration.json','upstream-split3-voting-stream/plan.json']]
    protected=set();bindings={}
    for p in own_proofs:
        proof=u.read(p);bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
        for key in proof.get('source_sha256',{}):protected.add((u.ROOT/key).resolve())
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256={**bindings,str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))},
        authorization='User explicitly requested storage cleanup and agreed to removing public-bank results because they are not our own effort. This discards only remaining downloaded public qualifying prediction caches, not earlier verified archives or own numerical values.',
        roots=[str(ROOT/'preds_3'),str(ROOT/'preds_old')],target_free_gib=20.25,
        policy='Only closed untracked unflagged single-link regular *.qual.npy vectors of2817131float32 rows in two explicit retired-public directories; exclude all own bound inputs. All original native writer processes and owners paused while recovering disk reserve. Full pre-deletion SHA/metadata catalog retained. Downloaded numerical caches intentionally discarded without archive; existing exact archives, data, source and all own models/checkpoints/fits untouched.'))
    before=shutil.disk_usage(u.ROOT).free;records=[]
    for directory in [ROOT/'preds_3',ROOT/'preds_old']:
        if directory.is_symlink():raise ValueError('Retired prediction directory must not redirect')
        for path in sorted(directory.glob('*.qual.npy')):
            if shutil.disk_usage(u.ROOT).free>20.25*2**30:break
            paused();s=path.lstat()
            if path.resolve() in protected or path.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_flags:continue
            tracked=subprocess.run(['git','ls-files','--error-unmatch',str(path.relative_to(old.OLD))],cwd=old.OLD,capture_output=True)
            if tracked.returncode!=1 or not old.closed(path):continue
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.shape!=(2817131,) or a.dtype!=np.float32:raise ValueError('Explicit downloaded public qualifying vector shape differs')
            del a
            h=u.sha(path);current=path.stat()
            if (current.st_ino,current.st_size,current.st_mtime_ns,current.st_nlink)!=(s.st_ino,s.st_size,s.st_mtime_ns,1) or not old.closed(path):raise ValueError('Retired cache changed during inventory')
            row=dict(path=str(path),sha256=h,bytes=s.st_size,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,discarded=False)
            records.append(row);u.write(REPORT/'partial.json',dict(records=records))
            path.unlink();row['discarded']=True;u.write(REPORT/'partial.json',dict(records=records))
            print('Retired downloaded public cache',path.name,flush=True)
    for key,h in bindings.items():
        if u.sha(u.ROOT/key)!=h:raise ValueError('Own active input bindings changed')
    u.write(REPORT/'verification.json',dict(state='complete_authorized_downloaded_public_prediction_cache_retirement',records=records,
        free_before=before,free_after=shutil.disk_usage(u.ROOT).free,logical_downloaded_cache_bytes_discarded=sum(r['bytes'] for r in records),
        own_training_inputs_models_checkpoints_modified=False,prior_lossless_archives_modified=False,qualifying_outcomes_read=False))

if __name__=='__main__':main()
