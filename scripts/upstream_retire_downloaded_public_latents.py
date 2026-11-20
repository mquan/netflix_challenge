#!/usr/bin/env python3
"""Retire only index-verified downloaded latent caches after completed TX output."""
import datetime as dt
from pathlib import Path
import re
import shutil
import stat
import subprocess
import numpy as np
import tomli
import upstream_repro as u
import upstream_public_stats_archive as old
import upstream_cf96_primary_lane as owners

REPORT=u.ROOT/'experiments/upstream-retire-downloaded-public-latents'
INDEX=old.OLD/'artifacts/public-source-audit-index-20261115.toml'
ROOTS=[(old.OLD/'artifacts/public-new/support','preds_new/lat',1128164),
       (old.OLD/'artifacts/public-bank/split3-support','preds_3/lat',1760494),
       (old.OLD/'artifacts/public-new/qual/preds_new/lat','preds_new/lat',2817131)]


def paused():
    for pid,token in [(40876,'upstream_cf96_disk_resume.py'),(40924,'cfnade-stage x')]:
        actual=owners.process(pid)
        if not actual or token not in actual['command'] or 'T' not in actual['state']:
            raise ValueError('Original CF native and owner must remain paused')
    native=owners.process(87902)
    if native and 'fit tx s3-tx100r' in native['command']:raise ValueError('Temporal native must already be terminal')
    proof=u.read(u.ROOT/'experiments/upstream-split3-next3/s3-tx100r-verification.json')
    if proof['state']!='complete_original_two_phase_split3_model' or proof['epochs_per_phase']!=22 or not proof['both_full_epoch_traces']:
        raise ValueError('Actual full temporal22/22 verification required')


def main():
    REPORT.mkdir(exist_ok=False);paused()
    proofs=[u.ROOT/'experiments'/name for name in [
        'upstream-own330-neural-inputs/manifest.json','upstream-own330-native149-data/manifest.json',
        'upstream-own6-tight-calibration/calibration.json','upstream-nn-pool6-assembly/calibration.json',
        'upstream-split3-full-raw-bank/manifest.json','upstream-split3-matched-mps/manifest.json',
        'upstream-split3-voting-stream/plan.json','upstream-high-capacity-original/execution-plan.json']]
    protected=set();bindings={}
    for path in proofs:
        proof=u.read(path);bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
        for key in proof.get('source_sha256',proof.get('bindings',{})):
            protected.add((u.ROOT/key).resolve())
    index={entry['path']:entry for entry in tomli.loads(INDEX.read_text())['file']}
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
            authorization='Prior explicit user request for storage cleanup and removal of public-bank results because they are not our own effort.',
            source_sha256=bindings,index_sha256=u.sha(INDEX),script_sha256=u.sha(Path(__file__)),
            roots=[str(root) for root,_,_ in ROOTS],
            policy='Discard only closed, untracked, unflagged single-link public latent vectors whose full bytes match original downloaded-source index MD5/size and exact shape. Preserve every own bound input and all earlier lossless archives. Keep complete pre-deletion SHA/metadata catalog. Original CF state stays paused; temporal must already have complete22/22 proof. Downloaded cache discard is intentional, not lossless archival.'))
    before=shutil.disk_usage(u.ROOT).free;records=[]
    for root,prefix,count in ROOTS:
        if root.is_symlink():raise ValueError('Retired public root must not redirect')
        for path in sorted(root.glob('*.npy')):
            if not re.fullmatch(r'(?:s3-)?(?:rbm50lat\.u\d\d|tx20lat\.[ui]\d\d)\.(?:probex|probe_3|qual)\.npy',path.name):continue
            paused();s=path.lstat()
            if path.resolve() in protected or path.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_flags:continue
            tracked=subprocess.run(['git','ls-files','--error-unmatch',str(path.relative_to(old.OLD))],cwd=old.OLD,capture_output=True)
            if tracked.returncode!=1 or not old.closed(path):continue
            entry=index[prefix+'/'+path.name]
            if s.st_size!=entry['size'] or u.sha(path,'md5')!=entry['md5']:
                raise ValueError('Cache is not the exact downloaded public value; preserve it')
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.shape!=(count,) or value.dtype!=np.float32:raise ValueError('Exact downloaded latent schema differs')
            del value
            digest=u.sha(path);current=path.stat()
            if (current.st_ino,current.st_size,current.st_mtime_ns,current.st_nlink)!=(s.st_ino,s.st_size,s.st_mtime_ns,1) or not old.closed(path):
                raise ValueError('Retired downloaded cache changed during inventory')
            row=dict(path=str(path),sha256=digest,downloaded_index_md5=entry['md5'],bytes=s.st_size,
                     uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,discarded=False)
            records.append(row);u.write(REPORT/'partial.json',dict(records=records))
            path.unlink();row['discarded']=True;u.write(REPORT/'partial.json',dict(records=records))
    for rel,digest in bindings.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Protected own input manifest changed')
    u.write(REPORT/'verification.json',dict(state='complete_authorized_index_verified_downloaded_latent_retirement',records=records,
            logical_downloaded_bytes_discarded=sum(r['bytes'] for r in records),free_before=before,free_after=shutil.disk_usage(u.ROOT).free,
            own_models_inputs_checkpoints_modified=False,previous_lossless_archives_modified=False,qualifying_outcomes_read=False))
    print('Retired downloaded latent vectors:',len(records),'bytes:',sum(r['bytes'] for r in records),'freeGiB:',shutil.disk_usage(u.ROOT).free/2**30,flush=True)


if __name__=='__main__':main()
