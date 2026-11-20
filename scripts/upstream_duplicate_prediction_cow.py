#!/usr/bin/env python3
"""Reclaim exact inactive prediction duplicates without removing logical files."""
from pathlib import Path
import collections
import datetime as dt
import filecmp
import json
import os
import shutil
import subprocess
import upstream_cow_cache_storage as safety
ROOT=safety.ROOT
REPORT=ROOT/'experiments/upstream-duplicate-prediction-cow'

def main():
    if REPORT.exists():raise ValueError('Existing storage evidence preserved')
    REPORT.mkdir()
    inventory=ROOT/'experiments/upstream-old-cache-storage/own-sample-inventory.json'
    candidates=[]
    for group in json.loads(inventory.read_text())['groups'][6:]:
        unique={}
        for row in group:
            p=Path(row['path'])
            if not p.exists() or p.suffix not in {'.npy','.f32'}:continue
            rel=p.relative_to(ROOT/'artifacts')
            if any(v in str(rel) for v in ['upstream-cfnade-checkpoint','upstream-mps-item-autorec','/data/']):continue
            s=p.lstat()
            if p.is_symlink() or s.st_nlink!=1 or s.st_flags or not p.is_file():continue
            unique[s.st_ino]=p
        if len(unique)>1:candidates.append(list(unique.values()))
    names=[str(p) for g in candidates for p in g]
    observed=subprocess.run(['/usr/sbin/lsof','-Fn','--',*names],capture_output=True,text=True)
    if observed.returncode not in [0,1] or observed.stderr.strip():raise ValueError('Uncertain open-file observation')
    opened={line[1:] for line in observed.stdout.splitlines() if line.startswith('n')}
    candidates=[g for g in candidates if not any(str(p) in opened for p in g)]
    safety.write(REPORT/'plan.json',dict(observed=safety.stamp(),script_sha256=safety.sha(Path(__file__)),
        inventory_sha256=safety.sha(inventory),candidates=[[str(p.relative_to(ROOT)) for p in g] for g in candidates],
        rule='Every full byte and SHA256 must agree before clone. Only single-link unflagged closed regular inactive prediction/cache copies. No data directories, live CF96 or raw item-model files. Preserve all logical paths/bytes/modes/owners/mtime with APFS separate-inode clones. No numerical/model/recipe changes.'))
    before=shutil.disk_usage(ROOT).free;records=[];logical=0
    for group in candidates:
        source=group[0];digest=safety.sha(source)
        if subprocess.check_output(['/usr/bin/xattr',str(source)]):continue
        for destination in group[1:]:
            if safety.sha(destination)!=digest or not filecmp.cmp(source,destination,shallow=False):continue
            if subprocess.check_output(['/usr/bin/xattr',str(destination)]):continue
            if len(subprocess.check_output(['/bin/ls','-lde',str(destination)],text=True).splitlines())!=1:continue
            safety.assert_closed([source,destination])
            old=safety.identity(destination)
            temp=destination.with_name(destination.name+'.exact-cow.tmp')
            if temp.exists():raise ValueError('Unexpected existing temporary file')
            subprocess.run(['/bin/cp','-c','-p',str(source),str(temp)],check=True)
            os.chmod(temp,old['mode']&0o7777);os.utime(temp,ns=(destination.stat().st_atime_ns,old['mtime_ns']))
            if safety.sha(temp)!=digest or safety.identity(destination)!=old:raise ValueError('Concurrent mutation or clone error')
            if temp.stat().st_uid!=old['uid'] or temp.stat().st_gid!=old['gid']:raise ValueError('Owner differs')
            safety.assert_closed([source,destination]);temp.replace(destination)
            current=safety.identity(destination)
            if any(current[k]!=old[k] for k in ['bytes','mtime_ns','mode','uid','gid','nlink','flags']):raise ValueError('Metadata differs')
            if current['ino']==source.stat().st_ino or safety.sha(destination)!=digest:raise ValueError('Independent identical logical clone required')
            logical+=old['bytes'];records.append(dict(source=str(source.relative_to(ROOT)),destination=str(destination.relative_to(ROOT)),sha256=digest,bytes=old['bytes'],before=old,after=current))
            safety.write(REPORT/'partial.json',dict(records=records,logical_bytes_shared=logical))
    safety.write(REPORT/'verification.json',dict(state='complete_exact_inactive_prediction_cow',observed=safety.stamp(),
        records=records,logical_bytes_shared=logical,free_before=before,free_after=shutil.disk_usage(ROOT).free,
        logical_files_removed=0,numerical_values_changed=False))
    print(len(records),logical,shutil.disk_usage(ROOT).free-before,flush=True)

if __name__=='__main__':main()
