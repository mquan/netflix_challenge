#!/usr/bin/env python3
"""Share only independently byte-identical inactive caches; keep every path and hash."""
from pathlib import Path
import datetime as dt
import fcntl
import hashlib
import os
import shutil
import subprocess
import uuid
import upstream_repro as u
REPORT=u.ROOT/'experiments/upstream-exact-cache-links'
PAIRS=[('upstream-source-extension','upstream-extension-context','design.npy'),
       ('upstream-source-extension','upstream-extension-context','features.npy'),
       ('upstream-context-blend','upstream-voting-blend','design.npy'),
       ('upstream-context-blend','upstream-voting-blend','features.npy')]

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()
def signature(p):
    s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode,uid=s.st_uid,gid=s.st_gid,links=s.st_nlink)

def compare(a,b):
    hashes=[hashlib.sha256(),hashlib.sha256()]
    with a.open('rb') as x,b.open('rb') as y:
        if hasattr(fcntl,'F_NOCACHE'):
            fcntl.fcntl(x.fileno(),fcntl.F_NOCACHE,1);fcntl.fcntl(y.fileno(),fcntl.F_NOCACHE,1)
        while True:
            aa,bb=x.read(4<<20),y.read(4<<20)
            if aa!=bb:raise ValueError('Distinct cache bytes must never be linked')
            if not aa:break
            hashes[0].update(aa);hashes[1].update(bb)
    if hashes[0].hexdigest()!=hashes[1].hexdigest():raise ValueError('Complete file hashes differ')
    return hashes[0].hexdigest()

def run():
    if REPORT.exists():raise ValueError('Existing cache-link audit preserved')
    REPORT.mkdir(parents=True);records=[];bound={};planned=[]
    for source,target,name in PAIRS:
        a,b=[u.ROOT/'artifacts'/v/name for v in [source,target]]
        ma,mb=[u.ROOT/'experiments'/v/'manifest.json' for v in [source,target]]
        da,db=[u.read(p)['prepared_sha256'][name] for p in [ma,mb]]
        sa,sb=signature(a),signature(b)
        if da!=db or sa['size']!=sb['size'] or sa['inode']==sb['inode'] or sb['links']!=1:
            raise ValueError('Only equal declared hashes on distinct single-link target inodes allowed')
        if any(sa[k]!=sb[k] for k in ['device','mode','uid','gid']):raise ValueError('File metadata differs')
        for p in [ma,mb]:bound[str(p.relative_to(u.ROOT))]=u.sha(p)
        planned.append(dict(canonical=str(a.relative_to(u.ROOT)),target=str(b.relative_to(u.ROOT)),sha256=da,
            canonical_before=sa,target_before=sb))
    bound[str(Path(__file__).relative_to(u.ROOT))]=u.sha(Path(__file__))
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=bound,planned=planned,
        rule='Require two equal immutable manifest hashes and a complete independent byte comparison. Exclude every open candidate. Atomically replace only the redundant target inode with a hardlink to its existing exact byte-identical canonical file. Keep all filenames, bytes, permissions, owners, shapes, prediction/model/input values and original manifest hashes. Never touch distinct caches or model/checkpoint/data files.',qualifying_labels_read=False))
    paths=[str(u.ROOT/r[k]) for r in planned for k in ['canonical','target']]
    opened=subprocess.run(['lsof','-Fpn','--',*paths],capture_output=True,text=True)
    if opened.returncode not in [0,1] or opened.stdout.strip():raise ValueError('A candidate is open or cannot be audited; no linking allowed')
    before=shutil.disk_usage(u.ROOT).free
    # Verify every pair before publishing any filesystem mutation.
    for r in planned:
        a,b=[u.ROOT/r[k] for k in ['canonical','target']]
        if compare(a,b)!=r['sha256']:raise ValueError('Actual complete bytes differ from original declarations')
        if signature(a)!=r['canonical_before'] or signature(b)!=r['target_before']:raise ValueError('Candidate changed during comparison')
    for r in planned:
        a,b=[u.ROOT/r[k] for k in ['canonical','target']]
        opened=subprocess.run(['lsof','-Fpn','--',str(a),str(b)],capture_output=True,text=True)
        if opened.returncode not in [0,1] or opened.stdout.strip():raise ValueError('Candidate opened after verification; preserve remaining paths')
        if signature(a)!=r['canonical_before'] or signature(b)!=r['target_before']:raise ValueError('Candidate changed before atomic replacement')
        temp=b.with_name(b.name+'.link-'+uuid.uuid4().hex);os.link(a,temp);os.replace(temp,b)
        directory=os.open(b.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        if u.sha(a)!=r['sha256'] or u.sha(b)!=r['sha256'] or a.stat().st_ino!=b.stat().st_ino:
            raise ValueError('Retained path/hash/inode verification failed')
        records.append(dict(**r,canonical_after=signature(a),target_after=signature(b),full_byte_comparison_passed=True,
            both_original_paths_and_hashes_preserved=True,candidate_open_handles=0))
        u.write(REPORT/'partial-receipt.json',dict(observed=stamp(),records=records))
    for name,digest in bound.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original bound manifest/source changed')
    after=shutil.disk_usage(u.ROOT).free
    u.write(REPORT/'receipt.json',dict(state='complete_exact_inactive_cache_storage_shared',observed=stamp(),records=records,
        source_sha256=bound,plan_sha256=u.sha(REPORT/'plan.json'),all_original_paths_bytes_and_manifest_hashes_preserved=True,
        datasets_models_checkpoints_predictions_and_distinct_caches_removed=False,logical_redundant_bytes=sum(r['target_before']['size'] for r in records),
        disk_free_bytes_before=before,disk_free_bytes_after=after,observed_free_byte_increase=after-before,
        measurement_caveat='Concurrent training/checkpoints can change free space during this audit; this is the observed net change, not a claim of isolated physical extent savings.',qualifying_labels_read=False))
    print(dict(pairs=len(records),logical_duplicate_bytes=sum(r['target_before']['size'] for r in records),observed_free_increase=after-before),flush=True)

if __name__=='__main__':run()
