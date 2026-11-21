#!/usr/bin/env python3
"""Retire five explicitly authorized public-bank statistic caches, preserving evidence."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import upstream_repro as u
import upstream_complete_trajectory_ops as ops
import upstream_read_cache_policy as identity

OLD = Path('/Users/quan/workspace/netflix_prize')
PUBLIC = OLD/'artifacts/public-new'
NAMES = ['late539-bank','early539-bank','MT8-537-bank','A2h537-bank','ensemble-tree-s1']
REPORT = u.ROOT/'experiments/upstream-retire-public-stat-caches'
CLOSURES = [
    'upstream-own14-matched391-calibration/frozen.json',
    'upstream-complete-cf-delta/plan.json',
    'upstream-original1200/execution-plan.json',
    'upstream-complete-trajectory-data-queue/plan.json',
    'upstream-complete-trajectory-neural-queue/plan.json',
    'upstream-complete-trajectory-fold-bag-queue/plan.json',
    'upstream-own15-complete-trajectory-queue/plan.json',
    *['upstream-remote-weekend-return-queue/'+job+'/plan.json'
      for job in ['s3-rbm400pf','s3-tx300r','s3-rbm800p']],
    'upstream-own391-cpu-tree/manifest.json',
]


def durable(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def closed(paths):
    result=subprocess.run(['/usr/sbin/lsof','-Fpn','--',*[str(p) for p in paths]],capture_output=True,text=True)
    return result.returncode==1 and not result.stdout.strip() and not result.stderr.strip()


def write(path, value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);durable(path.parent)


def main():
    if REPORT.exists():raise ValueError('Existing public retirement evidence retained')
    if PUBLIC.is_symlink() or PUBLIC.resolve()!=PUBLIC:raise ValueError('Exact public cache directory required')
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=OLD).split(b'\0'))
    protected={};proofs={}
    for name in CLOSURES:
        path=u.ROOT/'experiments'/name;record=u.read(path);proofs[str(path)]=ops.sha(path)
        bindings=record.get('source_sha256',record.get('bindings',{}))
        if not bindings:raise ValueError('Nonempty actual scientific source closure required')
        for relative in bindings:
            source=u.ROOT/relative
            if source.is_file():
                s=source.stat();protected[(s.st_dev,s.st_ino)]=str(source)
    paths=[PUBLIC/(name+'.stats.npz.zst') for name in NAMES]
    if not closed(paths):raise ValueError('All five public caches must have no open users')
    records=[];evidence={str(Path(__file__)):ops.sha(Path(__file__))}
    for name,path in zip(NAMES,paths):
        s=path.lstat()
        if (not stat.S_ISREG(s.st_mode) or path.is_symlink() or s.st_nlink!=1 or s.st_flags
            or str(path.relative_to(OLD)).encode() in tracked or (s.st_dev,s.st_ino) in protected):
            raise ValueError('Only untracked, single-link, unflagged, unbound public statistic caches may be retired')
        inputs=PUBLIC/(name+'.inputs.json');result=PUBLIC/(name+'.json');original=u.read(inputs)
        if original['config']['output']!='artifacts/public-new/'+name:
            raise ValueError('Exact original public-bank output prefix required')
        manifest=OLD/original['config']['public']
        if (original['config']['public']!='artifacts/public-new/production-dev-manifest.json'
            or ops.sha(manifest)!=original['public_manifest_sha256']
            or sum('/artifacts/public-new/prior-columns/' in p or '/artifacts/public-new/columns/' in p
                   for p in original['inputs'])<500):
            raise ValueError('Original manifest and at least500 downloaded-public inputs required')
        # Stats archives are deliberately discarded, not decoded into fitting input.
        with path.open('rb') as stream:
            if stream.read(4)!=bytes.fromhex('28b52ffd'):raise ValueError('Original Zstandard statistic-cache envelope required')
        for keep in [inputs,result,manifest]:evidence[str(keep)]=ops.sha(keep)
        records.append(dict(path=str(path),identity=identity.identity(path),sha256=ops.sha(path),
                            inputs_sha256=evidence[str(inputs)],public_manifest_sha256=original['public_manifest_sha256']))
    for row in records:
        if identity.identity(Path(row['path']))!=row['identity']:raise ValueError('Public cache changed during inventory')
    if not closed(paths):raise ValueError('Public cache acquired an open user')
    REPORT.mkdir();before=shutil.disk_usage(u.ROOT).free
    plan=dict(state='declared_five_authorized_obsolete_public_bank_statistic_cache_discards',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),records=records,
        protected_proof_sha256=proofs,preserved_metadata_sha256=evidence,
        authorization='User explicitly authorized removal of public-bank results because they are not our independent effort.',
        policy='Discard ONLY five named obsolete public-derived stats.npz.zst caches. Preserve original inputs/results/manifests, all models/checkpoints/predictions/data/source, every current own source inode and live process. Full cache SHA/metadata journal fsynced before unlink. No score selection or cache values used in fitting. Intentional discard, not lossless archival.')
    write(REPORT/'plan.json',plan);durable(REPORT);durable(REPORT.parent)
    for path,digest in {**proofs,**evidence}.items():
        if ops.sha(path)!=digest:raise ValueError('Protected source proof or metadata changed')
    removed=[]
    for row in records:
        path=Path(row['path'])
        if (identity.identity(path)!=row['identity'] or ops.sha(path)!=row['sha256']
            or identity.identity(path)!=row['identity'] or not closed([path])):
            raise ValueError('Exact closed unchanged public cache required at unlink')
        path.unlink();durable(path.parent);removed.append(str(path))
        write(REPORT/'partial.json',dict(removed=removed,plan_sha256=ops.sha(REPORT/'plan.json')))
    for path,digest in {**proofs,**evidence}.items():
        if ops.sha(path)!=digest:raise ValueError('Protected metadata or scientific proof changed')
    if any(path.exists() for path in paths):raise ValueError('All five retired public caches must be accounted')
    write(REPORT/'verification.json',dict(state='complete_five_authorized_public_stat_caches_retired_and_all_protected_proofs_equal',
        plan_sha256=ops.sha(REPORT/'plan.json'),removed=removed,logical_bytes_discarded=sum(r['identity']['bytes'] for r in records),
        free_before_bytes=before,free_after_bytes=shutil.disk_usage(u.ROOT).free,
        numerical_own_files_removed=False,models_or_checkpoints_removed=False,
        original_metadata_preserved=True,qualifying_outcomes_read=False,processes_changed=False))


if __name__=='__main__':main()
