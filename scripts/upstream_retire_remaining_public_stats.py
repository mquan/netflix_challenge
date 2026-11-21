#!/usr/bin/env python3
"""Journal and discard only three obsolete public-dependent blending caches."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import upstream_retire_public_stat_caches as prior

ROOT,OLD,ops,identity=prior.u.ROOT,prior.OLD,prior.ops,prior.identity
ARCHIVE_PROOF=ROOT/'experiments/upstream-retired-public-stats-archive/verification.json'
SPECS=[
 ('artifacts/public-new/weighted/bank','artifacts/public-new/production-dev-manifest.json',
  '2ebe131109e01ff6bc05a0c87546740086cb8f4a287162b16a7798d18ca11f18',536,
  '9a5f2658b38d5de8f7ce8a6132738041517e2dcf8cade06b0526197b68e0463e'),
 ('artifacts/public-bank/union466-full-tree-s1','artifacts/public-bank/merged-manifest.json',
  'a21e2e93d8f6618f3b8f8b77517ea6a4b611fd21475ca1a8ca4c62cf7b410591',387,
  '98c0d35438c4033dedf4c5810f0025c05dbaf01c394b29a2ecd93060f0d50e80'),
 ('artifacts/public-bank/union-tree-s1','artifacts/public-bank/columns/manifest.json',
  'e65e2b90ce645f7ac6d753475137de137615e9cc87b998618b79b8d7946417bc',122,
  '7bad7d2095fa17353387418a9266806dfdb882b68b6516b5970d212e8e4a2842'),
]
PROOFS=[
 'upstream-own14-matched391-calibration/independent-canonical-verification.json',
 'upstream-own391-paired-h128-mps/independent-canonical-verification.json',
 'upstream-own391-paired-h512-mps/independent-canonical-verification.json',
 'upstream-own391-fold-bag/independent-verification.json',
 'upstream-original1200/independent-verification.json',
 'upstream-original1200/execution-plan.json',
 'upstream-complete-cf-delta/input-manifest.json',
 'upstream-complete-cf-delta/input-verification.json',
 'upstream-complete-cf-delta/fit-manifest.json',
 'upstream-own391-cpu-tree/manifest.json',
]


def report(attempt):
    if type(attempt) is not int or attempt<1:raise ValueError('Positive bounded report attempt required')
    return ROOT/('experiments/upstream-retire-remaining-public-stats-%d'%attempt)


def classify(spec,record):
    prefix,manifest_name,manifest_sha,count,_=spec
    if record['config']['output']!=prefix or record['config']['public']!=manifest_name:
        raise ValueError('Exact original public-dependent cache configuration required')
    manifest=OLD/manifest_name
    if ops.sha(manifest)!=manifest_sha or record['public_manifest_sha256']!=manifest_sha:
        raise ValueError('Pinned original downloaded-public manifest required')
    rows=json.loads(manifest.read_text())['columns']
    matched=[r for r in rows if record['inputs'].get(r['path'])==r['sha256']]
    if len(rows)!=count or len(matched)!=count or not all('/artifacts/public-' in r['path'] for r in rows):
        raise ValueError('Every pinned downloaded-public input must match original cache provenance')
    return count


def regular_unbound(path,tracked,protected,old_root=OLD):
    s=path.lstat()
    if (not stat.S_ISREG(s.st_mode) or path.is_symlink() or path.resolve()!=path or
        s.st_nlink!=1 or s.st_flags or
        str(path.relative_to(old_root)).encode() in tracked or (s.st_dev,s.st_ino) in protected):
        raise ValueError('Only exact untracked closed single-link unbound public caches may be discarded')
    if subprocess.check_output(['/usr/bin/xattr',str(path)],text=True).strip():
        raise ValueError('Public cache extended attributes must be absent')
    return s


def protected_state():
    protected={};proofs={}
    # Protect all current own artifacts, including outputs created since older declarations.
    for p in (ROOT/'artifacts').rglob('*'):
        if p.is_file():
            s=p.stat();protected[(s.st_dev,s.st_ino)]=str(p)
    for name in PROOFS:
        p=ROOT/'experiments'/name;v=json.loads(p.read_text());proofs[str(p)]=ops.sha(p)
        bindings={**v.get('bindings',{}),**v.get('source_sha256',{}),**v.get('outputs',{})}
        if not bindings:raise ValueError('Nonempty actual own scientific proof closure required: '+name)
        for name in bindings:
            source=ROOT/name
            if source.is_file():
                s=source.stat();protected[(s.st_dev,s.st_ino)]=str(source)
    return protected,proofs


def prepare(attempt):
    target=report(attempt)
    if target.exists():raise ValueError('Existing preparation evidence retained')
    protected,proofs=protected_state()
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=OLD).split(b'\0'))
    archives={r['archive']:r for r in json.loads(ARCHIVE_PROOF.read_text())['records']}
    paths=[OLD/(s[0]+'.stats.npz.zst') for s in SPECS]
    if not prior.closed(paths):raise ValueError('All three obsolete public caches must be closed')
    records=[];metadata={str(ARCHIVE_PROOF):ops.sha(ARCHIVE_PROOF)}
    for spec,path in zip(SPECS,paths):
        s=regular_unbound(path,tracked,protected);inputs=OLD/(spec[0]+'.inputs.json')
        count=classify(spec,json.loads(inputs.read_text()));old=archives[str(path)]
        if old['archive_sha256']!=spec[4] or old['archive_bytes']!=s.st_size or ops.sha(path)!=spec[4]:
            raise ValueError('Exact previously verified public statistic archive required')
        if Path(old['original']).exists():raise ValueError('Unexpected restored public cache must be reviewed')
        with path.open('rb') as f:
            if f.read(4)!=bytes.fromhex('28b52ffd'):raise ValueError('Original Zstandard envelope required')
        for p in (inputs,OLD/spec[1]):metadata[str(p)]=ops.sha(p)
        records.append(dict(path=str(path),identity=identity.identity(path),sha256=spec[4],
            allocated_bytes=s.st_blocks*512,matched_downloaded_public_inputs=count,original_uncompressed_path=old['original']))
    for p in [Path(__file__),ROOT/'scripts/verify_remaining_public_stat_retirement.py',Path(prior.__file__),Path(ops.__file__),Path(identity.__file__)]:
        metadata[str(p)]=ops.sha(p)
    target.mkdir();prior.write(target/'plan.json',dict(state='declared_three_authorized_obsolete_public_dependent_stat_archive_discards',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),records=records,protected_scientific_proof_sha256=proofs,
        preserved_metadata_sha256=metadata,protected_current_inode_count=len(protected),
        authorization='User authorized removal of public-bank results because they are not our independent training effort.',
        policy='Intentionally discard only these three public-dependent blending-statistic archives after exact provenance/SHA/inode/closed checks. Preserve every independent model/checkpoint/prediction/dataset, all own working state and all original provenance metadata. No process signals or numerical fitting. These three obsolete public caches cease to be restorable by the historical archive helper.'))
    prior.durable(target);prior.durable(target.parent)
    print('Prepared only; no file deleted. Public-dependent cache bytes:',sum(r['identity']['bytes'] for r in records),flush=True)


def retire(attempt):
    target=report(attempt);p=target/'plan.json';plan=json.loads(p.read_text())
    if (target/'verification.json').exists() or (target/'partial.json').exists():
        raise ValueError('Existing retirement/partial evidence retained; no automatic retry')
    expected={str(OLD/(s[0]+'.stats.npz.zst')) for s in SPECS}
    if {r['path'] for r in plan['records']}!=expected or len(plan['records'])!=3:
        raise ValueError('Exact three-file declaration required')
    protected,_=protected_state();tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=OLD).split(b'\0'))
    def kept():
        for name,digest in {**plan['protected_scientific_proof_sha256'],**plan['preserved_metadata_sha256']}.items():
            if ops.sha(name)!=digest:raise ValueError('Own proof or preserved metadata changed; no deletion permitted')
    kept();before=shutil.disk_usage(ROOT).free;removed=[]
    for row in plan['records']:
        path=Path(row['path']);regular_unbound(path,tracked,protected)
        if (identity.identity(path)!=row['identity'] or ops.sha(path)!=row['sha256'] or
            identity.identity(path)!=row['identity'] or not prior.closed([path])):
            raise ValueError('Exact unchanged closed public archive required at unlink')
        kept()
        path.unlink();prior.durable(path.parent);removed.append(str(path))
        prior.write(target/'partial.json',dict(removed=removed,plan_sha256=ops.sha(p)))
    kept()
    prior.write(target/'verification.json',dict(state='complete_three_authorized_public_stat_archives_retired',
        plan_sha256=ops.sha(p),removed=removed,logical_bytes_discarded=sum(r['identity']['bytes'] for r in plan['records']),
        free_before_bytes=before,free_after_bytes=shutil.disk_usage(ROOT).free,
        independent_model_checkpoint_prediction_or_dataset_files_removed=False,
        original_metadata_preserved=True,qualifying_outcomes_read=False,processes_changed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--retire',action='store_true')
    args=parser.parse_args();(prepare if args.prepare else retire)(args.attempt)
