"""Retire only manifest/index-exact downloaded qualifying prediction inodes."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import numpy as np
import tomli
import upstream_retire_remaining_public_stats as previous

ROOT,OLD=previous.ROOT,previous.OLD
REPORT=ROOT/'experiments/upstream-retire-public-qual-arrays'
INDEX=OLD/'artifacts/public-source-audit-index-20261115.toml'
INDEX_SHA256='8714114b33345a63f5b254348015a5ff04427c9c901ffead120e6d4bc46e15e8'
MANIFEST=OLD/'artifacts/public-qual/manifest.json'
DOWNLOAD_PLAN=OLD/'artifacts/public-qual/plan.json'
ALLOWED=tuple(OLD/'artifacts'/name for name in (
    'public-qual','public-new/qual','public-probe-blenders/source','public-probe-blenders-expanded/source'))


def read(path):return json.loads(Path(path).read_text())


def sha(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write(path,value,exclusive=False):
    path=Path(path)
    if exclusive and path.exists():raise ValueError('Existing retirement evidence retained')
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def allowed(path):
    return any(path.is_relative_to(folder) for folder in ALLOWED)


def snapshot(s):
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,nlink=s.st_nlink,
        mode=s.st_mode,uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,flags=s.st_flags)


def closed(paths):
    p=subprocess.run(['/usr/sbin/lsof','-Fpn','--',*[str(p) for p in paths]],capture_output=True,text=True)
    return p.returncode==1 and not p.stdout.strip() and not p.stderr.strip()


def classify(paths,anchors,index,protected,tracked):
    """Every alias must be eligible; at least one actual alias is a download manifest entry."""
    if not paths or len(set(paths))!=len(paths):raise ValueError('Complete distinct public aliases required')
    first=paths[0];s=first.lstat();key=(s.st_dev,s.st_ino)
    if key in protected:raise ValueError('Own scientific input/output inode protected')
    if len(paths)!=s.st_nlink:raise ValueError('Every hardlink must be accounted for')
    anchor=[anchors[str(p)] for p in paths if str(p) in anchors]
    if not anchor:raise ValueError('Actual original download-manifest alias required')
    for p in paths:
        a=p.lstat()
        if (not allowed(p) or not p.name.endswith('.qual.npy') or not stat.S_ISREG(a.st_mode)
            or p.is_symlink() or p.resolve()!=p or (a.st_dev,a.st_ino)!=key
            or a.st_nlink!=len(paths) or a.st_flags or str(p.relative_to(OLD)).encode() in tracked):
            raise ValueError('Only unchanged untracked regular public prediction aliases eligible')
    for row in anchor:
        entry=index.get(row['source'])
        if (not entry or not row['source'].endswith('.qual.npy') or Path(row['source']).name!=first.name
            or entry['size']!=s.st_size or entry['md5']!=row['md5']):
            raise ValueError('Original qualifying prediction download index differs')
    digests={k:hashlib.new(k) for k in ('sha256','md5')}
    with first.open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version!=(1,0):raise ValueError('Original public float32 NPY header required')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        if shape!=(2817131,) or fortran or dtype!=np.dtype('<f4') or f.tell()+4*2817131!=s.st_size:
            raise ValueError('Original full qualifying prediction vector required')
        f.seek(0)
        for block in iter(lambda:f.read(2**20),b''):
            for h in digests.values():h.update(block)
    actual={k:h.hexdigest() for k,h in digests.items()}
    if any(row['sha256']!=actual['sha256'] or row['md5']!=actual['md5'] for row in anchor):
        raise ValueError('Not the exact previously downloaded public numerical bytes')
    return dict(paths=[str(p) for p in paths],metadata=snapshot(s),**actual,
        download_manifest_entries=anchor,index_entries=[index[row['source']] for row in anchor])


def source_bindings():
    files=[INDEX,MANIFEST,DOWNLOAD_PLAN,Path(__file__),
        ROOT/'scripts/verify_upstream_public_qual_retirement.py',
        ROOT/'scripts/test_public_qual_retirement.py',Path(previous.__file__),
        Path(previous.prior.__file__),Path(previous.prior.ops.__file__),Path(previous.prior.u.__file__)]
    return {str(p):sha(p) for p in files}


def check_bindings(bindings):
    for name,digest in bindings.items():
        if sha(name)!=digest:raise ValueError('Original public provenance or protected own proof changed: '+name)


def manifests():
    # Same original index already bound by the earlier committed reference retirement.
    if sha(INDEX)!=INDEX_SHA256:raise ValueError('Fixed historical public download index required')
    index={row['path']:row for row in tomli.loads(INDEX.read_text())['file']}
    manifest=read(MANIFEST);plan=read(DOWNLOAD_PLAN)
    if manifest['plan']!=plan or plan['rows']!=2817131 or plan['labels_requested'] is not False:
        raise ValueError('Original unlabeled downloaded prediction plan required')
    anchors={row['path']:row for row in manifest['columns']}
    if len(anchors)!=len(manifest['columns']):raise ValueError('Distinct original downloaded paths required')
    entries={row['path']:row for row in plan['entries']}
    for path,row in anchors.items():
        p=Path(path)
        if p!=OLD/'artifacts/public-qual'/row['source'] or entries[row['source']]!=index[row['source']]:
            raise ValueError('Exact original download manifest and complete index membership required')
    return anchors,index


def prepare():
    if REPORT.exists():raise ValueError('Existing preparation/evidence retained')
    for folder in ALLOWED:
        if folder.is_symlink() or folder.resolve()!=folder:raise ValueError('Original public roots must not redirect')
    anchors,index=manifests();protected,proofs=previous.protected_state()
    for name in ('input-manifest.json','input-verification.json','manifest.json'):
        p=ROOT/'experiments/upstream-complete-trajectory-data'/name
        value=read(p);proofs[str(p)]=sha(p)
        for rel in value['source_sha256']:
            source=ROOT/rel
            if source.is_file():
                s=source.stat();protected[(s.st_dev,s.st_ino)]=str(source)
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=OLD).split(b'\0'))
    aliases={}
    for folder,dirs,files in os.walk(OLD/'artifacts',followlinks=False):
        for name in files:
            p=Path(folder)/name
            if p.is_symlink():continue
            s=p.stat();aliases.setdefault((s.st_dev,s.st_ino),[]).append(p)
    records=[];skipped=[]
    for key,paths in aliases.items():
        if not any(str(p) in anchors and p.name.endswith('.qual.npy') for p in paths):continue
        try:record=classify(sorted(paths),anchors,index,protected,tracked)
        except ValueError as exc:
            skipped.append(dict(paths=[str(p) for p in paths],reason=str(exc)));continue
        if any(subprocess.check_output(['/usr/bin/xattr',str(p)],text=True).strip() for p in paths):
            skipped.append(dict(paths=[str(p) for p in paths],reason='Extended attributes retained'));continue
        records.append(record)
    if not records:raise ValueError('No manifest/index-exact unbound downloaded prediction inodes')
    if not closed([Path(p) for row in records for p in row['paths']]):raise ValueError('Open or unobservable public candidate retained')
    bindings={**source_bindings(),**proofs};check_bindings(bindings)
    REPORT.mkdir();fsync_directory(REPORT.parent)
    write(REPORT/'plan.json',dict(state='prepared_exact_downloaded_public_qualifying_prediction_retirement',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),authorization='Existing user authorization to remove downloaded public-bank results because they are not our own effort.',
        source_sha256=bindings,allowed_roots=[str(p) for p in ALLOWED],records=records,skipped=skipped,
        logical_candidate_bytes=sum(row['metadata']['bytes'] for row in records),files_deleted=0,
        qualifying_outcomes_read=False,own_models_outputs_checkpoints_data_modified=False,
        policy='Only complete manifest-SHA256 and original download-index-MD5 exact prediction arrays with every public hardlink accounted for. Exclude every current own artifact and scientific proof inode. Never delete data, own predictions/models, archives or metadata. Preserve durable full pre-deletion journal. Intentional public-cache discard, not lossless archival.'),exclusive=True)
    print('Prepared',len(records),'public prediction inode groups; no files deleted',flush=True)


def discard():
    plan=read(REPORT/'plan.json')
    if plan['state']!='prepared_exact_downloaded_public_qualifying_prediction_retirement' or plan['allowed_roots']!=[str(p) for p in ALLOWED]:
        raise ValueError('Exact original preparation required')
    if (REPORT/'partial.json').exists():raise ValueError('Existing discard attempt retained; no automatic retry')
    check_bindings(plan['source_sha256']);anchors,index=manifests();protected,_=previous.protected_state()
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=OLD).split(b'\0'))
    before=shutil.disk_usage(ROOT).free
    progress=dict(state='discarding_only_prepared_downloaded_public_prediction_inodes',plan_sha256=sha(REPORT/'plan.json'),removed_paths=[])
    write(REPORT/'partial.json',progress,exclusive=True)
    for row in plan['records']:
        paths=[Path(p) for p in row['paths']]
        actual=classify(paths,anchors,index,protected,tracked)
        if actual!=row:raise ValueError('Prepared public prediction identity or bytes changed')
        if any(subprocess.check_output(['/usr/bin/xattr',str(p)],text=True).strip() for p in paths) or not closed(paths):
            raise ValueError('Prepared public prediction opened or attributes changed; retained')
        # Full row is already durable in plan.json before any unlink.
        for i,p in enumerate(paths):
            s=p.lstat();expected=row['metadata']
            if ((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink,s.st_mode,s.st_flags)!=
                (expected['device'],expected['inode'],expected['bytes'],expected['mtime_ns'],len(paths)-i,expected['mode'],expected['flags'])):
                raise ValueError('Unexpected new/changed alias before exact public discard')
            p.unlink();fsync_directory(p.parent)
            progress['removed_paths'].append(str(p));write(REPORT/'partial.json',progress)
    check_bindings(plan['source_sha256'])
    if any(Path(p).exists() for row in plan['records'] for p in row['paths']):raise ValueError('Prepared alias remains')
    after=shutil.disk_usage(ROOT).free
    write(REPORT/'verification.json',dict(state='complete_exact_downloaded_public_qualifying_prediction_retirement',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(REPORT/'plan.json'),
        source_sha256=plan['source_sha256'],inode_groups=len(plan['records']),removed_aliases=len(progress['removed_paths']),
        logical_bytes_discarded=plan['logical_candidate_bytes'],free_before=before,free_after=after,
        own_models_outputs_checkpoints_data_modified=False,scientific_processes_signalled=False,
        qualifying_outcomes_read=False,earlier_archives_and_provenance_modified=False),exclusive=True)
    print('Discarded only original downloaded public prediction groups:',len(plan['records']),'measured free gain',after-before,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--discard',action='store_true')
    args=parser.parse_args()
    if args.prepare and not args.discard:prepare()
    elif args.discard and not args.prepare:discard()
    else:parser.error('One explicit prepare/discard phase required')
