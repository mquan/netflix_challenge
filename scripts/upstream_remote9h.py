#!/usr/bin/env python3
"""Recover our completed remote qualifying outputs and transfer own CF48 state."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PIN='49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c'
RECEIPT_SHA='12421a92ea6495ece7347ecf8e7d794c52d52c8e1f14e7fbc5914e455708dac7'


def sha(path,algorithm='sha256'):
    digest=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def vector(path,rows):
    values=np.load(path,mmap_mode='r',allow_pickle=False)
    if values.shape!=(rows,) or values.dtype!=np.dtype('float32') or not np.isfinite(values).all():
        raise ValueError('Invalid own prediction: '+str(path))
    return values


def export_temporal(tree,output):
    receipt_path=ROOT/'experiments/upstream-remote-12h/tsvdx5-150o/native-receipt.json'
    receipt=read(receipt_path)
    if sha(receipt_path)!=RECEIPT_SHA or receipt['source_commit']!=PIN or receipt['job']!='tsvdx5-150o':
        raise ValueError('Our exact completed remote native receipt required')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=tree,text=True).splitlines()
    if actual!=PIN or set(changed)-{'data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Original remote native source pin/tracked code changed')
    if not receipt.get('finished') or len(receipt['outputs'])!=63:raise ValueError('Complete original63-output receipt required')
    ratings=np.load(tree/'data/qual/ratings.npy',mmap_mode='r')
    if ratings.shape!=(2817131,) or np.any(ratings):raise ValueError('Qualifying targets must remain all zero')
    for relative,digest in receipt['outputs'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not relative.startswith('preds_new/'):
            raise ValueError('Native output outside owned prediction tree')
        if sha(tree/path)!=digest:raise ValueError('Own native output changed: '+relative)
    if output.exists():raise ValueError('Existing export preserved; use a fresh output directory')
    output.mkdir(parents=True);records=[]
    for suffix in ['', '_ep06','_ep16','_ep24']:
        name='tsvdx5-150o'+suffix;source=tree/'preds_new'/(name+'.qual.npy')
        values=vector(source,2817131);target=output/(name+'.qual.npz')
        np.savez_compressed(target,prediction=values)
        with np.load(target,allow_pickle=False) as saved:
            if saved.files!=['prediction'] or not np.array_equal(saved['prediction'],values):raise ValueError('Lossless export differs')
        records.append(dict(name=name,rows=len(values),source_npy_sha256=sha(source),npz=target.name,
            npz_sha256=sha(target),prediction_bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    shutil.copyfile(receipt_path,output/'native-receipt.json')
    report=dict(state='verified_complete_own_remote_temporal_qualifying_export',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_pin=PIN,native_receipt_sha256=RECEIPT_SHA,all63_native_outputs_rehashed=True,
        qual_identity_sha256={p.name:sha(p) for p in (tree/'data/qual').glob('*.npy')},
        qualifying_targets_all_zero=True,qualifying_labels_read=False,records=records,
        provenance='Our completed original two-phase source job, recovered from its own remote native output tree. No released/pretrained prediction input.')
    write(output/'manifest.json',report);print(json.dumps(report),flush=True)


def package(coordinator,archive,metadata):
    base=coordinator/'artifacts/upstream-cfnade-sparse-rows';tree=base/'source'
    plan_path=coordinator/'experiments/upstream-cfnade48/plan.json';plan=read(plan_path)
    if archive.exists() or archive.with_suffix('').exists():raise ValueError('Existing package preserved')
    stage=archive.parent/'cfnade48-package'
    if stage.exists():raise ValueError('Existing package staging preserved')
    stage.mkdir(parents=True);phases={}
    for section in ['source_sha256','input_sha256','binaries']:
        for relative,digest in plan[section].items():
            if sha(coordinator/relative)!=digest:raise ValueError('Frozen local CF48 input changed: '+relative)
    for relative in plan['source_sha256']:link(coordinator/relative,stage/'evidence'/relative)
    for p in tree.iterdir():
        if p.is_file() and not p.name.startswith('.'):link(p,stage/'native-source'/p.name)
    for p in (tree/'src').rglob('*'):
        if p.is_file():link(p,stage/'native-source'/p.relative_to(tree))
    for name in ['cfnade48-stage','cfnade48-parity']:link(base/'target/release'/name,stage/'bin'/name)
    for train,query,rows in [('trainx','probex',1128164),('fulltrain','qual',2817131)]:
        source=base/'preset48-checkpoints'/train;pointer=read(source/'latest.json');epoch=pointer['epoch']
        old=source/f'epoch-{epoch:05}';state=read(old/'state.json')
        if sha(old/'state.json','md5')!=pointer['manifest_md5'] or len(state['files'])!=84:
            raise ValueError('Complete84-array checkpoint required')
        if (state['epoch']!=epoch or state['identity']['source']!=plan['source_id'] or
                state['identity']['data']!=plan['data_id'] or state['identity']['config']!=plan['description']):
            raise ValueError('Checkpoint source/data/config differs')
        target=stage/'checkpoints'/train/f'epoch-{epoch:05}'
        for name,digest in state['files'].items():
            p=old/name
            if sha(p,'md5')!=digest or not np.isfinite(np.load(p,mmap_mode='r')).all():raise ValueError('Checkpoint tensor changed')
            link(p,target/name)
        link(old/'state.json',target/'state.json');write(target.parent/'latest.json',pointer)
        outputs={}
        for ep in range(1,epoch+1):
            name=f'cfnade-48_ep{ep:02}.{query}.npy';p=tree/'preds_preset48'/name
            vector(p,rows);link(p,stage/'native-source/preds_preset48'/name);outputs[name]=sha(p)
        phases[train]=dict(epoch=epoch,query=query,rows=rows,checkpoint_manifest_md5=pointer['manifest_md5'],prediction_sha256=outputs)
    write(stage/'origin-plan.json',plan)
    manifest=dict(source_pin=PIN,original_plan_sha256=sha(plan_path),source_id=plan['source_id'],data_id=plan['data_id'],
        configuration=plan['description'],phases=phases,architecture=platform.machine(),
        input_sha256={k.split('/data/',1)[1]:v for k,v in plan['input_sha256'].items()},
        files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()},
        qualifying_labels_read=False,scope='Our own exact160/48/95 trajectory:84 tensors, Adam/SWA/RNG, both phase checkpoints and all earlier native snapshots. No raw datasets or downloaded pretrained features.')
    write(stage/'manifest.json',manifest);archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(stage,arcname='cfnade48-package')
    result=dict(asset=archive.name,archive_sha256=sha(archive),bytes=archive.stat().st_size,
        manifest_sha256=sha(stage/'manifest.json'),phases=phases,release_tag='remote9h-cfnade48-20261119',
        source_pin=PIN,source_id=plan['source_id'],data_id=plan['data_id'],qualifying_labels_read=False)
    write(metadata,result);print(json.dumps({k:v for k,v in result.items() if k!='phases'}),flush=True)


def verify_bundle(bundle,data):
    manifest=read(bundle/'manifest.json');plan=read(bundle/'origin-plan.json')
    for relative,digest in manifest['files'].items():
        if sha(bundle/relative)!=digest:raise ValueError('Bundle member changed: '+relative)
    for relative,digest in manifest['input_sha256'].items():
        if sha(data/relative)!=digest:raise ValueError('Remote data differs: '+relative)
    if np.any(np.load(data/'qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets are not zero')
    if platform.machine()!=manifest['architecture']:raise ValueError('Use the frozen sources to rebuild and independently verify on this architecture')
    description=subprocess.check_output([str(bundle/'bin/cfnade48-stage'),'--describe'],text=True).strip()
    if description!=plan['description']:raise ValueError('Original95-epoch configuration differs')
    subprocess.run([str(bundle/'bin/cfnade48-parity')],check=True,cwd=bundle/'native-source')
    print('Bundle, exact data/config and native160/48 parity verified. Follow REMOTE_9H_HANDOFF.md for supervised resume.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('export-temporal');p.add_argument('--native-tree',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('package-cf48');p.add_argument('--coordinator-root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p=sub.add_parser('verify-bundle');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='export-temporal':export_temporal(args.native_tree.resolve(),args.output.resolve())
    elif args.action=='package-cf48':package(args.coordinator_root.resolve(),args.archive.resolve(),args.metadata.resolve())
    else:verify_bundle(args.bundle.resolve(),args.data_root.resolve())
