#!/usr/bin/env python3
"""Transfer our original checkpointed320/96/95 trajectory without raw datasets."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import numpy as np
import upstream_remote9h as io

ROOT=Path(__file__).resolve().parents[1]
PHASES={'trainx':('probex',1128164),'fulltrain':('qual',2817131)}


def safe_member(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe package member')
    target=root/p
    if target.is_symlink():raise ValueError('Package member must not be a symlink')
    return target


def phase_checkpoint(base,source_id,data_id,query,rows):
    pointer=io.read(base/'latest.json');directory=base/f"epoch-{pointer['epoch']:05}";state=io.read(directory/'state.json')
    identity=state['identity']
    if (io.sha(directory/'state.json','md5')!=pointer['manifest_md5'] or state['epoch']!=pointer['epoch'] or
            not 0<pointer['epoch']<=95 or len(state['files'])!=84 or
            identity['source']!=source_id or identity['data']!=data_id or identity['query']!=query or
            identity['query_rows']!=rows or io.read(base/'owner.json')!=identity or
            not all(token in identity['config'] for token in ['n_hidden: 320,','rank: 96,','n_epochs: 95,'])):
        raise ValueError('Original96 checkpoint pointer/owner/config/identity differs')
    for name,digest in state['files'].items():
        path=safe_member(directory,name)
        if Path(name).name!=name or io.sha(path,'md5')!=digest:raise ValueError('Checkpoint tensor differs')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.dtype!=np.dtype('float32') or not np.isfinite(values).all():raise ValueError('Invalid checkpoint tensor')
    return pointer,state,directory


def package(archive,metadata):
    import upstream_cfnade_canonical as reference
    reference.verify_candidate()
    base=ROOT/'artifacts/upstream-cfnade-checkpoint';tree=base/'source'
    plan_path=ROOT/'experiments/upstream-cfnade-parallel/plan.json';plan=io.read(plan_path)
    stage=archive.parent/'cfnade96-package'
    if archive.exists() or stage.exists():raise ValueError('Existing transfer package preserved')
    for relative,digest in plan['input_sha256'].items():
        if io.sha(ROOT/relative)!=digest:raise ValueError('Original96 source data changed')
    preparation=ROOT/'experiments/upstream-cfnade-checkpoint/preparation.json'
    proof=ROOT/'experiments/upstream-cfnade-checkpoint/verification.json';verified=io.read(proof)
    if io.sha(preparation)!=plan['source_id'] or io.sha(proof)!=plan['proof_sha256']:
        raise ValueError('Original96 code/proof identity differs')
    stage.mkdir(parents=True);shutil.copytree(tree/'src',stage/'native-source/src')
    for path in tree.iterdir():
        if path.is_file() and not path.name.startswith('.') and path.suffix in ['.toml','.lock','.md']:
            io.link(path,stage/'native-source'/path.name)
    for name,digest in {**plan['binaries'],**verified['binaries']}.items():
        source=base/'target/release'/name
        if io.sha(source)!=digest:raise ValueError('Original native binary changed')
        io.link(source,stage/'bin'/name)
    phases={}
    for train,(query,rows) in PHASES.items():
        source=base/'parallel-checkpoints'/train
        pointer,state,directory=phase_checkpoint(source,plan['source_id'],plan['data_id'],query,rows)
        destination=stage/'checkpoints'/train/f"epoch-{pointer['epoch']:05}"
        for name in state['files']:io.link(directory/name,destination/name)
        io.link(directory/'state.json',destination/'state.json')
        # Owner is included explicitly; the native protocol refuses unowned state.
        io.link(source/'owner.json',destination.parent/'owner.json')
        io.write(destination.parent/'latest.json',pointer)
        outputs={}
        for epoch in range(1,pointer['epoch']+1):
            name=f'cfnade-96_ep{epoch:02}.{query}.npy';path=tree/'preds_new'/name;io.vector(path,rows)
            target=stage/'native-source/preds_new'/name;target.parent.mkdir(parents=True,exist_ok=True)
            # Prediction files can be regenerated in place by a later local resume.
            shutil.copyfile(path,target);outputs[name]=io.sha(target)
        phases[train]=dict(epoch=pointer['epoch'],query=query,rows=rows,
            checkpoint_manifest_md5=pointer['manifest_md5'],prediction_sha256=outputs)
    for source,name in [(plan_path,'origin-plan.json'),(preparation,'source-preparation.json'),(proof,'source-verification.json')]:
        shutil.copyfile(source,stage/name)
    manifest=dict(source_pin=io.PIN,source_id=plan['source_id'],data_id=plan['data_id'],
        architecture=platform.machine(),phases=phases,
        input_sha256={name.split('/data/',1)[1]:digest for name,digest in plan['input_sha256'].items()},
        files={str(p.relative_to(stage)):io.sha(p) for p in stage.rglob('*') if p.is_file()},
        qualifying_labels_read=False,raw_datasets_included=False,
        scope='Own original320/96/95 training, exact84-array Adam/SWA/RNG checkpoints, owner identity and all completed epoch predictions. Partial state is not a completed model.')
    io.write(stage/'manifest.json',manifest)
    with tarfile.open(archive,'w:gz',compresslevel=1) as output:output.add(stage,arcname='cfnade96-package')
    result=dict(asset=archive.name,archive_sha256=io.sha(archive),bytes=archive.stat().st_size,
        manifest_sha256=io.sha(stage/'manifest.json'),source_pin=io.PIN,source_id=plan['source_id'],data_id=plan['data_id'],
        phases=phases,release_tag='remote9h-cfnade96-20261119',training_deadline_utc='2026-11-19T14:33:28Z',
        return_deadline_utc='2026-11-19T15:03:28Z',raw_datasets_included=False,qualifying_labels_read=False)
    io.write(metadata,result);print(json.dumps({k:v for k,v in result.items() if k!='phases'}),flush=True)


def verify_bundle(bundle,data):
    manifest=io.read(bundle/'manifest.json');plan=io.read(bundle/'origin-plan.json')
    if manifest['source_pin']!=io.PIN or manifest['raw_datasets_included'] is not False:
        raise ValueError('Original-source own-state package required')
    for relative,digest in manifest['files'].items():
        if io.sha(safe_member(bundle,relative))!=digest:raise ValueError('Package member changed: '+relative)
    if (io.sha(bundle/'source-preparation.json')!=manifest['source_id'] or
            io.sha(bundle/'source-verification.json')!=plan['proof_sha256'] or
            plan['source_id']!=manifest['source_id'] or plan['data_id']!=manifest['data_id']):
        raise ValueError('Original package source/data/proof differs')
    for relative,digest in manifest['input_sha256'].items():
        if io.sha(data/relative)!=digest:raise ValueError('Actual remote data differs: '+relative)
    if np.any(np.load(data/'qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets must remain zero')
    if platform.machine()!=manifest['architecture']:raise ValueError('Rebuild frozen sources and rerun parity on this architecture')
    for train,(query,rows) in PHASES.items():
        pointer,state,_=phase_checkpoint(bundle/'checkpoints'/train,manifest['source_id'],manifest['data_id'],query,rows)
        phase=manifest['phases'][train]
        if pointer['epoch']!=phase['epoch'] or state['identity']['train']!=train:raise ValueError('Phase checkpoint identity differs')
        for name,digest in phase['prediction_sha256'].items():
            path=bundle/'native-source/preds_new'/name;io.vector(path,rows)
            if io.sha(path)!=digest:raise ValueError('Transferred epoch prediction changed')
    env=dict(os.environ)
    for name in ['CF_CHECKPOINT_DIR','CF_CHECKPOINT_SOURCE','CF_CHECKPOINT_DATA','CF_EXIT_AFTER_CHECKPOINT','CF_PROOF_OUTPUT','CF_PROOF_INTERRUPT']:
        env.pop(name,None)
    subprocess.run([str(bundle/'bin/cfnade-fast-parity')],cwd=bundle/'native-source',env=env,check=True)
    print('Original96 package, owners,84 tensors per phase, actual data and native parity verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('package');p.add_argument('--archive',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p=sub.add_parser('verify-bundle');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='package':package(args.archive.resolve(),args.metadata.resolve())
    else:verify_bundle(args.bundle.resolve(),args.data_root.resolve())
