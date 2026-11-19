#!/usr/bin/env python3
"""Verify a complete own temporal return in isolation; never control live jobs."""
import argparse
import datetime as dt
from pathlib import Path
import tarfile
import numpy as np
import upstream_remote9h as io

ROOT=Path(__file__).resolve().parents[1]
TOP='temporal-native-complete'


def extract_exact(archive,destination,outputs):
    """Reject links, duplicate paths and undeclared payloads before extraction."""
    expected={f'{TOP}/manifest.json',f'{TOP}/native-receipt.json'}|{
        f'{TOP}/outputs/{name}' for name in outputs}
    directories={str(parent) for name in expected for parent in Path(name).parents if str(parent)!='.'}
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();seen=set()
        for member in members:
            if member.name in seen:raise ValueError('Duplicate archive member')
            seen.add(member.name)
            if not ((member.isfile() and member.name in expected) or
                    (member.isdir() and member.name in directories)):
                raise ValueError('Unexpected archive member or link: '+member.name)
        if {m.name for m in members if m.isfile()}!=expected:
            raise ValueError('Incomplete temporal archive')
        if destination.exists():raise ValueError('Existing intake preserved')
        destination.mkdir(parents=True)
        for member in members:
            target=destination/member.name
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with stream.extractfile(member) as source,target.open('xb') as output:
                    while block:=source.read(4*1024*1024):output.write(block)
    return destination/TOP


def verify(archive,transfer_path,destination,data_root,report):
    receipt_path=ROOT/'experiments/upstream-remote-12h/tsvdx5-150o/native-receipt.json'
    if io.sha(receipt_path)!=io.RECEIPT_SHA:raise ValueError('Original own receipt changed')
    receipt=io.read(receipt_path);transfer=io.read(transfer_path)
    if (receipt['source_commit']!=io.PIN or receipt['job']!='tsvdx5-150o' or
            not receipt['finished'] or len(receipt['outputs'])!=63):
        raise ValueError('Complete original own native job required')
    if (transfer['state']!='complete_own_native_temporal_archive' or
            transfer['native_receipt_sha256']!=io.RECEIPT_SHA or transfer['original_outputs']!=63 or
            transfer['raw_datasets_included'] is not False or transfer['qualifying_labels_read'] is not False or
            archive.stat().st_size!=transfer['bytes'] or io.sha(archive)!=transfer['archive_sha256']):
        raise ValueError('Transfer binding differs')
    stage=extract_exact(archive,destination,receipt['outputs'])
    if (io.sha(stage/'native-receipt.json')!=io.RECEIPT_SHA or
            io.sha(stage/'manifest.json')!=transfer['manifest_sha256']):
        raise ValueError('Archive provenance differs')
    manifest=io.read(stage/'manifest.json')
    outputs={'outputs/'+name:digest for name,digest in receipt['outputs'].items()}
    if (manifest['state']!='complete_own_native_temporal_export' or manifest['source_pin']!=io.PIN or
            manifest['native_receipt_sha256']!=io.RECEIPT_SHA or manifest['outputs']!=outputs or
            manifest['qualifying_targets_zero'] is not True or manifest['qualifying_labels_read'] is not False or
            manifest['raw_datasets_included'] is not False):raise ValueError('Manifest differs')
    splits=['trainx','probex','fulltrain','qual'];rows={};inputs={}
    for split in splits:
        for name in ['user_idxs.npy','item_idxs.npy','dates.npy','ratings.npy']:
            path=data_root/split/name
            if f'{split}/{name}' not in manifest['input_sha256']:raise ValueError('Row identity missing')
        ratings=np.load(data_root/split/'ratings.npy',mmap_mode='r',allow_pickle=False)
        rows[split]=len(ratings)
        if split=='qual' and (len(ratings)!=2817131 or np.any(ratings)):
            raise ValueError('Qualifying targets must remain zero')
    for relative,digest in manifest['input_sha256'].items():
        path=Path(relative)
        if path.is_absolute() or len(path.parts)!=2 or path.parts[0] not in splits or path.suffix!='.npy':
            raise ValueError('Unexpected data identity path')
        if io.sha(data_root/path)!=digest:raise ValueError('Native input differs: '+relative)
        inputs[relative]=digest
    if rows!=manifest['rows']:raise ValueError('Population sizes differ')
    for relative,digest in outputs.items():
        path=stage/relative
        if io.sha(path)!=digest:raise ValueError('Native output differs: '+relative)
        if path.suffix=='.npy':io.vector(path,rows[path.name.rsplit('.',2)[-2]])
    # Rebind immutable inputs and archive after the full scan.
    for relative,digest in inputs.items():
        if io.sha(data_root/relative)!=digest:raise ValueError('Input changed during verification')
    if io.sha(archive)!=transfer['archive_sha256']:raise ValueError('Archive changed during verification')
    result=dict(state='complete',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        verifier_sha256=io.sha(Path(__file__)),archive_sha256=io.sha(archive),
        transfer_sha256=io.sha(transfer_path),manifest_sha256=io.sha(stage/'manifest.json'),
        native_receipt_sha256=io.RECEIPT_SHA,source_pin=io.PIN,rows=rows,
        input_sha256=inputs,outputs=outputs,verified_outputs=63,stage=str(stage.resolve()),
        qualifying_labels_read=False,qualifying_targets_zero=True,
        live_jobs_controlled=False,active_prediction_tree_modified=False,
        scope='Own complete remote counterpart verified. This does not declare any coordinator native job complete or admit outputs into an active tree.')
    io.write(report,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['archive','transfer','destination','data-root','report']:
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(verify(a.archive,a.transfer,a.destination,a.data_root,a.report))
