#!/usr/bin/env python3
"""Return all63 outputs of our already completed original remote temporal job."""
import argparse
import datetime as dt
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np
import upstream_remote9h as io

ROOT=Path(__file__).resolve().parents[1]


def checked_outputs(tree,receipt):
    if len(receipt['outputs'])!=63:raise ValueError('Complete63-output original receipt required')
    result={}
    for relative,digest in receipt['outputs'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='preds_new' or len(p.parts)!=2:
            raise ValueError('Output outside original native prediction directory')
        if not p.name.startswith('tsvdx5-150o') or p.suffix not in ['.npy','.out']:
            raise ValueError('Unexpected native temporal output')
        source=tree/p
        if source.is_symlink() or io.sha(source)!=digest:raise ValueError('Original native output changed: '+relative)
        result[relative]=source
    return result


def export(tree,output):
    receipt_path=ROOT/'experiments/upstream-remote-12h/tsvdx5-150o/native-receipt.json'
    receipt=io.read(receipt_path)
    if (io.sha(receipt_path)!=io.RECEIPT_SHA or receipt['source_commit']!=io.PIN or
            receipt['job']!='tsvdx5-150o' or not receipt.get('finished')):
        raise ValueError('Our exact completed original remote job required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()!=io.PIN:
        raise ValueError('Original native source pin differs')
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=tree,text=True).splitlines()
    if set(changed)-{'data/qual_ratings/qual_ratings.csv.gz'}:raise ValueError('Tracked original source changed')
    if output.exists():raise ValueError('Existing complete-job export preserved')
    members=checked_outputs(tree,receipt);sizes={}
    for split in ['trainx','probex','fulltrain','qual']:
        ratings=np.load(tree/'data'/split/'ratings.npy',mmap_mode='r',allow_pickle=False);sizes[split]=len(ratings)
        if split=='qual' and (len(ratings)!=2817131 or np.any(ratings)):
            raise ValueError('Native qualifying targets must remain zero')
    for relative,source in members.items():
        if source.suffix=='.npy':
            split=source.name.rsplit('.',2)[-2]
            if split not in sizes:raise ValueError('Unknown native prediction row population')
            io.vector(source,sizes[split])
    output.mkdir(parents=True);stage=output/'temporal-native-complete';stage.mkdir()
    for relative,source in members.items():io.link(source,stage/'outputs'/relative)
    shutil.copyfile(receipt_path,stage/'native-receipt.json')
    manifest=dict(state='complete_own_native_temporal_export',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_pin=io.PIN,native_receipt_sha256=io.RECEIPT_SHA,rows=sizes,
        input_sha256={str(p.relative_to(tree/'data')):io.sha(p) for split in sizes for p in (tree/'data'/split).glob('*.npy')},
        outputs={str((Path('outputs')/relative)):digest for relative,digest in receipt['outputs'].items()},
        qualifying_targets_zero=True,qualifying_labels_read=False,raw_datasets_included=False,
        scope='All63 outputs, including both training residual vectors, from our own completed original25-epoch two-phase native job. No current coordinator job is claimed complete.')
    io.write(stage/'manifest.json',manifest);archive=output/'temporal-native-complete.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as stream:stream.add(stage,arcname=stage.name)
    checked_outputs(tree,receipt)
    result=dict(state='complete_own_native_temporal_archive',archive=archive.name,bytes=archive.stat().st_size,
        archive_sha256=io.sha(archive),manifest_sha256=io.sha(stage/'manifest.json'),native_receipt_sha256=io.RECEIPT_SHA,
        original_outputs=63,raw_datasets_included=False,qualifying_labels_read=False)
    io.write(output/'transfer.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native-tree',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();export(args.native_tree.resolve(),args.output.resolve())
