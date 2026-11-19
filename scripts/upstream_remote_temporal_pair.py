#!/usr/bin/env python3
"""Independently reconstruct our original remote temporal NPY pairs from exports."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import upstream_temporal_phase as phase

u=phase.u
JOB,NAMES=phase.JOB,phase.NAMES
RUN=u.ROOT/'artifacts/upstream-remote-temporal-pair'
REPORT=u.ROOT/'experiments/upstream-remote-temporal-pair'
PREFIX='experiments/upstream-remote9h/temporal-qual/'
RECEIPT_SHA='12421a92ea6495ece7347ecf8e7d794c52d52c8e1f14e7fbc5914e455708dac7'


def native_header(path,rows):
    with Path(path).open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,order,dtype=reader(stream)
        if shape!=(rows,) or order or dtype!=np.dtype('<f4'):raise ValueError('Native NPY header layout differs')
        end=stream.tell();stream.seek(0);return stream.read(end)


def native_digest(values,header):
    if values.ndim!=1 or values.dtype!=np.dtype('float32') or not np.isfinite(values).all():
        raise ValueError('Malformed native prediction values')
    digest=hashlib.sha256(header);digest.update(np.asarray(values,dtype='<f4').tobytes());return digest.hexdigest()


def reconstruct(values,header,digest,path):
    if native_digest(values,header)!=digest:raise ValueError('Transferred values do not reproduce original native NPY hash')
    with path.open('wb') as stream:stream.write(header);stream.write(np.asarray(values,dtype='<f4').tobytes())
    if u.sha(path)!=digest:raise ValueError('Reconstructed native file changed')


def ingest(ref):
    if (REPORT/'paired-receipt.json').exists():return verify_receipt()
    if RUN.exists():raise ValueError('Partial remote temporal intake preserved')
    commit=subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=u.ROOT,text=True).strip()
    def get(name):return subprocess.check_output(['git','show',commit+':'+PREFIX+name],cwd=u.ROOT)
    # Fetch every object by one immutable commit before creating any local cache.
    transfer_bytes=get('manifest.json');native_bytes=get('native-receipt.json')
    transfer=json.loads(transfer_bytes);receipt=json.loads(native_bytes)
    if (hashlib.sha256(native_bytes).hexdigest()!=RECEIPT_SHA or transfer['native_receipt_sha256']!=RECEIPT_SHA or
            receipt['source_commit']!=u.PIN or transfer['source_pin']!=u.PIN or receipt['job']!=JOB or
            not receipt.get('finished') or len(receipt['outputs'])!=63 or not transfer['qualifying_targets_all_zero'] or
            transfer['qualifying_labels_read'] is not False or not transfer['all63_native_outputs_rehashed']):
        raise ValueError('Our complete original remote native job/zero-target export required')
    if receipt['command']!=u.read(u.BASE/'registry.json')['jobs'][JOB]['command']:
        raise ValueError('Remote original model recipe differs')
    for name,digest in transfer['qual_identity_sha256'].items():
        if Path(name).name!=name or u.sha(u.FRESH/'data/qual'/name)!=digest:
            raise ValueError('Remote qualifying row identity differs')
    expected_names={'user_idxs.npy','item_idxs.npy','dates.npy','ratings.npy'}
    if not expected_names.issubset(transfer['qual_identity_sha256']):raise ValueError('Qualifying row identity coverage incomplete')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Native qualifying targets are not zero')
    rows=transfer['records']
    if len(rows)!=4 or {r['name'] for r in rows}!=set(NAMES):raise ValueError('Exactly four original temporal columns required')
    original=u.ROOT/'experiments/upstream-remote-12h'/JOB/'native-receipt.json'
    if u.sha(original)!=RECEIPT_SHA:raise ValueError('Recorded original native receipt changed')
    qual_header_path=phase.mapped.TREE/phase.PRED/(JOB+'_ep20.qual.npy')
    if u.sha(qual_header_path)!=receipt['outputs']['preds_new/'+JOB+'_ep20.qual.npy']:
        raise ValueError('Independently reproduced native header reference changed')
    headers=dict(qual=native_header(qual_header_path,2817131),
                 probex=native_header(u.FRESH/'preds_new/cfnade-12.probex.npy',1128164))
    probex,sources=phase.remote_columns();values={};raw={}
    for record in rows:
        name=record['name'];filename=name+'.qual.npz'
        if record['npz']!=filename or record['rows']!=2817131:raise ValueError('Unexpected remote vector path/shape')
        blob=get(filename);raw[filename]=blob
        if hashlib.sha256(blob).hexdigest()!=record['npz_sha256']:raise ValueError('Transfer archive changed')
        with np.load(io.BytesIO(blob),allow_pickle=False) as saved:
            if saved.files!=['prediction']:raise ValueError('Unexpected transfer array fields')
            value=saved['prediction']
        if value.shape!=(2817131,):raise ValueError('Qualifying prediction shape differs')
        expected=receipt['outputs']['preds_new/'+name+'.qual.npy']
        if record['source_npy_sha256']!=expected or native_digest(value,headers['qual'])!=expected:
            raise ValueError('Qualifying contents differ from our original native receipt')
        if native_digest(probex[name][0],headers['probex'])!=receipt['outputs']['preds_new/'+name+'.probex.npy']:
            raise ValueError('ProbeX contents differ from our original native receipt')
        values[name]=value
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    (RUN/'transfer-manifest.json').write_bytes(transfer_bytes);(RUN/'native-receipt.json').write_bytes(native_bytes)
    pairs={};outputs={}
    for name in NAMES:
        pairs[name]={}
        for split,value in [('probex',probex[name][0]),('qual',values[name])]:
            path=RUN/(name+'.'+split+'.npy');digest=receipt['outputs']['preds_new/'+path.name]
            reconstruct(value,headers[split],digest,path)
            relative=str(path.relative_to(u.ROOT));pairs[name][split]=relative;outputs[relative]=digest
    for path in [original,qual_header_path,Path(__file__),RUN/'transfer-manifest.json',RUN/'native-receipt.json']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    sources.update({str((u.FRESH/'data/qual'/name).relative_to(u.ROOT)):digest for name,digest in transfer['qual_identity_sha256'].items()})
    result=dict(state='complete',job=JOB,source_commit=u.PIN,original_native_receipt_sha256=RECEIPT_SHA,
        transfer_commit=commit,script_sha256=u.sha(Path(__file__)),pairs=pairs,outputs=outputs,source_sha256=sources,
        epochs_per_phase=25,both_original_phases_complete=True,qualifying_outcomes_zero=True,qualifying_labels_read=False,
        all_eight_reconstructed_npy_hashes_match_original_native_receipt=True,
        scope='Own completed original remote two-phase source job. Four pairs independently reconstructed to their original native NPY SHA256, using independently reproduced native headers. No current local phase output directory or active native prediction tree modified.')
    temporary=REPORT/'paired-receipt.pending.json';u.write(temporary,result);os.replace(temporary,REPORT/'paired-receipt.json')
    return result


def verify_receipt(verify_files=True):
    result=u.read(REPORT/'paired-receipt.json')
    if (result['state']!='complete' or result['source_commit']!=u.PIN or result['original_native_receipt_sha256']!=RECEIPT_SHA or
            result['script_sha256']!=u.sha(Path(__file__)) or not result['all_eight_reconstructed_npy_hashes_match_original_native_receipt']):
        raise ValueError('Remote paired receipt binding changed')
    if verify_files:
        for relative,digest in {**result['source_sha256'],**result['outputs']}.items():
            if u.sha(u.ROOT/relative)!=digest:raise ValueError('Remote paired source/output changed: '+relative)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--import-ref');args=parser.parse_args()
    print(ingest(args.import_ref) if args.import_ref else verify_receipt())
