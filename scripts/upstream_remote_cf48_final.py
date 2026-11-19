#!/usr/bin/env python3
"""Verify our completed remote CF48 terminal pair from an immutable Git commit."""
import argparse
import datetime as dt
import hashlib
import io
from pathlib import Path
import subprocess
import json
import numpy as np
import upstream_cfnade48 as cf
from upstream_remote_temporal_pair import native_header,native_digest,reconstruct

u=cf.u
RUN=u.ROOT/'artifacts/upstream-remote-cf48-final'
REPORT=u.ROOT/'experiments/upstream-remote-cf48-final'
PREFIX='experiments/upstream-remote9h/cfnade48-final/'


def ingest(ref):
    if RUN.exists() or REPORT.exists():raise ValueError('Existing final CF48 intake preserved')
    commit=subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=u.ROOT,text=True).strip()
    def get(path):return subprocess.check_output(['git','show',commit+':'+path],cwd=u.ROOT)
    raw=get(PREFIX+'verification.json');verified=json.loads(raw)
    if verified['state']!='verified_complete_resumed_cfnade48' or verified['failures'] or set(verified['phases'])!={'trainx','fulltrain'}:
        raise ValueError('Complete two-phase sender verification required')
    plan=cf.verify_plan();package_path=u.ROOT/'experiments/upstream-remote9h/cfnade48-package.json';package=u.read(package_path)
    manifest_path=u.ROOT/'artifacts/remote9h-transfer/cfnade48-package/manifest.json';manifest=u.read(manifest_path)
    if (u.sha(manifest_path)!=package['manifest_sha256'] or manifest['source_id']!=plan['source_id'] or
            manifest['data_id']!=plan['data_id'] or manifest['original_plan_sha256']!=u.sha(cf.REPORT/'plan.json') or
            (plan['hidden'],plan['rank'],plan['epochs_per_phase'])!=(160,48,95)):
        raise ValueError('Our original warm trajectory/configuration differs')
    receipt=get(PREFIX+'RECEIPT.md');text=receipt.decode();binary=next(v for k,v in plan['binaries'].items() if k.endswith('/cfnade48-stage'))
    if binary not in text or package['archive_sha256'] not in text or plan['source_id'] not in text or plan['data_id'] not in text:
        raise ValueError('Final receipt must bind the original executable/package/source/data')
    migration=u.ROOT/'experiments/upstream-cfnade48/remote-migration-verification.json';prior=u.read(migration)
    restore=get('experiments/upstream-remote9h/run/RESUME_PROOF.md')
    if hashlib.sha256(restore).hexdigest()!=prior['remote_proof_sha256']:
        raise ValueError('Previously verified exact remote restore proof changed')
    records=[];arrays={};blobs={'verification.json':raw,'RECEIPT.md':receipt,'RESUME_PROOF.md':restore}
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(cf.__file__),
        u.ROOT/'scripts/upstream_remote_temporal_pair.py',cf.REPORT/'plan.json',package_path,manifest_path,migration]}
    for train,query,rows in [('trainx','probex',1128164),('fulltrain','qual',2817131)]:
        phase=verified['phases'][train];identity=phase['identity'];latest=phase['final_latest']
        if (phase['query']!=query or phase['rows']!=rows or latest['epoch']!=95 or
                phase['checkpoint_state_md5']!=latest['manifest_md5'] or phase['tensors_verified']!=84 or
                phase['swa_count']!=78 or phase['timestep']<=0 or
                identity['source']!=plan['source_id'] or identity['data']!=plan['data_id'] or
                identity['config']!=plan['description'] or identity['train']!=train or
                identity['query']!=query or identity['query_rows']!=rows):
            raise ValueError('Final original95-epoch phase identity/counts differ')
        snapshots=phase['snapshot_sha256'];expected={f'cfnade-48_ep{e:02}.{query}.npy' for e in range(1,96)}
        if set(snapshots)!=expected:raise ValueError('Sender snapshot inventory incomplete')
        for name,digest in manifest['phases'][train]['prediction_sha256'].items():
            if snapshots[name]!=digest:raise ValueError('Pre-resume snapshot trajectory differs')
        dataset=u.FRESH/'data'/query;actual_files={p.name:p for p in dataset.glob('*.npy')}
        if set(phase['row_identity_sha256'])!=set(actual_files):raise ValueError('Query row identity inventory differs')
        for name,path in actual_files.items():
            digest=u.sha(path)
            if digest!=phase['row_identity_sha256'][name]:raise ValueError('Query rows/labels differ')
            sources[str(path.relative_to(u.ROOT))]=digest
        filename=f'cfnade-48.{query}.npz'
        if phase['npz']!=filename:raise ValueError('Unexpected terminal export path')
        blob=get(PREFIX+filename)
        if hashlib.sha256(blob).hexdigest()!=phase['npz_sha256']:raise ValueError('Terminal export transport hash differs')
        with np.load(io.BytesIO(blob),allow_pickle=False) as saved:
            if saved.files!=['prediction']:raise ValueError('Unexpected exported fields')
            values=saved['prediction']
        if values.shape!=(rows,) or values.dtype!=np.dtype('float32') or not np.isfinite(values).all():
            raise ValueError('Malformed terminal prediction')
        if hashlib.sha256(values.tobytes()).hexdigest()!=phase['prediction_bytes_sha256']:
            raise ValueError('Lossless prediction content hash differs')
        reference=cf.OUT/f'cfnade-48_ep25.{query}.npy'
        if u.sha(reference)!=manifest['phases'][train]['prediction_sha256'][reference.name]:
            raise ValueError('Own native header reference differs')
        header=native_header(reference,rows);digest=native_digest(values,header)
        if digest!=phase['terminal_npy_sha256'] or digest!=snapshots[f'cfnade-48_ep95.{query}.npy']:
            raise ValueError('Terminal contents do not reconstruct native terminal/epoch95 SHA256')
        logname=f'cfnade-48.phase-{query}.out';log=get(PREFIX+logname)
        if hashlib.sha256(log).hexdigest()!=phase['other_outputs_sha256'][logname] or b'Epoch 95 | RMSE' not in log:
            raise ValueError('Actual final native log differs')
        blobs[filename]=blob;blobs[logname]=log;arrays[query]=(values,header,digest)
        records.append(dict(train=train,query=query,rows=rows,native_npy_sha256=digest,npz_sha256=phase['npz_sha256'],
            final_epoch=95,sender_verified_tensors=84,sender_verified_snapshots=95,matched_all_captured_warm_snapshot_hashes=True))
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets must remain zero')
    RUN.mkdir();REPORT.mkdir();pairs={};outputs={}
    for name,blob in blobs.items():
        path=RUN/name;path.write_bytes(blob);sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for query,(values,header,digest) in arrays.items():
        path=RUN/f'cfnade-48_ep95.{query}.npy';reconstruct(values,header,digest,path)
        pairs[query]=str(path.relative_to(u.ROOT));outputs[str(path.relative_to(u.ROOT))]=digest
    result=dict(state='verified_complete_resumed_remote_prediction_intake',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_pin=u.PIN,source_id=plan['source_id'],data_id=plan['data_id'],transfer_commit=commit,
        records=records,pairs={'cfnade-48_ep95':pairs},outputs=outputs,source_sha256=sources,
        full95_model_complete=True,qualifying_labels_read=False,qualifying_targets_zero=True,
        reconstructed_native_terminal_and_epoch95_sha256_match=True,
        coordinator_rehashed_final_checkpoint_tensors=False,coordinator_received_all95_snapshots=False,
        evidence_scope='Sender verified all final84 tensors and95 snapshots per phase; coordinator checks immutable-commit transport, own original package/source/data/config/restore lineage, all captured warm snapshot hashes, query identities, final logs, finite float32 terminal values and exact native terminal/epoch95 SHA reconstruction. No claim of independently re-running all training or rehashing unreturned final tensors.',
        native_dag_receipt_fabricated=False,scope='Our completed resumed160/48/95 trajectory. No public prediction input and no qualifying outcome fitting/selection. Qualifying log RMSE is against zero sentinels and is not accuracy.')
    u.write(REPORT/'paired-receipt.json',result);print(dict(state=result['state'],commit=commit,outputs=outputs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);ingest(p.parse_args().ref)
