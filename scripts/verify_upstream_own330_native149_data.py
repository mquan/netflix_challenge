#!/usr/bin/env python3
"""Replay every own330 ridge input, eligible statistic, solve and stored value."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_own330_native149_data as data

u=data.u


def main():
    report,run=data.REPORT,data.RUN
    if (report/'independent-verification.json').exists():raise ValueError('Existing full own330 replay preserved')
    m=u.read(report/'manifest.json');data.check(m['source_sha256'])
    for name,digest in m['prepared_sha256'].items():
        if u.sha(run/name)!=digest:raise ValueError('Prepared complete ridge output changed')
    for name in ['labels','users','groups']:
        actual=np.load(run/(name+'.npy'))
        original=np.load(u.ROOT/'artifacts/upstream-own181-recent15-data'/(name+'.npy'))
        if not np.array_equal(actual,original):raise ValueError('Original canonical user/data partitions changed')
    labels=np.load(run/'labels.npy');groups=np.load(run/'groups.npy')
    if len(labels)!=1128164 or m['design_columns']!=406 or len(m['added_rows'])!=149 or len(m['control_rows'])!=15:
        raise ValueError('Full original330 input schema required')
    fixed=u.read(data.INVENTORY)
    if m['added_rows']!=fixed['rows'] or m['control_rows']!=u.read(data.CONTROL/'manifest.json')['recent_rows']:
        raise ValueError('The exact fixed source/order of all original149 additions must match')
    # The data producer's bounded reader is checked against a separate
    # sequential payload reader: no shared block/seek arithmetic or operators.
    paths=[u.ROOT/'artifacts/upstream-warm166/design.npy']+[u.ROOT/r['paired_files']['probex'] for r in m['control_rows']+m['added_rows']]
    streams=[]
    for path in paths:
        stream=path.open('rb');version=np.lib.format.read_magic(stream)
        if version!=(1,0):raise ValueError('Original NPY version differs')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        if fortran:raise ValueError('Original C-order payload required')
        streams.append((stream,shape,dtype,stream.tell()))
    actual=data.Inputs(m['control_rows'],m['added_rows'])
    def rewind():
        for stream,_,_,offset in streams:stream.seek(offset)
    def reference(count):
        old=np.fromfile(streams[0][0],dtype=streams[0][2],count=count*242).reshape(count,242)
        z=np.empty((count,406),dtype=np.float64);control=np.empty((count,257),dtype=np.float64)
        z[:,:241]=old[:,:241];z[:,-1]=old[:,-1]
        control[:,:241]=old[:,:241];control[:,-1]=old[:,-1]
        added=[]
        for j,(stream,_,dtype,_) in enumerate(streams[1:]):
            a=np.fromfile(stream,dtype=dtype,count=count)
            if len(a)!=count:raise ValueError('Short independent full input payload')
            a=np.maximum(np.minimum(a,np.float32(6)),np.float32(0))
            z[:,241+j]=a
            if j<15:control[:,241+j]=a
            else:added.append(a)
        return z,control,added
    grams=np.zeros((10,406,406));rhs=np.zeros((10,406));hashes=[hashlib.sha256(),hashlib.sha256()]
    raw_hashes=[hashlib.sha256() for _ in range(149)]
    try:
        for lo in range(0,len(labels),8192):
            hi=min(lo+8192,len(labels));z,control,added=reference(hi-lo);az,ac,aa=actual.block(lo,hi)
            if not np.array_equal(z.view('u8'),az.view('u8')) or not np.array_equal(control.view('u8'),ac.view('u8')):
                raise ValueError('All original/new bounded row-reader input bits must match')
            for j,a in enumerate(added):
                if not np.array_equal(a.view('u4'),aa[:,j].view('u4')):raise ValueError('Every added float32 bit differs')
                raw_hashes[j].update(a.tobytes(order='C'))
            hashes[0].update(z.tobytes(order='C'));hashes[1].update(control.tobytes(order='C'))
            for group in range(10):
                keep=groups[lo:hi]==group;x=z[keep]
                grams[group]+=x.T@x;rhs[group]+=x.T@labels[lo:hi][keep]
        if hashes[0].hexdigest()!=m['full_design_c_order_sha256'] or hashes[1].hexdigest()!=m['original181_control_c_order_sha256']:
            raise ValueError('Complete original input payload hashes differ')
        if hashes[1].hexdigest()!=u.read(data.CONTROL/'manifest.json')['virtual_c_order_sha256'][0]:
            raise ValueError('Every retained control bit must match original181')
        if [h.hexdigest() for h in raw_hashes]!=m['complete_added_payload_sha256']:
            raise ValueError('All149 added source payloads differ')
        if [h.hexdigest() for h in raw_hashes]!=[r['canonical_clipped_payload_sha256'] for r in fixed['rows']]:
            raise ValueError('Every full149 payload must match the original fixed inventory')
        saved=np.load(run/'stats.npz')
        if not np.array_equal(grams,saved['gram']) or not np.array_equal(rhs,saved['rhs']):
            raise ValueError('All eligible group sufficient statistics must match')
        penalty=np.eye(406)*1000;penalty[-1,-1]=0
        weights=np.stack([np.linalg.solve(grams[np.arange(10)//2!=fold].sum(0)+penalty,
                             rhs[np.arange(10)//2!=fold].sum(0)) for fold in range(5)])
        if not np.array_equal(weights,np.load(run/'weights.npy')):raise ValueError('All independent eligible-user solves differ')
        prediction=np.empty(len(labels),dtype='<f4');rewind()
        for lo in range(0,len(labels),8192):
            hi=min(lo+8192,len(labels));z,_,_=reference(hi-lo)
            for fold in range(5):
                keep=groups[lo:hi]//2==fold
                prediction[lo:hi][keep]=np.clip(z[keep]@weights[fold],1.05,4.95)
    finally:
        actual.close()
        for stream,_,_,_ in streams:stream.close()
    stored=np.fromfile(run/'ridge.f32',dtype='<f4')
    if not np.array_equal(prediction.view('u4'),stored.view('u4')):raise ValueError('Every actual held ridge prediction must match')
    rmse=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(stored.astype(np.float64),1.05,4.95),labels))/len(labels))
    if abs(rmse-m['ridge_rmse'])>1e-14:raise ValueError('Complete canonical compensated score differs')
    data.check(m['source_sha256'])
    u.write(report/'independent-verification.json',dict(state='complete_all_original330_ridge_inputs_statistics_solves_and_prediction_bits',
            manifest_sha256=u.sha(report/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),
            rows=len(labels),raw_predictions=330,design_columns=406,
            all_original181_control_bits_preserved=True,all149_added_payloads_replayed=True,
            all_group_statistics_and_eligible_user_solves_match=True,all_actual_float32_predictions_match=True,
            ridge_rmse=rmse,original181_ridge_rmse=m['original181_ridge_rmse'],
            actual_prediction_sha256=u.sha(run/'ridge.f32'),qualifying_labels_read=False,neural_models_fitted=0,
            scope='Complete bounded transport and original full ridge replay. No neural feature/scaler/model admission, global-best promotion or qualifying performance claim.'))
    print('Independent complete own330 ridge:',rmse,'original181:',m['original181_ridge_rmse'],flush=True)


if __name__=='__main__':main()
