#!/usr/bin/env python3
"""Separate full-row own181 operator/statistic and actual ridge replay admission."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_own181_recent15_data as data

u=data.u

def main():
    report=data.REPORT;run=data.RUN;manifest=u.read(report/'manifest.json')
    if (report/'independent-verification.json').exists():raise ValueError('Existing complete operator proof retained')
    data.check(manifest['source_sha256'])
    for name,h in manifest['prepared_sha256'].items():
        if u.sha(run/name)!=h:raise ValueError('Actual prepared known data/statistics changed')
    for name in ['labels','users','groups']:
        if not np.array_equal(np.load(run/(name+'.npy')),np.load(data.original.RUN/(name+'.npy'))):raise ValueError('Original canonical data/user partitions changed')
    old_z=np.load(data.original.RUN/'design.npy',mmap_mode='r');old_f=np.load(data.original.RUN/'features.npy',mmap_mode='r')
    added=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in manifest['recent_rows']]
    y=np.load(run/'labels.npy');groups=np.load(run/'groups.npy');actual=data.Inputs(manifest['recent_rows'])
    grams=np.zeros((10,257,257));rhs=np.zeros((10,257));hashes=[hashlib.sha256(),hashlib.sha256()]
    def reference(lo,hi):
        z=np.empty((hi-lo,257),dtype=np.float64);f=np.empty((hi-lo,407),dtype=np.float32)
        z[:,:241]=old_z[lo:hi,:241];z[:,-1]=old_z[lo:hi,-1];f[:,:392]=old_f[lo:hi]
        for j,a in enumerate(added):
            clipped=np.clip(a[lo:hi],np.float32(0),np.float32(6))
            z[:,241+j]=clipped;f[:,392+j]=clipped-np.float32(3.6)
        return z,f
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));z,f=reference(lo,hi);az,af=actual.block(lo,hi)
        if not np.array_equal(z.view('u8'),az.view('u8')) or not np.array_equal(f.view('u4'),af.view('u4')):raise ValueError('Every original/new ridge/nonlinear input bit must independently match')
        hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
        for g in range(10):
            keep=groups[lo:hi]==g;a=z[keep];grams[g]+=a.T@a;rhs[g]+=a.T@y[lo:hi][keep]
    if [h.hexdigest() for h in hashes]!=manifest['virtual_c_order_sha256']:raise ValueError('Complete original C-order input hashes differ')
    with np.load(run/'stats.npz') as saved:
        if not np.array_equal(grams,saved['gram']) or not np.array_equal(rhs,saved['rhs']):raise ValueError('Every full original group statistic differs')
    regularizer=np.eye(257)*1000;regularizer[-1,-1]=0
    weights=np.stack([np.linalg.solve(grams[np.arange(10)//2!=fold].sum(0)+regularizer,rhs[np.arange(10)//2!=fold].sum(0)) for fold in range(5)])
    if not np.array_equal(weights,np.load(run/'ridge-weights.npy')):raise ValueError('Independent full eligible-user ridge solutions differ')
    p=np.empty(len(y),dtype='<f4')
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));z,_=reference(lo,hi)
        for fold in range(5):
            keep=groups[lo:hi]//2==fold;p[lo:hi][keep]=np.clip(z[keep]@weights[fold],1.05,4.95)
    stored=np.fromfile(run/'ridge.f32',dtype='<f4')
    if not np.array_equal(p.view('u4'),stored.view('u4')):raise ValueError('Every actual stored canonical ridge prediction differs')
    canonical=np.clip(stored.astype(np.float64),1.05,4.95)
    rmse=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(canonical,y))/len(y))
    if abs(rmse-manifest['ridge_rmse'])>1e-14:raise ValueError('Independent complete canonical score differs')
    control=np.load(data.original.RUN/'baseline.npy');control_rmse=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(control.astype(np.float64),1.05,4.95),y))/len(y))
    data.check(manifest['source_sha256'])
    u.write(report/'independent-verification.json',dict(state='complete_all_own181_retained_added_operator_bits_original_statistics_and_ridge_replay',
        manifest_sha256=u.sha(report/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),rows=len(y),raw_predictions=181,context_columns=90,
        design_columns=257,nonlinear_columns=407,all_original_and_new_input_values_bitwise_equal=True,
        all_original_user_partitions_retained=True,all_group_statistics_bitwise_equal=True,all_actual_ridge_prediction_bits_equal=True,
        ridge_rmse=rmse,preceding_original_ridge_rmse=control_rmse,preceding_original_ridge_dtype=str(control.dtype),ridge_improves_original=rmse<control_rmse,
        actual_ridge_prediction_sha256=u.sha(run/'ridge.f32'),qualifying_labels_read=False,neural_models_fitted=0,
        scope='Full canonical own181 input/ridge admission only. Complete neural fitting, original eligible408-input scaler proof and saved-model replay still mandatory; no development-best promotion or qualifying score.'))
    print('Independent own181 ridge',rmse,'original166 ridge',control_rmse,'all input/statistic/vector bits match; no NN promotion',flush=True)

if __name__=='__main__':main()
