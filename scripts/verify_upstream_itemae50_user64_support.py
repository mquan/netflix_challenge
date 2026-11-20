#!/usr/bin/env python3
"""Independent covariance/eigenpair/projection and every query row checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import upstream_itemae50_user64_support as source

u=source.u

def verify():
    receipt=u.read(source.REPORT/'receipt.json')
    if receipt['state']!='complete_own_unsupervised_itemae50_user_context':raise ValueError('Complete producer required')
    bindings=dict(receipt['source_sha256'])
    for name,digest in receipt['output_sha256'].items():bindings[str((source.RUN/name).relative_to(u.ROOT))]=digest
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source/output differs')
    arrays,queries,_=source.inputs();table=np.load(source.RUN/'user64.npy');max_residual=0.
    if table.shape!=(480189,64) or table.dtype!=np.float32 or not np.isfinite(table).all():raise ValueError('User table differs')
    for j,(name,a) in enumerate(zip(source.MODELS,arrays)):
        with np.load(source.RUN/(name+'-projection.npz')) as p:
            center,vectors,scales,values=[p[k] for k in ['center','vectors','scales','eigenvalues']]
        independent_center=np.sum(a,axis=0,dtype=np.float64)/len(a)
        covariance=np.cov(a,rowvar=False,bias=True)
        spectrum=np.linalg.eigvalsh(covariance)[::-1][:source.RANK]
        if not np.allclose(center,independent_center,rtol=0,atol=1e-13):raise ValueError('Center differs')
        if not np.allclose(values,spectrum,rtol=1e-10,atol=1e-12):raise ValueError('Top eigenvalues differ')
        if not np.allclose(vectors.T@vectors,np.eye(source.RANK),rtol=0,atol=1e-11):raise ValueError('Basis is not orthonormal')
        residual=float(np.max(np.abs(covariance@vectors-vectors*values)))
        max_residual=max(max_residual,residual)
        if residual>1e-9*max(1.,float(values.max())):raise ValueError('Eigenpair residual too large')
        if not np.allclose(scales,np.sqrt(values),rtol=0,atol=0):raise ValueError('Projection scales differ')
        pivots=vectors[np.abs(vectors).argmax(axis=0),np.arange(source.RANK)]
        if not (pivots>0).all():raise ValueError('Canonical signs differ')
        replay=((a-independent_center)@vectors/scales).astype(np.float32)
        if not np.array_equal(replay.view(np.uint32),table[:,j*32:(j+1)*32].view(np.uint32)):
            raise ValueError('Every original factor projection must replay bitwise')
    for query,ids in queries.items():
        a=np.load(source.RUN/(query+'.npy'),mmap_mode='r')
        if a.shape!=(len(ids),64) or a.dtype!=np.float32:raise ValueError('Query support shape differs')
        for lo in range(0,len(ids),8192):
            if not np.array_equal(a[lo:lo+8192].view(np.uint32),table[ids[lo:lo+8192]].view(np.uint32)):
                raise ValueError('Every gathered query support must replay bitwise')
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Evidence changed during verification')
    u.write(source.REPORT/'verification.json',dict(state='complete_independent_itemae50_user_projection_and_all_query_rows',
        receipt_sha256=u.sha(source.REPORT/'receipt.json'),verifier_sha256=u.sha(Path(__file__)),bindings=bindings,
        rows={'probex':1128164,'qual':2817131},support_columns=64,max_eigenpair_residual=max_residual,
        independent_top_spectrum_checked=True,all_projected_users_and_query_rows_replayed_bitwise=True,
        qualifying_labels_read=False,public_numerical_features_read=False,scope='Independent unsupervised projection and full gather check. No nonlinear fitting, ensemble accuracy, qualifying score or public schema parity claim.'))

if __name__=='__main__':verify()
