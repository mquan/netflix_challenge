#!/usr/bin/env python3
"""Separate covariance, complete spectrum and every-item projection replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import upstream_rank1000_item32 as source
u=source.u

def verify():
    r=u.read(source.REPORT/'receipt.json')
    if r['state']!='complete_own_rank1000_item32_projection':raise ValueError('Complete original projection required')
    a,bindings=source.inputs();bindings.update(r['source_sha256'])
    for name,digest in r['output_sha256'].items():bindings[str((source.RUN/name).relative_to(u.ROOT))]=digest
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Projection evidence differs')
    with np.load(source.RUN/'projection.npz') as p:center,vectors,scales,values=[p[k] for k in ['center','vectors','scales','eigenvalues']]
    independent_center=np.sum(a,axis=0,dtype=np.float64)/len(a);cov=np.cov(a,rowvar=False,bias=True);top=np.linalg.eigvalsh(cov)[::-1][:32]
    if not np.allclose(center,independent_center,rtol=0,atol=1e-13) or not np.allclose(values,top,rtol=1e-10,atol=1e-12):raise ValueError('Independent center/top spectrum differs')
    residual=float(np.max(np.abs(cov@vectors-vectors*values)))
    if residual>1e-9*max(1.,float(values.max())) or not np.allclose(vectors.T@vectors,np.eye(32),rtol=0,atol=1e-11):raise ValueError('Independent eigenpairs differ')
    if not np.array_equal(scales,np.sqrt(values)) or not (vectors[np.abs(vectors).argmax(axis=0),np.arange(32)]>0).all():raise ValueError('Canonical signs/scales differ')
    table=np.load(source.RUN/'item32.npy');replay=((a-independent_center)@vectors/scales).astype(np.float32)
    if table.shape!=(17770,32) or table.dtype!=np.float32 or not np.isfinite(table).all() or not np.array_equal(table.view(np.uint32),replay.view(np.uint32)):raise ValueError('Every item must replay bitwise')
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Source evidence changed')
    u.write(source.REPORT/'verification.json',dict(state='complete_independent_rank1000_item32_projection',receipt_sha256=u.sha(source.REPORT/'receipt.json'),
        verifier_sha256=u.sha(Path(__file__)),bindings=bindings,independent_top_spectrum_checked=True,all_projected_items_replayed_bitwise=True,
        max_eigenpair_residual=residual,support_columns=32,qualifying_labels_read=False,public_numerical_features_read=False,accuracy_claimed=False))
    print('All17770 item projections and independent eigenspectrum verified',flush=True)
if __name__=='__main__':verify()
