#!/usr/bin/env python3
"""Independent complete column construction and original ridge-statistics replay."""
import hashlib
from pathlib import Path
import numpy as np
import upstream_own167_context218_features as data
import upstream_nonlinear as original

u=data.u

def verify():
    output=data.REPORT/'independent-verification.json'
    if output.exists():raise ValueError('Existing complete-source verification preserved')
    manifest=u.read(data.REPORT/'manifest.json');bindings,arrays=data.sources()
    if bindings!=manifest['source_sha256']:raise ValueError('Original source list differs')
    for name,digest in manifest['prepared_sha256'].items():
        if u.sha(data.RUN/name)!=digest:raise ValueError('Prepared identities/statistics changed')
    oldz,oldf,raw,item,user=arrays
    complete=np.empty((1128164,243),dtype=np.float64)
    fingerprints=[hashlib.sha256(),hashlib.sha256()]
    # Different block boundaries and direct column assignment from original sources.
    for lo in range(0,len(complete),10007):
        hi=min(lo+10007,len(complete));count=hi-lo
        z=complete[lo:hi];f=np.empty((count,521),dtype=np.float32)
        z[:,:241]=oldz[lo:hi,:241];z[:,241]=np.minimum(np.maximum(raw[lo:hi],0),6);z[:,242]=oldz[lo:hi,241]
        f[:,:302]=oldf[lo:hi,:302];f[:,302]=z[:,241].astype(np.float32)-np.float32(3.6)
        f[:,303:393]=oldf[lo:hi,302:392];f[:,393:457]=item[lo:hi];f[:,457:521]=user[lo:hi]
        if not np.isfinite(f).all() or not np.isfinite(z).all() or not np.all(z[:,-1]==1):raise ValueError('Complete finite feature/intercept domain differs')
        for digest,a in zip(fingerprints,[z,f]):digest.update(a.tobytes(order='C'))
    if [h.hexdigest() for h in fingerprints]!=manifest['virtual_c_order_sha256']:raise ValueError('Independent complete virtual tensors differ')
    del arrays,oldz,oldf,raw,item,user,z,f
    y=np.load(data.RUN/'labels.npy');groups=np.load(data.RUN/'groups.npy')
    grams,rhs=original.grouped_stats(complete,y,groups)
    with np.load(data.RUN/'stats.npz') as stored:
        if not np.array_equal(grams.view('u8'),stored['gram'].view('u8')) or not np.array_equal(rhs.view('u8'),stored['rhs'].view('u8')):
            raise ValueError('Original full ridge group statistic bits differ')
    for key,digest in bindings.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Source changed during complete verification')
    u.write(output,dict(state='complete_independent_all_columns_and_original_group_statistics',
        manifest_sha256=u.sha(data.REPORT/'manifest.json'),rows=len(complete),design_columns=243,nonlinear_features=521,
        virtual_c_order_sha256=[h.hexdigest() for h in fingerprints],different_block_boundaries_and_independent_column_assignment=True,
        original_reference_all_ridge_group_statistic_bits_equal=True,qualifying_labels_read=False,models_fitted=0,
        verifier_sha256=u.sha(Path(__file__)),reference_sha256=u.sha(Path(original.__file__)),
        scope='Full original inputs/statistics only; no model or score claim. Complete522-input normalization parity and actual nested fits still required.'))
    print('Every new feature bit and all original group statistics independently replayed',flush=True)

if __name__=='__main__':verify()
