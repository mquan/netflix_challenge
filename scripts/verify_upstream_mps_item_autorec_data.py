#!/usr/bin/env python3
"""Independent lexicographic sort and every-row rating/query CSR validation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import time
import numpy as np
import upstream_mps_item_autorec_data as data

u=data.u

def verify_phase(phase):
    rp=data.REPORT/(phase+'-receipt.json');r=u.read(rp);out=data.RUN/phase;query,rows,qrows=data.PHASES[phase]
    if r['state']!='prepared_lossless_csr_pending_independent_verification' or r['script_sha256']!=u.sha(Path(data.__file__)):
        raise ValueError('Original preparation receipt differs')
    for rel,digest in r['source_sha256'].items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original rating/query source changed')
    for name,digest in r['outputs'].items():
        if u.sha(out/name)!=digest:raise ValueError('CSR output changed')
    source=u.FRESH/'data'/phase;qs=u.FRESH/'data'/query
    users=np.load(source/'item_idxs.npy',mmap_mode='r');items=np.load(source/'user_idxs.npy',mmap_mode='r');ratings=np.load(source/'ratings.npy',mmap_mode='r')
    ci=np.load(out/'columns.npy',mmap_mode='r');cr=np.load(out/'ratings.npy',mmap_mode='r');off=np.load(out/'offsets.npy')
    if ci.dtype!=np.uint32 or cr.dtype!=np.uint8 or ci.shape!=(rows,) or cr.shape!=(rows,) or off.shape!=(data.USERS+1,) or off[0]!=0 or off[-1]!=rows:
        raise ValueError('CSR schema differs')
    counts=np.bincount(users,minlength=data.USERS)
    if not np.array_equal(np.diff(off.astype(np.int64)),counts):raise ValueError('Original per-item counts differ')
    order=np.lexsort((items,users));keys=np.empty(rows,dtype=np.uint64)
    for lo in range(0,rows,500000):
        hi=min(lo+500000,rows);idx=order[lo:hi]
        if not np.array_equal(ci[lo:hi],items[idx]) or not np.array_equal(cr[lo:hi],ratings[idx]):raise ValueError('Independent every-row CSR replay differs')
        keys[lo:hi]=users[idx].astype(np.uint64)*data.ITEMS+ci[lo:hi].astype(np.uint64)
    del order
    for lo in range(1,rows,500000):
        hi=min(lo+500000,rows)
        if np.any(keys[lo:hi]<=keys[lo-1:hi-1]):raise ValueError('Duplicate user/item pair')
    qu=np.load(qs/'item_idxs.npy',mmap_mode='r');qi=np.load(qs/'user_idxs.npy',mmap_mode='r')
    for lo in range(0,qrows,500000):
        hi=min(lo+500000,qrows);q=qu[lo:hi].astype(np.uint64)*data.ITEMS+qi[lo:hi].astype(np.uint64);idx=np.searchsorted(keys,q);inside=idx<rows
        if np.any(keys[idx[inside]]==q[inside]):raise ValueError('Held query pair leaked into CSR history')
    qo=np.load(out/'query-order.npy');qitems=np.load(out/'query-columns.npy');qoff=np.load(out/'query-offsets.npy')
    if qoff.shape!=(data.USERS+1,) or qoff[0]!=0 or qoff[-1]!=qrows or np.any(np.diff(qoff.astype(np.int64))<0):
        raise ValueError('Invalid query offsets')
    if (qo.shape!=(qrows,) or not np.array_equal(np.sort(qo),np.arange(qrows,dtype=np.uint32)) or
        not np.array_equal(qitems,qi[qo]) or not np.array_equal(qu[qo],np.repeat(np.arange(data.USERS),np.diff(qoff.astype(np.int64))))):
        raise ValueError('Full query projection differs')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    result=dict(state='complete_independent_lossless_csr_and_query_verification',phase=phase,query=query,rows=rows,query_rows=qrows,
        receipt_sha256=u.sha(rp),verifier_sha256=u.sha(Path(__file__)),every_training_tuple_replayed=True,
        independently_lexicographically_sorted=True,query_pairs_absent_from_training=True,full_query_projection_checked=True,
        qualifying_targets_zero=True,qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Independent complete original item-row/user-column CSR and query-map replay. No model fitting, inference replay or score claimed.')
    u.write(data.REPORT/(phase+'-verification.json'),result);print(result,flush=True)

if __name__=='__main__':
    for phase in data.PHASES:
        while not (data.REPORT/(phase+'-receipt.json')).exists():time.sleep(20)
        verify_phase(phase)
