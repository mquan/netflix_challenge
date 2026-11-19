#!/usr/bin/env python3
"""Build lossless item-row/user-column CSR histories, excluding each phase's query pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import fcntl
import shutil
import time
import numpy as np
import upstream_repro as u
import upstream_latent90_local as resources

RUN=u.ROOT/'artifacts/upstream-mps-item-autorec-data'
REPORT=u.ROOT/'experiments/upstream-mps-item-autorec-data'
# Compatibility dimension aliases for the shared CSR algorithm: USERS counts
# item rows; ITEMS counts user columns. Source filenames below are swapped.
USERS,ITEMS=17770,480189
PHASES={'trainx':('probex',99352343,1128164),'fulltrain':('qual',100480507,2817131)}
BLOCK=1_000_000

def build_phase(phase):
    query,rows,qrows=PHASES[phase];out=RUN/phase
    if out.exists():raise ValueError('Existing partial CSR phase preserved')
    source=u.FRESH/'data'/phase;qs=u.FRESH/'data'/query;sources={}
    arrays={name:np.load(source/(name+'.npy'),mmap_mode='r') for name in ['item_idxs','user_idxs','ratings','item_cnts']}
    users,items,ratings,counts=[arrays[k] for k in ['item_idxs','user_idxs','ratings','item_cnts']]
    if users.shape!=(rows,) or items.shape!=(rows,) or ratings.shape!=(rows,) or counts.shape!=(USERS,):raise ValueError('Original data shapes differ')
    if np.any(counts<0) or int(counts.astype(np.int64).sum())!=rows:raise ValueError('Original item counts differ')
    for name in arrays:sources[str((source/(name+'.npy')).relative_to(u.ROOT))]=u.sha(source/(name+'.npy'))
    qu=np.load(qs/'item_idxs.npy',mmap_mode='r');qi=np.load(qs/'user_idxs.npy',mmap_mode='r')
    if qu.shape!=(qrows,) or qi.shape!=(qrows,):raise ValueError('Original query shape differs')
    for name in ['user_idxs','item_idxs']:sources[str((qs/(name+'.npy')).relative_to(u.ROOT))]=u.sha(qs/(name+'.npy'))
    zero=u.FRESH/'data/qual/ratings.npy'
    if np.any(np.load(zero,mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    sources[str(zero.relative_to(u.ROOT))]=u.sha(zero)
    keys=np.empty(rows,dtype=np.uint64)
    for lo in range(0,rows,BLOCK):
        hi=min(lo+BLOCK,rows)
        if np.any((users[lo:hi]<0)|(users[lo:hi]>=USERS)) or np.any((items[lo:hi]<0)|(items[lo:hi]>=ITEMS)) or np.any((ratings[lo:hi]<1)|(ratings[lo:hi]>5)):
            raise ValueError('Invalid original rating tuple')
        keys[lo:hi]=users[lo:hi].astype(np.uint64)*ITEMS+items[lo:hi].astype(np.uint64)
    order=np.argsort(keys,kind='stable');sorted_keys=keys[order];del keys
    for lo in range(1,rows,BLOCK):
        hi=min(lo+BLOCK,rows)
        if np.any(sorted_keys[lo:hi]<=sorted_keys[lo-1:hi-1]):raise ValueError('Duplicate or unsorted source user/item pair')
    for lo in range(0,qrows,BLOCK):
        hi=min(lo+BLOCK,qrows)
        if np.any((qu[lo:hi]<0)|(qu[lo:hi]>=USERS)) or np.any((qi[lo:hi]<0)|(qi[lo:hi]>=ITEMS)):raise ValueError('Invalid query identity')
        qkey=qu[lo:hi].astype(np.uint64)*ITEMS+qi[lo:hi].astype(np.uint64);idx=np.searchsorted(sorted_keys,qkey)
        inside=idx<rows
        if np.any(sorted_keys[idx[inside]]==qkey[inside]):raise ValueError('Query pair appears in training history')
    out.mkdir(parents=True)
    offsets=np.empty(USERS+1,dtype=np.uint32);offsets[0]=0;offsets[1:]=np.cumsum(counts,dtype=np.uint64)
    np.save(out/'offsets.npy',offsets)
    ci=np.lib.format.open_memmap(out/'columns.npy',mode='w+',dtype=np.uint32,shape=(rows,))
    cr=np.lib.format.open_memmap(out/'ratings.npy',mode='w+',dtype=np.uint8,shape=(rows,))
    observed_counts=np.zeros(USERS,dtype=np.int64)
    for lo in range(0,rows,BLOCK):
        hi=min(lo+BLOCK,rows);index=order[lo:hi];sk=sorted_keys[lo:hi]
        ci[lo:hi]=items[index];cr[lo:hi]=ratings[index]
        if not np.array_equal(ci[lo:hi],sk%ITEMS) or not np.array_equal(cr[lo:hi].astype(np.int8),ratings[index]):
            raise ValueError('Lossless CSR reconstruction differs')
        observed_counts+=np.bincount((sk//ITEMS).astype(np.int64),minlength=USERS)
    if not np.array_equal(observed_counts,counts):raise ValueError('CSR offsets disagree with every original item count')
    ci.flush();cr.flush();del ci,cr,order,sorted_keys
    qo=np.argsort(qu,kind='stable').astype(np.uint32);qc=np.bincount(qu,minlength=USERS)
    qoff=np.empty(USERS+1,dtype=np.uint32);qoff[0]=0;qoff[1:]=np.cumsum(qc,dtype=np.uint64)
    np.save(out/'query-order.npy',qo);np.save(out/'query-columns.npy',qi[qo].astype(np.uint32));np.save(out/'query-offsets.npy',qoff)
    for rel,digest in sources.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original source changed during CSR preparation')
    receipt=dict(state='prepared_lossless_csr_pending_independent_verification',phase=phase,query=query,ratings=rows,query_rows=qrows,
        row_items=USERS,column_users=ITEMS,axis_order=['item','user'],source_sha256=sources,outputs={p.name:u.sha(p) for p in out.iterdir()},
        all_original_rating_tuples_checked=True,query_pairs_absent_from_training=True,qualifying_targets_zero=True,
        qualifying_labels_read=False,public_prediction_values_read=False,source_pin=u.PIN,script_sha256=u.sha(Path(__file__)),
        scope='Lossless item-oriented original rating histories only. Column IDs are uint32 canonical user indices; no16-bit truncation. Query labels are excluded from this phase history. No model fitting or accuracy result.')
    u.write(REPORT/(phase+'-receipt.json'),receipt);print(phase,'lossless CSR prepared',flush=True)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            previous=resources.memory.memory();good=0
            while good<2:
                current=resources.memory.memory();workers=resources.footprints();free=shutil.disk_usage(u.ROOT).free
                good=good+1 if (sum(v['bytes'] for v in workers.values())<16*2**30 and current['compressor_bytes']<5*2**30 and
                    current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>12*2**30) else 0
                u.write(REPORT/'status.json',dict(state='waiting_for_csr_preparation_headroom',observed=resources.stamp(),workers=workers,memory=current,disk_free_bytes=free,good_samples=good))
                if good<2:previous=current;time.sleep(30)
            for phase in PHASES:
                if (REPORT/(phase+'-receipt.json')).exists():raise ValueError('Existing CSR receipt preserved; use explicit recovery after inspection')
                u.write(REPORT/'status.json',dict(state='preparing_lossless_rating_histories',phase=phase,observed=resources.stamp()))
                build_phase(phase)
            u.write(REPORT/'status.json',dict(state='both_lossless_histories_prepared_independent_verification_required',observed=resources.stamp()))

if __name__=='__main__':run()
