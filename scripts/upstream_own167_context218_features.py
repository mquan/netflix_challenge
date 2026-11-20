#!/usr/bin/env python3
"""Stream exact own167/context218 tensors without persistent large matrices."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import datetime as dt
import hashlib
from pathlib import Path
import numpy as np
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-own167-context218-features'
REPORT=u.ROOT/'experiments/upstream-own167-context218-features'
PARENT=u.ROOT/'artifacts/upstream-warm166'
PARENT_REPORT=u.ROOT/'experiments/upstream-warm166'

def sources():
    bindings={}
    def bind(path,digest=None):
        key=str(path.relative_to(u.ROOT));actual=u.sha(path)
        if digest is not None and actual!=digest:raise ValueError('Original expected source differs: '+key)
        if key in bindings and bindings[key]!=actual:raise ValueError('Conflicting source binding')
        bindings[key]=actual
    mp=PARENT_REPORT/'manifest.json';m=u.read(mp)
    if m['prediction_columns']!=166 or m['design_columns']!=242 or m['nonlinear_features']!=392:raise ValueError('Exact own166 parent required')
    for collection in [m['code_sha256'],m['source_manifests_sha256']]:
        for key,digest in collection.items():bind(u.ROOT/key,digest)
    for key,digest in m['prepared_sha256'].items():bind(PARENT/key,digest)
    bind(mp)
    supports=[]
    for namespace,state,flag in [('upstream-item64-support','complete_independent_item_projection_and_all_query_rows','all_projected_items_and_query_rows_replayed_bitwise'),
        ('upstream-itemae50-user64-support','complete_independent_itemae50_user_projection_and_all_query_rows','all_projected_users_and_query_rows_replayed_bitwise')]:
        report=u.ROOT/'experiments'/namespace;proof=u.read(report/'verification.json')
        if (proof['state']!=state or not proof[flag] or proof['qualifying_labels_read'] is not False or
            proof['receipt_sha256']!=u.sha(report/'receipt.json')):raise ValueError('Full independent original support proof required')
        for key,digest in proof['bindings'].items():bind(u.ROOT/key,digest)
        for p in [report/'verification.json',report/'receipt.json']:bind(p)
        path=u.ROOT/'artifacts'/namespace/'probex.npy';bind(path)
        supports.append(np.load(path,mmap_mode='r'))
    raw_report=u.ROOT/'experiments/upstream-mps-item-autorec'
    pair=u.read(raw_report/'pair-00050.json');proof=u.read(raw_report/'snapshot-00050-verification.json')
    if (pair['epoch']!=50 or not pair['declared_full_budget_complete'] or pair['qualifying_labels_read'] is not False or
        proof['state']!='complete_independent_raw_snapshot_bindings_and_known_score' or proof['epoch']!=50 or
        not proof['actual_original_saved_query_replay_evidence_verified']):raise ValueError('Complete own AE50 pair required')
    for key,digest in proof['bindings'].items():bind(u.ROOT/key,digest)
    for p in [raw_report/'pair-00050.json',raw_report/'snapshot-00050-verification.json',Path(__file__),u.ROOT/'OWN167_CONTEXT218_PLAN.md']:bind(p)
    raw=np.load(u.ROOT/pair['pairs']['probex'],mmap_mode='r')
    if raw.shape!=(1128164,) or raw.dtype!=np.float32 or any(a.shape!=(1128164,64) or a.dtype!=np.float32 for a in supports):raise ValueError('Complete raw/support dimensions differ')
    return bindings,(np.load(PARENT/'design.npy',mmap_mode='r'),np.load(PARENT/'features.npy',mmap_mode='r'),raw,*supports)

def block(arrays,lo,hi):
    oldz,oldf,raw,item,user=arrays
    extra=np.clip(raw[lo:hi],0,6)
    z=np.column_stack([oldz[lo:hi,:-1],extra,oldz[lo:hi,-1]])
    f=np.column_stack([oldf[lo:hi,:302],extra-np.float32(3.6),oldf[lo:hi,302:],item[lo:hi],user[lo:hi]]).astype(np.float32)
    return z,f

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing virtual source evidence preserved')
    REPORT.mkdir();RUN.mkdir()
    bindings,arrays=sources();labels=np.load(PARENT/'labels.npy');users=np.load(PARENT/'users.npy');groups=np.load(PARENT/'groups.npy')
    canonical=np.load(u.FRESH/'data/probex/user_idxs.npy')
    if not np.array_equal(users,canonical) or not np.array_equal(labels,np.load(u.FRESH/'data/probex/ratings.npy')):raise ValueError('Canonical identities differ')
    unique,first,inverse=np.unique(users,return_index=True,return_inverse=True)
    if not np.array_equal(groups,groups[first][inverse]):raise ValueError('User crosses inner/outer partitions')
    if set(np.unique(groups))!=set(range(10)):raise ValueError('All original inner/outer groups required')
    grams=np.zeros((10,243,243));rhs=np.zeros((10,243));fingerprints=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,len(labels),8192):
        hi=min(lo+8192,len(labels));z,f=block(arrays,lo,hi);oz,of,raw,item,user=arrays
        if z.shape!=(hi-lo,243) or f.shape!=(hi-lo,521) or not np.isfinite(z).all() or not np.isfinite(f).all():raise ValueError('Virtual complete schema/finite values differ')
        if (not np.array_equal(z[:,:241].view('u8'),oz[lo:hi,:241].view('u8')) or
            not np.array_equal(z[:,-1],oz[lo:hi,-1]) or
            not np.array_equal(f[:,:302].view('u4'),of[lo:hi,:302].view('u4')) or
            not np.array_equal(f[:,303:393].view('u4'),of[lo:hi,302:].view('u4')) or
            not np.array_equal(f[:,393:457].view('u4'),item[lo:hi].view('u4')) or
            not np.array_equal(f[:,457:].view('u4'),user[lo:hi].view('u4'))):raise ValueError('Original retained/added context bits differ')
        for group in range(10):
            mask=groups[lo:hi]==group;a=z[mask];target=labels[lo:hi][mask]
            grams[group]+=a.T@a;rhs[group]+=a.T@target
        for digest,a in zip(fingerprints,[z,f]):digest.update(a.tobytes(order='C'))
    np.savez(RUN/'stats.npz',gram=grams,rhs=rhs)
    for key in ['labels','users','groups']:u.link_or_copy(PARENT/(key+'.npy'),RUN/(key+'.npy'))
    for key,digest in bindings.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Original source changed during assembly')
    u.write(REPORT/'manifest.json',dict(state='complete_own167_context218_virtual_inputs_and_group_statistics',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),prediction_columns=167,support_columns=218,
        rows=len(labels),users=len(unique),design_columns=243,nonlinear_features=521,nn_inputs=522,
        virtual_c_order_sha256=[h.hexdigest() for h in fingerprints],source_sha256=bindings,
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},all_original_retained_and_added_context_bits_checked=True,
        users_disjoint_inner_and_outer=True,qualifying_labels_read=False,models_fitted=0,
        scope='Complete known input/source construction only; no accuracy claim. Independent source/statistics/normalization checks and actual nested NN fitting remain. No multiGB matrices written.'))
    print('Full own167/context218 source rows and group statistics prepared without large caches',flush=True)

if __name__=='__main__':prepare()
