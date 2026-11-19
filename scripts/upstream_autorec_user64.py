#!/usr/bin/env python3
"""Fixed epoch20 own encoder: unsupervised64-component user context."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import fcntl
import hashlib
import subprocess
import sys
import time
import numpy as np
import upstream_mps_autorec as raw

u=raw.u
RUN=u.ROOT/'artifacts/upstream-autorec-user64'
REPORT=u.ROOT/'experiments/upstream-autorec-user64'
EPOCH,RANK,PHASE=20,64,'trainx'
CONFIG=dict(epoch=20,rank=64,users=480189,items=17770,hidden=512,batch_users=256,phase='trainx')

def source_inputs():
    plan=raw.verify_plan();path=raw.REPORT/f'snapshot-{EPOCH:05}-verification.json';proof=u.read(path)
    if (proof['state']!='complete_independent_raw_snapshot_bindings_and_known_score' or proof['epoch']!=EPOCH or
            not proof['actual_original_saved_query_replay_evidence_verified'] or proof['qualifying_labels_read'] is not False):
        raise ValueError('Complete independently checked fixed epoch20 source required')
    sources={**plan['source_sha256'],**plan['code_sha256'],**proof['bindings']}
    for p in [path,raw.REPORT/'plan.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    weights=raw.RUN/PHASE/'snapshots'/f'epoch-{EPOCH:05}'/'model.pt'
    if sources.get(str(weights.relative_to(u.ROOT)))!=u.sha(weights):raise ValueError('Actual own encoder snapshot differs')
    queries={}
    for query,count in [('probex',1128164),('qual',2817131)]:
        p=u.FRESH/'data'/query/'user_idxs.npy';ids=np.load(p,mmap_mode='r')
        if ids.shape!=(count,) or ids.min()<0 or ids.max()>=CONFIG['users']:raise ValueError('Canonical user query differs')
        if sources.get(str(p.relative_to(u.ROOT)))!=u.sha(p):raise ValueError('Query not bound by original raw history proof')
        queries[query]=ids
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original own encoder lineage changed')
    return weights,queries,sources

def load_encoder(weights):
    import torch
    saved=torch.load(weights,map_location='cpu',weights_only=False)
    if (saved['config']!=raw.CONFIG or saved['phase']!=PHASE or saved['epoch']!=EPOCH or
            saved['plan_sha256']!=u.sha(raw.REPORT/'plan.json')):raise ValueError('Own saved encoder recipe differs')
    model=raw.make_model(raw.CONFIG);model.load_state_dict(saved['model'])
    return model[:2].to('mps').eval()

def hidden_batches(encoder):
    import torch
    with torch.no_grad():
        for lo in range(0,CONFIG['users'],CONFIG['batch_users']):
            hi=min(lo+CONFIG['batch_users'],CONFIG['users'])
            x=torch.from_numpy(raw.dense_history(PHASE,np.arange(lo,hi))).to('mps')
            h=np.ascontiguousarray(encoder(x).cpu().numpy())
            if h.shape!=(hi-lo,CONFIG['hidden']) or h.dtype!=np.float32 or not np.isfinite(h).all():raise ValueError('Finite complete encoder outputs required')
            yield lo,hi,h

def query_hashes(table,queries):
    result={}
    for query,ids in queries.items():
        digest=hashlib.sha256()
        for lo in range(0,len(ids),16384):digest.update(np.ascontiguousarray(table[ids[lo:lo+16384]]).tobytes())
        result[query]=dict(rows=len(ids),columns=RANK,dtype='float32',virtual_c_order_sha256=digest.hexdigest())
    return result

def extract():
    import torch
    if RUN.exists():raise ValueError('Existing own user context outputs preserved')
    weights,queries,sources=source_inputs();torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.20)
    RUN.mkdir();encoder=load_encoder(weights);total=np.zeros(CONFIG['hidden']);second=np.zeros((CONFIG['hidden'],CONFIG['hidden']))
    digest=hashlib.sha256()
    for lo,hi,h in hidden_batches(encoder):
        a=h.astype(np.float64);total+=a.sum(axis=0);second+=a.T@a;digest.update(h.tobytes())
    first_hash=digest.hexdigest();del encoder
    center=total/CONFIG['users'];covariance=second/CONFIG['users']-np.outer(center,center)
    values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1][:RANK];eigenvalues=values[order];basis=vectors[:,order]
    if eigenvalues.min()<=1e-12:raise ValueError('Positive fixed-rank spectrum required')
    basis*=np.sign(basis[np.abs(basis).argmax(axis=0),np.arange(RANK)]);scales=np.sqrt(eigenvalues)
    np.savez(RUN/'projection.npz',center=center,basis=basis,scales=scales,eigenvalues=eigenvalues,covariance=covariance)
    np.savez(RUN/'moments.npz',sum=total,second=second,count=np.array(CONFIG['users']))
    # A fresh load must repeat every hidden user vector, not just a sample.
    encoder=load_encoder(weights);digest=hashlib.sha256()
    table=np.lib.format.open_memmap(RUN/'user64.npy',mode='w+',dtype=np.float32,shape=(CONFIG['users'],RANK))
    for lo,hi,h in hidden_batches(encoder):
        digest.update(h.tobytes());table[lo:hi]=((h.astype(np.float64)-center)@basis/scales).astype(np.float32)
    table.flush();del encoder
    if digest.hexdigest()!=first_hash:raise ValueError('Actual complete saved encoder replay differs bitwise')
    hashes=query_hashes(table,queries);del table
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source changed during extraction')
    u.write(REPORT/'receipt.json',dict(state='complete_own_fixed_user64_projection_pending_independent_check',config=CONFIG,
        source_sha256=sources,output_sha256={p.name:u.sha(p) for p in RUN.iterdir()},hidden_c_order_sha256=first_hash,
        actual_full_saved_encoder_replay_bitwise=True,query_features=hashes,
        explained_variance_fraction=float(eigenvalues.sum()/np.trace(covariance)),qualifying_labels_read=False,
        raw_full50_completion_claimed=False,scope='Own epoch20 trainx encoder, all users in canonical index order. Unsupervised centered top64 PCA with canonical signs and population-variance scaling. Same trainx user table for both queries. No public numerical features, outcome-based rank/epoch selection, nonlinear fitting or accuracy claimed.'))

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing user projection declaration preserved')
        paths=[Path(__file__),u.ROOT/'scripts/verify_upstream_autorec_user64.py']
        u.write(REPORT/'plan.json',dict(created=raw.stamp(),config=CONFIG,code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths},
            rule='Fixed epoch20, rank64, all480189 trainx histories and learned encoder. No dropout at extraction. Two complete saved-model encoder passes must match bitwise. Center/covariance fitted without query ratings; no epoch/rank or qualifying-based search. Require separate saved-encoder inference and covariance/projection check before downstream use.',qualifying_labels_read=False))
        while not (raw.REPORT/f'pair-{EPOCH:05}.json').exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_complete_fixed_epoch20_raw_pair',observed=raw.stamp()));time.sleep(20)
        check=raw.REPORT/f'snapshot-{EPOCH:05}-verification.json'
        if not check.exists():
            with (REPORT/'raw-snapshot-check.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_mps_autorec_snapshot.py'),'--epoch',str(EPOCH)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        u.write(REPORT/'status.json',dict(state='waiting_for_shared_encoder_extraction_slot',observed=raw.stamp()))
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX);extract()
            with (REPORT/'independent.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_autorec_user64.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        u.write(REPORT/'status.json',dict(state='complete_independently_verified_own_user64_context',observed=raw.stamp()))

if __name__=='__main__':run()
