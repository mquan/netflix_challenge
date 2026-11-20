#!/usr/bin/env python3
"""Fixed user contexts from independently completed own item-AE50 weights."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import datetime as dt
import numpy as np
import upstream_mps_item_autorec as raw
u=raw.u
RUN=u.ROOT/'artifacts/upstream-itemae50-user64-support'
REPORT=u.ROOT/'experiments/upstream-itemae50-user64-support'
MODELS=['encoder','decoder']
RANK=32

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def inputs():
    import torch
    raw.verify_plan()
    cp=raw.REPORT/'completion.json';completion=u.read(cp)
    pp=raw.REPORT/'pair-00050.json';pair=u.read(pp)
    vp=raw.REPORT/'snapshot-00050-verification.json';proof=u.read(vp)
    if (completion['state']!='complete_both_declared_item_oriented_phases' or
        completion['epochs_per_phase']!=50 or not completion['all_declared_snapshot_pairs_complete'] or
        completion['plan_sha256']!=u.sha(raw.REPORT/'plan.json') or
        completion['qualifying_labels_read'] is not False or
        pair['epoch']!=50 or not pair['declared_full_budget_complete'] or
        proof['state']!='complete_independent_raw_snapshot_bindings_and_known_score' or
        proof['epoch']!=50 or not proof['declared_full_budget_complete'] or
        not proof['actual_original_saved_query_replay_evidence_verified'] or
        not proof['independently_verified_csr_histories_bound'] or
        proof['qualifying_labels_read'] is not False):
        raise ValueError('Independently complete original own50/50 source and every-query replay evidence required')
    sources=dict(proof['bindings'])
    for p in [Path(__file__),cp,pp,vp,Path(raw.__file__)]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Complete raw source binding changed')
    path=raw.RUN/'trainx/snapshots/epoch-00050/model.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved['epoch']!=50 or saved['phase']!='trainx' or saved['config']!=raw.CONFIG or
        saved['plan_sha256']!=u.sha(raw.REPORT/'plan.json')):
        raise ValueError('Only actual fixed final trainx weights permitted')
    state=saved['model']
    arrays=[state['0.weight'].numpy().T.astype(np.float64),state['2.weight'].numpy().astype(np.float64)]
    del saved,state
    if any(a.shape!=(480189,128) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('All480189 original users and all128 learned dimensions required')
    queries={}
    for query,rows in [('probex',1128164),('qual',2817131)]:
        p=u.FRESH/'data'/query/'user_idxs.npy';a=np.load(p,allow_pickle=False)
        if a.shape!=(rows,) or a.min()<0 or a.max()>=480189:raise ValueError('Canonical query user IDs differ')
        queries[query]=a;sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    return arrays,queries,sources

def project(a,rank=RANK):
    center=a.mean(axis=0);x=a-center;cov=x.T@x/len(x)
    values,basis=np.linalg.eigh(cov);order=np.argsort(values)[::-1][:rank]
    values=values[order];vectors=basis[:,order]
    if values.min()<=0:raise ValueError('Positive complete projection spectrum required')
    signs=np.sign(vectors[np.abs(vectors).argmax(axis=0),np.arange(rank)])
    vectors*=signs;scales=np.sqrt(values)
    table=(x@vectors/scales).astype(np.float32)
    return table,dict(center=center,vectors=vectors,scales=scales,eigenvalues=values)

def run():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing final user-context evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=stamp(),rank_each=RANK,models=MODELS,
        script_sha256=u.sha(Path(__file__)),users=480189,original_width=128,
        rule='Final independently complete50/50 own item autoencoder only. ALL trainx encoder user columns transposed and decoder user rows. Fixed top32 centered covariance eigenvectors per matrix; canonical maximum-loading signs and population standard deviation. Same trainx user tables for known and qualifying queries, preserving coordinate identity. No outcome-based epoch/rank/side selection, ratings, raw numerical public output, qualifying labels or score claims. Full predictor/model recipe unchanged.',
        qualifying_labels_read=False,public_numerical_features_read=False))
    arrays,queries,sources=inputs();RUN.mkdir();tables=[]
    for name,a in zip(MODELS,arrays):
        table,projection=project(a);tables.append(table)
        np.savez(RUN/(name+'-projection.npz'),**projection)
    table=np.concatenate(tables,axis=1);np.save(RUN/'user64.npy',table)
    for query,ids in queries.items():
        out=np.lib.format.open_memmap(RUN/(query+'.npy'),mode='w+',dtype=np.float32,shape=(len(ids),64))
        for lo in range(0,len(ids),16384):out[lo:lo+16384]=table[ids[lo:lo+16384]]
        out.flush();del out
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source changed during extraction')
    u.write(REPORT/'receipt.json',dict(state='complete_own_unsupervised_itemae50_user_context',observed=stamp(),
        source_sha256=sources,output_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        users=480189,support_columns=64,trainx_weights_for_both_query_sets=True,
        qualifying_labels_read=False,public_numerical_features_read=False))

if __name__=='__main__':run()
