#!/usr/bin/env python3
"""Fixed unsupervised item context from two original freshly trained models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import datetime as dt
import numpy as np
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-item64-support'
REPORT=u.ROOT/'experiments/upstream-item64-support'
MODELS=['rbmx2-150','tsvdx4-400lm']
RANK=32

def inputs():
    sources={};arrays=[]
    inventory_path=u.ROOT/'experiments/upstream-latent90-local/existing-own-factor-inventory.json'
    inventory=u.read(inventory_path)
    sources[str(inventory_path.relative_to(u.ROOT))]=u.sha(inventory_path)
    for name in MODELS:
        row=next(r for r in inventory['records'] if r['model']==name)
        receipt_path=u.BASE/'fresh-receipts'/f'{name}.json'
        receipt=u.read(receipt_path)
        factor=next(r for r in row['item_factors'] if r['phase']=='trainx')
        path=u.ROOT/factor['path']
        if (u.sha(receipt_path)!=row['receipt_sha256'] or receipt['source_commit']!=u.PIN or
                receipt['provenance']!='Fresh execution of pinned upstream source; no released prediction inputs' or
                receipt['outputs']['preds_new/'+path.name]!=factor['sha256'] or u.sha(path)!=factor['sha256']):
            raise ValueError('Original own native factor lineage differs')
        a=np.load(path,allow_pickle=False)
        if list(a.shape)!=factor['shape'] or a.shape[0]!=17770 or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('Original own item factors differ')
        arrays.append(a.astype(np.float64))
        for p in [receipt_path,path]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    queries={}
    for query,rows in [('probex',1128164),('qual',2817131)]:
        path=u.FRESH/'data'/query/'item_idxs.npy';a=np.load(path,allow_pickle=False)
        if a.shape!=(rows,) or a.min()<0 or a.max()>=17770:raise ValueError('Canonical item query differs')
        queries[query]=a;sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    return arrays,queries,sources

def run():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing item context evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),models=MODELS,rank_each=RANK,
        script_sha256=u.sha(Path(__file__)),rule='Fixed top32 centered covariance eigenvectors per own trainx item table. Canonical maximum-absolute-loading sign, scores divided by population standard deviation. Same trainx tables for both query sets. No ratings or outcome-based selection in projection.',
        qualifying_labels_read=False,public_numerical_features_read=False,original_public_schema_parity_claimed=False))
    arrays,queries,sources=inputs();RUN.mkdir();tables=[];fractions=[]
    for name,a in zip(MODELS,arrays):
        center=a.mean(axis=0);x=a-center;cov=x.T@x/len(x)
        values,basis=np.linalg.eigh(cov);order=np.argsort(values)[::-1][:RANK]
        eigenvalues=values[order];vectors=basis[:,order]
        if eigenvalues.min()<=0:raise ValueError('Positive PCA spectrum required')
        signs=np.sign(vectors[np.abs(vectors).argmax(axis=0),np.arange(RANK)])
        vectors*=signs;scales=np.sqrt(eigenvalues)
        table=((x@vectors)/scales).astype(np.float32)
        np.savez(RUN/(name+'-projection.npz'),center=center,vectors=vectors,scales=scales,eigenvalues=eigenvalues)
        tables.append(table);fractions.append(float(eigenvalues.sum()/values.sum()))
    table=np.concatenate(tables,axis=1);np.save(RUN/'item64.npy',table)
    for query,ids in queries.items():
        out=np.lib.format.open_memmap(RUN/(query+'.npy'),mode='w+',dtype=np.float32,shape=(len(ids),64))
        for lo in range(0,len(ids),16384):out[lo:lo+16384]=table[ids[lo:lo+16384]]
        out.flush();del out
    sources[str(Path(__file__).relative_to(u.ROOT))]=u.sha(Path(__file__))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source changed')
    u.write(REPORT/'receipt.json',dict(state='complete_own_unsupervised_item_context',source_sha256=sources,
        output_sha256={p.name:u.sha(p) for p in RUN.iterdir()},explained_variance_fractions=fractions,
        support_columns=64,qualifying_labels_read=False,public_numerical_features_read=False,
        scope='New own PCA context recipe only. No model fitting, nonlinear ensemble gain, qualifying score or exact public schema parity claimed.'))

if __name__=='__main__':run()
