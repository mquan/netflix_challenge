#!/usr/bin/env python3
"""Check deployed support features without reading target rating outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['bank','models-dir','qual-bank','qual-root','dataset-manifest','output']:
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();b=json.loads(Path(a.bank+'.inputs.json').read_text())
    qb=json.loads(Path(a.qual_bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual_root+'.json').read_text())
    if q['qualifying_labels_read'] is not False or q['bank_sha256']!=digest(a.qual_bank+'.inputs.json'):
        raise ValueError('unverified target features')
    if b['names'][b['columns']:]!=qb['names'][qb['columns']:]:raise ValueError('support schemas differ')
    if digest(a.qual_root+'.x.npy')!=q['x_sha256']:raise ValueError('target matrix changed')
    raw=np.load(a.bank+'.x.npy',mmap_mode='r');target=np.load(a.qual_root+'.x.npy',mmap_mode='r')
    scaler=np.load(Path(a.models_dir)/'nn-scaler.npz')
    rows=np.linspace(0,len(target)-1,20000,dtype=int);shifts=[]
    for j,name in enumerate(b['names'][b['columns']:]):
        values=target[rows,qb['columns']+j].astype('f8');k=b['columns']+j+1
        normal=(np.clip(values,scaler['lo'][k],scaler['hi'][k])-scaler['mean'][k])/scaler['std'][k]
        shifts.append(dict(name=Path(name).name,normalized_target_mean=float(normal.mean()),
            normalized_target_std=float(normal.std()),
            clipped_fraction=float(np.mean((values<scaler['lo'][k])|(values>scaler['hi'][k])))))
    d=load(b['config']['data']);manifest=json.loads(Path(a.dataset_manifest).read_text())
    path=Path(a.dataset_manifest).parent/'train-probe-qualifying.bin';qd=load(path)
    comparisons=[]
    groups=[('rbm_user',range(50),'user'),('tx_item',range(50,70),'item'),('tx_user',range(70,90),'user')]
    for name,columns,field in groups:
        entities,di=np.unique(d['probe'][field],return_index=True)
        qe,qi=np.unique(qd['probe'][field],return_index=True)
        common,ia,ib=np.intersect1d(entities,qe,return_indices=True)
        pick=np.linspace(0,len(common)-1,min(2000,len(common)),dtype=int)
        maximum=0.;sq=0.;n=0
        for j in columns:
            delta=raw[di[ia[pick]],b['columns']+j].astype('f8')-target[qi[ib[pick]],qb['columns']+j]
            maximum=max(maximum,float(np.max(np.abs(delta))));sq+=float(np.dot(delta,delta));n+=len(delta)
        comparisons.append(dict(group=name,shared_entities=len(common),sampled_entities=len(pick),
            maximum_absolute_difference=maximum,rmse=float(np.sqrt(sq/n))))
    shifts.sort(key=lambda row:abs(row['normalized_target_mean']),reverse=True)
    report=dict(state='complete',qualifying_labels_read=False,rating_outcomes_loaded=False,
        bank_sha256=digest(a.bank+'.inputs.json'),qual_manifest_sha256=digest(a.qual_root+'.json'),
        source_learner='Same fitted train_3 latent models are declared for both datasets by upstream predfeat.rs',
        latent_shared_entity_checks=comparisons,feature_shifts=shifts)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(latents=comparisons,largest_shifts=shifts[:20])),flush=True)


if __name__=='__main__':main()
