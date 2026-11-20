#!/usr/bin/env python3
"""Admission requires frozen lineage and replay from serialized preprocessing."""
import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='2'
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import torch
import upstream_nn_pool6_full_refit as job
import upstream_repro as u

def verify(name):
    root=job.RUN/name;report=job.REPORT/name
    output=report/'independent-verification.json'
    if output.exists():raise ValueError('Existing admission evidence preserved')
    plan=u.read(report/'plan.json');receipt=u.read(report/'verification.json')
    if (receipt['state']!='complete_full_native_append_nn_and_both_query_replays' or
        receipt['plan_sha256']!=u.sha(report/'plan.json') or plan['qualifying_labels_read'] is not False):raise ValueError('Original complete fit required')
    for key,digest in plan['source_sha256'].items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Full source changed')
    for filename,digest in receipt['outputs'].items():
        if u.sha(root/filename)!=digest:raise ValueError('Serialized model/preprocessing changed')
    declaration=u.read(job.DECLARATION)
    entry=next(e for e in declaration['families'] if e['development']==name)
    mp=u.ROOT/'experiments'/name/'manifest.json';manifest=u.read(mp)
    if u.sha(mp)!=entry['manifest_sha256'] or plan['config']!=entry['config'] or plan['iterations']!=entry['full_iterations']:raise ValueError('Original declared recipe differs')
    # Validate original expected hashes independently; never overwrite with current values.
    for key,digest in manifest['code_sha256'].items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Original implementation changed: '+key)
    bm=u.read(job.bank.REPORT/'manifest.json');bp=u.read(job.bank.REPORT/'column-verification.json')
    if bp['state']!='complete' or bp['manifest_sha256']!=u.sha(job.bank.REPORT/'manifest.json'):raise ValueError('Production realization proof differs')
    for key,digest in bm['files'].items():
        if u.sha(job.bank.RUN/key)!=digest:raise ValueError('B is not original verified production input')
    for key in ['labels','users']:
        a=np.load(job.bank.RUN/(key+'.npy'))
        b=np.load(u.ROOT/'artifacts'/name/(key+'.npy'))
        if not np.array_equal(a,b):raise ValueError('Known row identity differs')
    recipe=job.module_for(name);rows,_=recipe.data.screen.verified_inputs()
    # Reopen complete preprocessing for the admission replay, not in-memory fit objects.
    weights=np.load(root/'ridge.npy')
    with np.load(root/'scaler.npz') as saved:
        if set(saved.files)!={'lo','hi','mean','std'}:raise ValueError('Saved scaler keys differ')
        scaler={key:saved[key].copy() for key in saved.files}
    if weights.shape!=(183+len(rows),):raise ValueError('Serialized ridge dimensions differ')
    models=[]
    for seed,count in zip(entry['seeds'],entry['full_iterations']):
        p=root/f'nn-{seed}.pt';saved=torch.load(p,map_location='cpu',weights_only=True)
        if saved['seed']!=seed or saved['epochs']!=count or saved['feature_width']!=398+len(rows) or saved['config']!=entry['config']:
            raise ValueError('Actual saved model budget/config differs')
        models.append(recipe.neural_model(p))
    records=[]
    for query,count in [('probex',1128164),('qual',2817131)]:
        raw=[np.load(u.ROOT/r['paired_files'][query],mmap_mode='r') for r in rows]
        stored=np.fromfile(root/(query+'.f32'),dtype='<f4')
        if stored.shape!=(count,) or not np.isfinite(stored).all():raise ValueError('Full query malformed')
        if query=='probex':
            zz=np.load(job.bank.RUN/'probex-design.npy',mmap_mode='r');ff=np.load(job.bank.RUN/'probex-features.npy',mmap_mode='r')
        for lo in range(0,count,65536):
            hi=min(lo+65536,count)
            z,f=(zz[lo:hi],ff[lo:hi]) if query=='probex' else job.bank.qual_chunk(lo,hi)
            extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
            z=np.column_stack([z[:,:182],extra,z[:,-1]])
            f=np.column_stack([f[:,:243],extra-np.float32(3.6),f[:,243:]]).astype(np.float32)
            baseline=np.clip(z@weights,1.05,4.95)
            x=np.column_stack([f,baseline]).astype(np.float32)
            # Original reference transform, separate from bounded fit implementation.
            x,_=job.n.scaled(x,None,scaler=scaler)
            correction=np.zeros(hi-lo,dtype=np.float64)
            for model in models:correction+=recipe.mps_backend.predict(model,x)/len(models)
            actual=np.clip(baseline+correction,1.05,4.95).astype('<f4')
            if not np.array_equal(actual.view('u4'),stored[lo:hi].view('u4')):raise ValueError('Serialized complete pipeline replay differs')
        records.append(dict(query=query,rows=count,bitwise_equal=True,prediction_sha256=u.sha(root/(query+'.f32'))))
    for key,digest in plan['source_sha256'].items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Source changed during admission')
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),report/'plan.json',report/'verification.json',job.DECLARATION,mp]}
    u.write(output,dict(state='complete_independent_serialized_pipeline_and_original_lineage_admission',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),development=name,records=records,bindings=bindings,
        original_expected_implementation_hashes_verified_without_overwrite=True,
        production_B_manifest_files_and_known_row_identities_verified=True,
        actual_saved_ridge_scaler_and_all_models_reopened=True,qualifying_labels_read=False,
        original_declared_budgets_verified=True,no_new_cv_score_claimed=True))
    print(name,'independently admitted',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('development');a=p.parse_args();verify(a.development)
