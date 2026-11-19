#!/usr/bin/env python3
"""Verify both full native latent-model phases and every published support row."""
import argparse
from pathlib import Path
import re
import time
import numpy as np
import upstream_latent90_local as native

u=native.u

def verify(name):
    plan=native.prepare();kind,params=native.JOBS[name]
    rp=native.REPORT/(name+'-receipt.json');r=u.read(rp)
    expected=['./target/release/fit',kind,name,'-p','pipeline-new.toml','--final','--preds','preds_new']
    if (r['state']!='complete_pending_independent_verification' or r['plan_sha256']!=u.sha(native.REPORT/'plan.json') or
        r['command']!=expected or r['model_environment']!=params or
        r['qualifying_labels_read'] is not False or r['public_prediction_values_read'] is not False):
        raise ValueError('Original full latent model provenance differs')
    bindings=dict(plan['bound_files']);log=native.REPORT/(name+'.log')
    if [int(v) for v in re.findall(r'^Epoch\s+(\d+)\s*\|',log.read_text(),re.M)]!=list(range(1,21))*2:
        raise ValueError('Both original20-epoch traces required')
    for p in [rp,log,native.REPORT/'plan.json',Path(__file__)]:bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    for rel,digest in r['outputs'].items():
        path=native.TREE/rel
        if u.sha(path)!=digest:raise ValueError('Original native output changed: '+rel)
        bindings[str(path.relative_to(u.ROOT))]=digest
    out=native.TREE/'preds_new'
    specs=[('u','ufeat','user_idxs',480189,50)] if kind=='rbm' else [
        ('u','ufeat','user_idxs',480189,20),('i','ifeat','item_idxs',17770,20)]
    for _,factor,_,rows,width in specs:
        for phase in ['trainx','fulltrain']:
            path=out/f'{name}.{factor}.{phase}.npy';a=np.load(path,mmap_mode='r')
            if str(path.relative_to(native.TREE)) not in r['outputs'] or a.shape!=(rows,width) or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError('Original learned factors invalid')
    support_vectors=0
    for split,rows in [('probex',1128164),('qual',2817131)]:
        a=np.load(out/f'{name}.{split}.npy',mmap_mode='r')
        if a.shape!=(rows,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Native query invalid')
        for prefix,factor,identity,_,width in specs:
            learned=np.load(out/f'{name}.{factor}.trainx.npy',mmap_mode='r')
            indices=np.load(native.TREE/'data'/split/(identity+'.npy'),mmap_mode='r')
            if indices.shape!=(rows,):raise ValueError('Query identity shape differs')
            for j in range(width):
                path=out/'lat'/f'{name}.{prefix}{j:02}.{split}.npy';actual=np.load(path,mmap_mode='r')
                if str(path.relative_to(native.TREE)) not in r['outputs'] or actual.shape!=(rows,) or actual.dtype!=np.float32:
                    raise ValueError('Unbound or malformed support column')
                for lo in range(0,rows,65536):
                    hi=min(lo+65536,rows);expected_values=learned[indices[lo:hi],j]
                    if not np.isfinite(actual[lo:hi]).all() or not np.array_equal(actual[lo:hi].view(np.uint32),expected_values.view(np.uint32)):
                        raise ValueError('Original factor gather differs')
                support_vectors+=1
    if np.any(np.load(native.TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    result=dict(state='complete_original_two_phase_latent_model',observed=native.stamp(),model=name,
        source_pin=u.PIN,epochs_per_phase=20,both_full_epoch_traces=True,support_vectors_verified=support_vectors,
        all_support_rows_independently_replayed_bitwise=True,original_trainx_factors_used_for_both_query_sets=True,
        bindings=bindings,qualifying_targets_zero=True,qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Separate complete original20/20 execution/data/code/output bindings, native factor/query validity and every support gather row checked. Original models are not retrained here; no blend or qualifying score claimed.')
    u.write(native.REPORT/(name+'-completion-verification.json'),result);print(name,result['state'],flush=True)

def watch():
    for name in native.JOBS:
        while not (native.REPORT/(name+'-receipt.json')).exists():time.sleep(30)
        verify(name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--model',choices=list(native.JOBS));a=p.parse_args()
    if a.watch:watch()
    elif a.model:verify(a.model)
    else:p.error('Choose --watch or --model')
