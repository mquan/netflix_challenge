#!/usr/bin/env python3
"""Full-budget native-append NN refits on the paired production realization."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import datetime as dt
import gc
import importlib
from pathlib import Path
import numpy as np
import torch
import upstream_repro as u
import upstream_nonlinear as n
import upstream_exact_neural_scaling as scaling
import upstream_item154_stream_bank as bank

RUN=u.ROOT/'artifacts/upstream-nn-pool6-full-refit'
REPORT=u.ROOT/'experiments/upstream-nn-pool6-full-refit'
DECLARATION=u.ROOT/'experiments/upstream-native-nn-pool6-production/declaration.json'
CHUNK=65536

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def module_for(name):
    module=importlib.import_module(name.replace('-','_'))
    module.configure()
    return getattr(module,'recipe',module)

def append(z,f,raw):
    extra=np.column_stack([np.clip(a,0,6) for a in raw])
    return (np.column_stack([z[:,:182],extra,z[:,-1]]),
        np.column_stack([f[:,:243],extra-np.float32(3.6),f[:,243:]]).astype(np.float32))

def worker(name):
    root=RUN/name;report=REPORT/name
    if root.exists() or report.exists():raise ValueError('Existing full refit preserved')
    root.mkdir(parents=True);report.mkdir(parents=True)
    declared=u.read(DECLARATION)
    entry=next(r for r in declared['families'] if r['development']==name)
    if entry['reuse_existing_full_base']:raise ValueError('Base full model must be reused')
    recipe=module_for(name);data=recipe.data
    mpath=u.ROOT/'experiments'/name/'manifest.json';manifest=u.read(mpath)
    if u.sha(mpath)!=entry['manifest_sha256'] or recipe.CONFIG!=entry['config']:
        raise ValueError('Frozen development recipe differs')
    for fold,digest in enumerate(entry['actual_fold_receipt_sha256']):
        if u.sha(u.ROOT/'experiments'/name/'nn'/f'{fold}.json')!=digest:
            raise ValueError('Actual original budget receipt changed')
    if entry['full_iterations']!=[int(np.median([u.read(u.ROOT/'experiments'/name/'nn'/f'{f}.json')['chosen_iterations'][j] for f in range(5)])) for j in range(3)]:
        raise ValueError('Original median budgets differ')
    rows,sources=data.screen.verified_inputs()
    bm=u.read(bank.REPORT/'manifest.json');bp=u.read(bank.REPORT/'column-verification.json')
    if bp['manifest_sha256']!=u.sha(bank.REPORT/'manifest.json') or bp['state']!='complete':raise ValueError('Paired bank proof differs')
    for collection in [bm['source_sha256'],bm['code_sha256'],manifest['code_sha256']]:
        for key,digest in collection.items():
            if key in sources and sources[key]!=digest:raise ValueError('Conflicting source provenance')
            sources[key]=digest
    for p in [Path(__file__),Path(scaling.__file__),Path(bank.__file__),mpath,DECLARATION,
              bank.REPORT/'manifest.json',bank.REPORT/'column-verification.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for filename,digest in bm['files'].items():sources[str((bank.RUN/filename).relative_to(u.ROOT))]=digest
    for row in rows:
        for key,count in [('probex',1128164),('qual',2817131)]:
            p=u.ROOT/row['paired_files'][key];a=np.load(p,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Full native pair differs')
            sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for key,digest in sources.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Source changed: '+key)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    plan=dict(state='declared_before_full_fit_and_any_new_qualifying_audit',created=stamp(),development=name,
        config=recipe.CONFIG,iterations=entry['full_iterations'],seeds=entry['seeds'],source_sha256=sources,
        raw_columns=[r['name'] for r in rows],qualifying_labels_read=False,
        input_realization='Existing paired production-known B and matching fulltrain qualifying B, with original own native additions. Development score belongs to A; no matched B-trained CV claim.',
        inference_chunk_rows=CHUNK,scaling='Original complete known-row percentile/mean/std, exact bounded implementation',
        rule='Same frozen architecture/config/seeds and median full refit budgets. No qualifying-based stopping or recipe selection. Reopen actual models and replay complete known/qualifying outputs.')
    u.write(report/'plan.json',plan)
    labels=np.load(bank.RUN/'labels.npy').astype(np.float64)
    raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in rows]
    oldz=np.load(bank.RUN/'probex-design.npy',mmap_mode='r')
    # Build ridge design only, release it before assembling the neural matrix.
    extra=np.column_stack([np.clip(a,0,6) for a in raw])
    z=np.column_stack([oldz[:,:182],extra,oldz[:,-1]])
    gram=np.zeros((z.shape[1],z.shape[1]));rhs=np.zeros(z.shape[1])
    for lo in range(0,len(z),8192):
        a=z[lo:lo+8192];gram+=a.T@a;rhs+=a.T@labels[lo:lo+8192]
    weights=n.ridge_weights(gram[None],rhs[None],np.array([True]),recipe.CONFIG['ridge_lambda'])
    np.save(root/'ridge.npy',weights);baseline=np.clip(z@weights,1.05,4.95)
    del z,oldz,a;gc.collect()
    oldf=np.load(bank.RUN/'probex-features.npy',mmap_mode='r')
    features=np.empty((len(labels),398+len(raw)),dtype=np.float32)
    features[:,:243]=oldf[:,:243];features[:,243:243+len(raw)]=extra-np.float32(3.6)
    features[:,243+len(raw):-1]=oldf[:,243:];features[:,-1]=baseline
    del oldf,extra,raw;gc.collect()
    scaler=scaling.fit_scaler(features,np.ones(len(labels),dtype=bool));np.savez(root/'scaler.npz',**scaler)
    scaling.apply_scaler(features,scaler,output=features)
    for seed,count in zip(entry['seeds'],entry['full_iterations']):
        u.write(report/'status.json',dict(state='full_known_refit',seed=seed,epochs=count,observed=stamp()))
        model,_=recipe.mps_backend.train_nn(features,labels-baseline,recipe.CONFIG,seed,count)
        torch.save(dict(state=model.state_dict(),config=recipe.CONFIG,feature_width=features.shape[1],seed=seed,epochs=count),root/f'nn-{seed}.pt')
        del model;torch.mps.empty_cache()
    del features,labels,baseline;gc.collect()
    def infer(query):
        count=1128164 if query=='probex' else 2817131
        native=[np.load(u.ROOT/r['paired_files'][query],mmap_mode='r') for r in rows]
        result=np.empty(count,dtype='<f4')
        models=[recipe.neural_model(root/f'nn-{seed}.pt') for seed in entry['seeds']]
        if query=='probex':
            zz=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');ff=np.load(bank.RUN/'probex-features.npy',mmap_mode='r')
        for lo in range(0,count,CHUNK):
            hi=min(lo+CHUNK,count)
            z,f=(zz[lo:hi],ff[lo:hi]) if query=='probex' else bank.qual_chunk(lo,hi)
            z,f=append(z,f,[a[lo:hi] for a in native]);base=np.clip(z@weights,1.05,4.95)
            x=scaling.append_baseline(f,base);scaling.apply_scaler(x,scaler,output=x)
            correction=np.zeros(hi-lo,dtype=np.float64)
            for model in models:correction+=recipe.mps_backend.predict(model,x)/len(models)
            result[lo:hi]=np.clip(base+correction,1.05,4.95).astype('<f4')
        del models;torch.mps.empty_cache()
        return result
    for query in ['probex','qual']:
        u.write(report/'status.json',dict(state='inference_and_actual_model_replay',query=query,observed=stamp()))
        path=root/(query+'.f32');infer(query).tofile(path)
        replay=infer(query);stored=np.fromfile(path,dtype='<f4')
        if not np.array_equal(replay.view('u4'),stored.view('u4')):raise ValueError('Actual full-query saved-model replay differs')
    for key,digest in sources.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Source changed during full fit')
    u.write(report/'verification.json',dict(state='complete_full_native_append_nn_and_both_query_replays',
        plan_sha256=u.sha(report/'plan.json'),observed=stamp(),development=name,
        outputs={p.name:u.sha(p) for p in root.iterdir()},actual_models_reopened_for_each_query=True,
        complete_known_and_qualifying_bitwise_replay=True,qualifying_labels_read=False,
        full_iterations=entry['full_iterations'],no_new_cv_score_claimed=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('development');args=parser.parse_args();worker(args.development)
