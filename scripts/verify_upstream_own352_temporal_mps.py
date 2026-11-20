#!/usr/bin/env python3
"""Independent original full scalers/statistics and actual saved own352 model replay."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import gc
import math
from pathlib import Path
import numpy as np
import upstream_own352_temporal_mps as study

u,n,data,mps=study.u,study.n,study.data,study.mps

def score(p,y):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(np.asarray(p,dtype=np.float64),1.05,4.95),y))/len(y))

def main():
    import torch
    report=study.REPORT;run=study.RUN;manifest=u.read(report/'manifest.json')
    if (report/'independent-verification.json').exists():raise ValueError('Existing actual complete neural replay retained')
    if u.read(report/'fit-completion.json')['state']!='complete_all_five_original_budget_own352_fits':raise ValueError('All five actual original-budget fits required')
    if manifest['config']!=study.CONFIG or manifest['code_sha256']!=study.hashes() or manifest['versions']!=n.versions():raise ValueError('Frozen numerical recipe/runtime changed')
    data.check(manifest['source_sha256']);mps.MODEL_FACTORY=study.model_factory
    rows=u.read(data.REPORT/'manifest.json')['recent_rows'];inputs=data.Inputs(u.read(data.REPORT/'manifest.json')['original_rows'],rows)
    y=np.load(run/'labels.npy');groups=np.load(run/'groups.npy')
    gram=np.zeros((10,428,428));rhs=np.zeros((10,428))
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));z,_=inputs.block(lo,hi)
        for g in range(10):
            keep=groups[lo:hi]==g;a=z[keep];gram[g]+=a.T@a;rhs[g]+=a.T@y[lo:hi][keep]
    with np.load(run/'stats.npz') as saved:
        if not np.array_equal(gram,saved['gram']) or not np.array_equal(rhs,saved['rhs']):raise ValueError('Every complete original eligible group statistic must match')
    records=[];prediction=np.empty(len(y),dtype='<f4')
    for fold in range(5):
        root=run/'nn'/str(fold);receipt_path=report/'nn'/f'{fold}.json';receipt=u.read(receipt_path);counts=receipt['chosen_iterations']
        if receipt['manifest_sha256']!=u.sha(report/'manifest.json') or len(counts)!=3 or any(type(c)!=int or not 0<=c<=40 for c in counts):raise ValueError('All three original seed budgets required')
        expected={'ridge.npz','scalers.npz','prediction.f32'}|{f'nn-{s}.pt' for s,c in zip(study.CONFIG['seeds'],counts) if c}
        if set(receipt['outputs'])!=expected or {p.name for p in root.iterdir()}!=expected:raise ValueError('Actual full saved pipeline inventory differs')
        for name,h in receipt['outputs'].items():
            if u.sha(root/name)!=h:raise ValueError('Actual saved model/preprocessing/vector changed')
        train=groups//2!=fold;held=~train;fit=train&(groups%2==0);eligible=np.arange(10)//2!=fold
        reg=np.eye(428)*study.CONFIG['ridge_lambda'];reg[-1,-1]=0
        selectors=[eligible&(np.arange(10)%2==0),eligible]
        weights=[np.linalg.solve(gram[s].sum(0)+reg,rhs[s].sum(0)) for s in selectors]
        with np.load(root/'ridge.npz') as saved:
            if not np.array_equal(weights[0],saved['inner']) or not np.array_equal(weights[1],saved['full']):raise ValueError('Independent eligible ridge weights differ')
        full_held=None;baseline=None
        for ident,w,mask in [('inner',weights[0],fit),('full',weights[1],train)]:
            raw=np.empty((len(y),579),dtype=np.float32);base=np.empty(len(y))
            for lo in range(0,len(y),8192):
                hi=min(lo+8192,len(y));z,f=inputs.block(lo,hi);base[lo:hi]=np.clip(z@w,1.05,4.95)
                raw[lo:hi,:-1]=f;raw[lo:hi,-1]=base[lo:hi]
            # Complete original eligible percentile/clip/mean/std operations;
            # release statistical buffers before pointwise normalization.
            selected=raw[mask];lo,hi=np.percentile(selected,[1,99],axis=0)
            clipped=np.clip(selected,lo,hi).astype(np.float64);del selected
            scaler=dict(lo=lo,hi=hi,mean=clipped.mean(0),std=np.maximum(clipped.std(0),1e-6))
            del clipped;gc.collect()
            with np.load(root/'scalers.npz') as saved:
                if any(not np.array_equal(value,saved[ident+'_'+key]) for key,value in scaler.items()):raise ValueError('Every original complete eligible scaler statistic must match')
            if ident=='full':full_held=np.empty((int(held.sum()),579),dtype=np.float32);baseline=base[held].copy()
            cursor=0
            for lo in range(0,len(y),8192):
                hi=min(lo+8192,len(y))
                reference=np.asarray((np.clip(raw[lo:hi],scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std'],dtype=np.float32,order='C')
                bounded=study.scaling.apply_scaler(raw[lo:hi],scaler)
                if not np.array_equal(reference.view('u4'),bounded.view('u4')):raise ValueError('Every original normalized579-input float32 bit must match')
                if ident=='full':
                    keep=held[lo:hi];count=int(keep.sum());full_held[cursor:cursor+count]=reference[keep];cursor+=count
            if ident=='full' and cursor!=len(full_held):raise ValueError('Complete original held rows required')
            del raw,base;gc.collect()
        correction=np.zeros(int(held.sum()));reopened=0
        for seed,count in zip(study.CONFIG['seeds'],counts):
            if not count:continue
            saved=torch.load(root/f'nn-{seed}.pt',map_location='cpu',weights_only=True)
            if saved['epochs']!=count or saved['seed']!=seed+fold or saved['feature_width']!=579:raise ValueError('Original full selected model seed/epoch/width differs')
            model=mps.load_model(root/f'nn-{seed}.pt',study.CONFIG);correction+=mps.predict(model,full_held)/3
            del model,saved;torch.mps.empty_cache();reopened+=1
        replay=np.clip(baseline+correction,1.05,4.95).astype('<f4');stored=np.fromfile(root/'prediction.f32',dtype='<f4')
        if not np.array_equal(replay.view('u4'),stored.view('u4')):raise ValueError('Every actual complete saved-model held prediction must match')
        prediction[held]=replay
        records.append(dict(fold=fold,rows=len(replay),actual_models_reopened=reopened,receipt_path=str(receipt_path.relative_to(u.ROOT)),
            receipt_sha256=u.sha(receipt_path),outputs=receipt['outputs'],both_original_eligible_scalers_recomputed=True,
            all_original_normalized_input_bits_equal=True,actual_saved_pipeline_bitwise_equal=True,rmse=score(replay,y[held])))
        del full_held,baseline;gc.collect();n.event(stage='independent_actual_own352_fold_verified',**records[-1])
    inputs.close()
    prediction.tofile(run/'nn.f32')
    tree=next(c for c in manifest['controls'] if c['id']=='tree154');tree_p=np.fromfile(u.ROOT/tree['prediction'],dtype='<f4')
    half=np.clip(.5*prediction.astype(np.float64)+.5*tree_p.astype(np.float64),1.05,4.95).astype('<f4');half.tofile(run/'fixed-half-tree154.f32')
    rmse=score(prediction,y);half_rmse=score(half,y);data.check(manifest['source_sha256'])
    for row in records:
        if u.sha(u.ROOT/row['receipt_path'])!=row['receipt_sha256']:raise ValueError('Actual original fit receipt changed during verification')
    controls={c['id']:c['rmse'] for c in manifest['controls']}
    u.write(report/'independent-verification.json',dict(state='complete_actual_own352_all_fits_original_statistics_scalers_and_saved_model_replays',
        manifest_sha256=u.sha(report/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),records=records,
        nn_rmse=rmse,fixed_half_tree154_rmse=half_rmse,nn_sha256=u.sha(run/'nn.f32'),fixed_half_sha256=u.sha(run/'fixed-half-tree154.f32'),
        nn_improves_nn330=rmse<controls['nn330'],
        declared_global_reference_rmse=manifest['declared_global_reference_rmse'],fixed_half_below_declared_global_reference=half_rmse<manifest['declared_global_reference_rmse'],
        qualifying_labels_read=False,production_started=False,goal_complete=False))
    print('Independent complete own352 NN',rmse,'fixed tree154 half',half_rmse,'no qualifying refit',flush=True)

if __name__=='__main__':main()
