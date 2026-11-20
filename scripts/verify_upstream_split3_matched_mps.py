#!/usr/bin/env python3
"""Independent full statistics, original scalers and actual saved-model replay."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import gc
import math
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_nonlinear as n
import upstream_mps_context as backend
import upstream_split3_matched_data as data
import upstream_split3_matched_mps as study

def score(p,y):
    canonical=np.clip(np.asarray(p,dtype=np.float64),1.05,4.95)
    return math.sqrt(math.fsum(float(v)*float(v) for v in canonical-np.asarray(y,dtype=np.float64))/len(y))

def verify():
    import torch
    report=study.REPORT;run=study.RUN;manifest=u.read(report/'manifest.json')
    if (report/'independent-verification.json').exists():raise ValueError('Existing actual full verification preserved')
    if u.read(report/'fit-completion.json')['state']!='complete_both_matched_arms_all_ten_outer_folds':raise ValueError('Every original full fit required')
    if manifest['code_sha256']!=study.code_hashes() or manifest['config']!=study.CONFIG:raise ValueError('Frozen study code changed')
    study.check_sources(manifest['source_sha256']);backend.MODEL_FACTORY=study.model_factory
    y,groups,canonical,mapping=study.arrays();inputs=study.admission.Inputs()
    fresh_g=np.zeros((2,10,data.DESIGN,data.DESIGN));fresh_r=np.zeros((2,10,data.DESIGN))
    try:
        for lo in range(0,data.ROWS,8192):
            hi=min(lo+8192,data.ROWS);z,_=inputs.block(lo,hi)
            for ai,arm in enumerate(study.ARMS):
                for g in range(10):
                    mask=(groups[lo:hi]==g)&(canonical[lo:hi] if arm=='control' else True);a=z[mask]
                    fresh_g[ai,g]+=a.T@a;fresh_r[ai,g]+=a.T@y[lo:hi][mask]
        with np.load(data.RUN/'stats.npz') as saved:
            if not np.array_equal(fresh_g,saved['gram']) or not np.array_equal(fresh_r,saved['rhs']):raise ValueError('Every full matched-arm ridge statistic must independently replay exactly')
        records=[];arm_scores=[];tree=next(r for r in manifest['controls'] if r['family']=='tree')
        tree_p=np.fromfile(u.ROOT/tree['path'],dtype='<f4');canonical_y=y[mapping]
        for ai,arm in enumerate(study.ARMS):
            prediction=np.empty(len(mapping),dtype='<f4')
            for fold in range(5):
                receipt=u.read(report/arm/f'{fold}.json');root=run/arm/str(fold)
                counts=receipt['chosen_iterations']
                if (receipt['manifest_sha256']!=u.sha(report/'manifest.json') or len(counts)!=3 or
                    any(type(c)!=int or not 0<=c<=study.CONFIG['nn_epochs'] for c in counts)):raise ValueError('All three original selected epoch budgets required')
                expected={'ridge.npz','scalers.npz','prediction.f32'}|{f'nn-{seed}.pt' for seed,c in zip(study.CONFIG['seeds'],counts) if c}
                if set(receipt['outputs'])!=expected or {p.name for p in root.iterdir()}!=expected:raise ValueError('Actual saved complete pipeline inventory differs')
                for name,h in receipt['outputs'].items():
                    if u.sha(root/name)!=h:raise ValueError('Actual saved pipeline changed')
                train,fit,valid=data.masks(groups,canonical,arm,fold);eligible=np.arange(10)//2!=fold
                iw=n.ridge_weights(fresh_g[ai],fresh_r[ai],eligible&(np.arange(10)%2==0),study.CONFIG['ridge_lambda'])
                fw=n.ridge_weights(fresh_g[ai],fresh_r[ai],eligible,study.CONFIG['ridge_lambda'])
                with np.load(root/'ridge.npz') as saved:
                    if not np.array_equal(iw,saved['inner']) or not np.array_equal(fw,saved['full']):raise ValueError('Independently recomputed eligible ridge coefficients differ')
                held=mapping[groups[mapping]//2==fold];full_held=None;baseline=None
                for name,weights,mask in [('inner',iw,fit),('full',fw,train)]:
                    raw=np.empty((data.ROWS,210),dtype=np.float32);base=np.empty(data.ROWS)
                    for lo in range(0,data.ROWS,8192):
                        hi=min(lo+8192,data.ROWS);z,f=inputs.block(lo,hi);base[lo:hi]=np.clip(z@weights,1.05,4.95)
                        raw[lo:hi,:-1]=f;raw[lo:hi,-1]=base[lo:hi]
                    # Original implementation, independently of the worker's bounded scaler.
                    normalized,scaler=n.scaled(raw,mask)
                    with np.load(root/'scalers.npz') as saved:
                        if any(not np.array_equal(value,saved[name+'_'+key]) for key,value in scaler.items()):raise ValueError('Original eligible percentile/mean/std differs from actual serialized scaler')
                    if name=='full':full_held=normalized[held].copy();baseline=base[held].copy()
                    del raw,normalized,base;gc.collect()
                correction=np.zeros(len(held));models=0
                for seed,count in zip(study.CONFIG['seeds'],counts):
                    if count:
                        checkpoint=torch.load(root/f'nn-{seed}.pt',map_location='cpu',weights_only=True)
                        if checkpoint['epochs']!=count or checkpoint['seed']!=seed+fold or checkpoint['feature_width']!=210:raise ValueError('Actual saved model epoch/seed/width differs')
                        model=backend.load_model(root/f'nn-{seed}.pt',study.CONFIG)
                        correction+=backend.predict(model,full_held)/3
                        del model,checkpoint;torch.mps.empty_cache();models+=1
                replay=np.clip(baseline+correction,1.05,4.95).astype('<f4');saved=np.fromfile(root/'prediction.f32',dtype='<f4')
                if not np.array_equal(replay.view('u4'),saved.view('u4')):raise ValueError('Complete actual saved-model held vector differs')
                prediction[groups[mapping]//2==fold]=replay
                records.append(dict(arm=arm,fold=fold,canonical_rows=len(replay),actual_models_reopened=models,
                    saved_pipeline_replay_bitwise_equal=True,both_original_eligible_scalers_independently_recomputed=True,rmse=score(replay,y[held])))
                del full_held,baseline;gc.collect();n.event(stage='independent_matched_fold_verified',**records[-1])
            prediction.tofile(run/(arm+'-canonical.f32'))
            half=np.clip(.5*prediction.astype(np.float64)+.5*tree_p.astype(np.float64),1.05,4.95).astype('<f4')
            half.tofile(run/(arm+'-tree-half.f32'))
            arm_scores.append(dict(arm=arm,canonical_rmse=score(prediction,canonical_y),fixed_tree_half_rmse=score(half,canonical_y),
                canonical_sha256=u.sha(run/(arm+'-canonical.f32')),fixed_half_sha256=u.sha(run/(arm+'-tree-half.f32'))))
        study.check_sources(manifest['source_sha256'])
        result=dict(state='complete_both_matched_arms_original_statistics_scalers_and_actual_saved_model_replays',
            manifest_sha256=u.sha(report/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),records=records,arms=arm_scores,
            expanded_beats_matched_control=arm_scores[1]['canonical_rmse']<arm_scores[0]['canonical_rmse'],
            expanded_half_beats_global=arm_scores[1]['fixed_tree_half_rmse']<manifest['preceding_global_development_rmse'],
            preceding_global_development_rmse=manifest['preceding_global_development_rmse'],qualifying_labels_read=False,production_started=False)
        u.write(report/'independent-verification.json',result)
    finally:inputs.close()

if __name__=='__main__':verify()
