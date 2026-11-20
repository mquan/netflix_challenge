#!/usr/bin/env python3
"""Original eligible statistics, actual saved-model replay and canonical scoring."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import gc
import math
from pathlib import Path
import numpy as np
import upstream_own167_context218_mps as experiment

u=experiment.u

def selected_inputs(mask,baseline):
    arrays=experiment.open_arrays();result=np.empty((int(mask.sum()),522),dtype=np.float32);offset=0
    for lo in range(0,len(mask),8192):
        hi=min(lo+8192,len(mask));keep=mask[lo:hi];count=int(keep.sum())
        _,features=experiment.data.block(arrays,lo,hi)
        result[offset:offset+count]=np.column_stack([features,baseline[lo:hi]]).astype(np.float32)[keep]
        offset+=count
    if offset!=len(result):raise ValueError('Incomplete original selected rows')
    return result

def original_scaler(mask,baseline):
    # Exact selected C-order dimensions/order, full percentiles and original std.
    values=selected_inputs(mask,baseline);lo,hi=np.percentile(values,[1,99],axis=0)
    clipped=np.asarray(np.clip(values,lo,hi),dtype=np.float64,order='C')
    del values;gc.collect()
    scaler=dict(lo=lo,hi=hi,mean=clipped.mean(0),std=np.maximum(clipped.std(0),1e-6))
    return scaler

def score(values,y):
    clipped=np.clip(values.astype(np.float64),1.05,4.95)
    return math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(clipped,y))/len(y))

def verify():
    import torch
    experiment.configure();root,report=experiment.RUN,experiment.REPORT
    if (report/'independent-verification.json').exists():raise ValueError('Existing complete proof preserved')
    mp=report/'manifest.json';manifest=u.read(mp)
    if (manifest['source_commit']!=u.PIN or manifest['config']!=experiment.CONFIG or
        manifest['code_sha256']!=experiment.code_hashes() or manifest['versions']!=experiment.n.versions()):raise ValueError('Frozen complete recipe changed')
    prepared={str((root/name).relative_to(u.ROOT)):digest for name,digest in manifest['prepared_sha256'].items()}
    signatures=experiment.verify_sources(dict(manifest['source_sha256'],**prepared))
    for name,digest in manifest['prepared_sha256'].items():
        if u.sha(root/name)!=digest:raise ValueError('Prepared data changed')
    y=np.load(root/'labels.npy');groups=np.load(root/'groups.npy');users=np.load(root/'users.npy')
    for key,canonical in [('labels','ratings'),('users','user_idxs')]:
        if not np.array_equal(np.load(root/(key+'.npy')),np.load(u.FRESH/'data/probex'/(canonical+'.npy'))):raise ValueError('Canonical known identity differs')
    _,first,inverse=np.unique(users,return_index=True,return_inverse=True)
    if not np.array_equal(groups,groups[first][inverse]):raise ValueError('A user crosses nested partitions')
    assembled=np.empty(len(y),dtype='<f4');records=[];bindings={str(mp.relative_to(u.ROOT)):u.sha(mp)}
    for fold in range(5):
        experiment.unchanged(signatures);path=report/'nn'/f'{fold}.json';receipt=u.read(path);folder=root/'nn'/str(fold)
        if receipt['manifest_sha256']!=u.sha(mp) or receipt['fold']!=fold or receipt['family']!='nn':raise ValueError('Actual complete fold receipt differs')
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
        for name,digest in receipt['outputs'].items():
            p=folder/name
            if u.sha(p)!=digest:raise ValueError('Saved model/preprocessing changed')
            bindings[str(p.relative_to(u.ROOT))]=digest
        with np.load(folder/'ridge.npz') as saved:weights={key:saved[key].copy() for key in ['inner','full']}
        with np.load(root/'stats.npz') as saved:
            eligible=np.arange(10)//2!=fold
            for arm,mask in [('inner',eligible&(np.arange(10)%2==0)),('full',eligible)]:
                expected=experiment.n.ridge_weights(saved['gram'],saved['rhs'],mask,experiment.CONFIG['ridge_lambda'])
                if not np.array_equal(expected.view('u8'),weights[arm].view('u8')):raise ValueError('Exact original fold-only ridge coefficients differ')
        z=experiment.design();baselines={arm:np.clip(z@w,1.05,4.95) for arm,w in weights.items()};del z;gc.collect()
        training=groups//2!=fold;held=~training;full_scaler=None
        for arm,mask in [('inner',training&(groups%2==0)),('full',training)]:
            expected=original_scaler(mask,baselines[arm])
            with np.load(folder/'scalers.npz') as saved:actual={key:saved[arm+'_'+key].copy() for key in expected}
            for key in expected:
                if not np.array_equal(expected[key].view('u8'),actual[key].view('u8')):raise ValueError(f'Original eligible scaler differs {fold}/{arm}/{key}')
            if arm=='full':full_scaler=actual
        inputs=selected_inputs(held,baselines['full'])
        inputs,_=experiment.n.scaled(inputs,None,scaler=full_scaler)
        correction=np.zeros(len(inputs),dtype=np.float64);models=0
        for seed,count in zip(experiment.CONFIG['seeds'],receipt['chosen_iterations']):
            if not count:continue
            cp=folder/f'nn-{seed}.pt';saved=torch.load(cp,map_location='cpu',weights_only=True)
            if (saved['seed']!=seed+fold or saved['epochs']!=count or saved['feature_width']!=522 or
                saved['config']!=experiment.CONFIG):raise ValueError('Actual saved architecture/seed/budget differs')
            model=experiment.mps.load_model(cp,experiment.CONFIG)
            correction+=experiment.mps.predict(model,inputs)/len(experiment.CONFIG['seeds'])
            del model,saved;torch.mps.empty_cache();models+=1
        actual=np.clip(baselines['full'][held]+correction,1.05,4.95).astype('<f4')
        stored=np.fromfile(folder/'prediction.f32',dtype='<f4')
        if not np.array_equal(actual.view('u4'),stored.view('u4')):raise ValueError('Every actual held prediction must replay bitwise')
        assembled[held]=stored
        records.append(dict(fold=fold,rows=len(stored),bitwise_equal=True,actual_models=models,
            both_original_complete_eligible_scaler_statistics_bitwise_equal=True,original_fold_only_ridge_bits_equal=True))
        u.write(report/'replay-status.json',dict(state='actual_saved_fold_and_original_statistics_replay',records=records))
        del inputs,baselines;gc.collect()
    assembled.tofile(root/'nn.f32')
    controls={r['id']:np.fromfile(u.ROOT/r['prediction'],dtype='<f4') for r in manifest['controls']}
    if any(a.shape!=y.shape or not np.isfinite(a).all() for a in controls.values()):raise ValueError('Complete finite known control vectors required')
    half=((controls['tree154'].astype(np.float64)+assembled.astype(np.float64))*.5).astype('<f4')
    half.tofile(root/'fixed-half-tree-nn.f32')
    scores=[]
    for arm,values in [('nn',assembled),('fixed-half-tree-nn',half)]:
        if not np.isfinite(values).all():raise ValueError('Nonfinite complete vector')
        scores.append(dict(arm=arm,rmse=score(values,y),fold_rmse=[score(values[groups//2==f],y[groups//2==f]) for f in range(5)],prediction_sha256=u.sha(root/(arm+'.f32'))))
    control_scores={key:score(values,y) for key,values in controls.items()}
    experiment.verify_sources(dict(manifest['source_sha256'],**prepared))
    u.write(report/'nn-replay.json',dict(records=records,family='nn',verifier_sha256=u.sha(Path(__file__))))
    u.write(report/'independent-verification.json',dict(state='complete_actual_own167_context218_folds_replays_statistics_and_scores',
        observed=experiment.stamp(),records=scores,fold_replays=records,control_scores=control_scores,
        nn_improves_nn154=scores[0]['rmse']<control_scores['nn154'],nn_improves_nn166=scores[0]['rmse']<control_scores['nn166'],
        fixed_half_improves_global=scores[1]['rmse']<manifest['preceding_global_development_rmse'],
        preceding_global_development_rmse=manifest['preceding_global_development_rmse'],bindings=bindings,
        source_sha256=manifest['source_sha256'],verifier_sha256=u.sha(Path(__file__)),
        all_actual_saved_models_reopened=True,both_original_eligible_scalers_independently_recomputed_each_fold=True,
        complete_canonical_compensated_known_scores=True,qualifying_labels_read=False,production_started=False))
    print(scores,control_scores,flush=True)

if __name__=='__main__':verify()
