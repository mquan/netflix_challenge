#!/usr/bin/env python3
"""Separate complete paired391 operator, saved-model and streamed-bag replay."""
import upstream_complete_trajectory_fold_bag as production
import hashlib
from pathlib import Path
import numpy as np
import torch
import verify_upstream_complete_trajectory_data as independent

u,study=production.u,production.study


def main():
    output=production.REPORT/'independent-verification.json'
    if output.exists():raise ValueError('Existing complete matched391 qualifying replay retained')
    m,data,bindings,actual=production.sources()
    plan=u.read(production.REPORT/'plan.json');complete=u.read(production.REPORT/'inference-completion.json')
    if (complete['state']!='complete_all_trajectory_streamed_qualifying_bag_pending_independent_replay' or
        complete['plan_sha256']!=study.data.ops.sha(production.REPORT/'plan.json') or
        plan['source_sha256']!=bindings or plan['config']!=m['config'] or
        plan['rows']!=2817131 or plan['chunk_rows']!=8192 or plan['folds']!=list(range(5)) or
        plan['seed_slots']!=m['config']['seeds'] or plan['qualifying_labels_read'] is not False or
        plan['models_fitted']!=0 or complete['models_fitted']!=0 or complete['qualifying_labels_read'] is not False or
        complete['actual_models']!=actual or plan['actual_models']!=actual or not plan['matched_B_both_queries']):
        raise ValueError('Exact frozen original fold-bag definition and complete inference required')
    path=production.RUN/'nn-bag.f32'
    if {p.name for p in production.RUN.iterdir()}!={'nn-bag.f32'} or path.stat().st_size!=2817131*4:
        raise ValueError('Exactly one complete streamed qualifying bag required')
    if study.data.ops.sha(path)!=complete['prediction_sha256']:raise ValueError('Original stored bag changed')
    study.mps.MODEL_FACTORY=study.model_factory
    trained,report=study.paths('h128');pipelines=[];seed_hashes=[];reopened=0
    for fold in range(5):
        root=trained/'nn'/str(fold);receipt=u.read(report/'nn'/('%s.json'%fold))
        with np.load(root/'ridge.npz') as saved:w=saved['full'].copy()
        with np.load(root/'scalers.npz') as saved:s={k:saved['full_'+k].copy() for k in ['lo','hi','mean','std']}
        models=[];hashes=[]
        for seed,count in zip(m['config']['seeds'],receipt['chosen_iterations']):
            if count==0:models.append(None);hashes.append(None);continue
            model_path=root/('nn-%s.pt'%seed);checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
            if checkpoint['config']!=m['config'] or checkpoint['feature_width']!=m['nn_inputs'] or checkpoint['seed']!=seed+fold or checkpoint['epochs']!=count:
                raise ValueError('Every actual original selected model metadata must match')
            models.append(study.mps.load_model(model_path,m['config']));hashes.append(hashlib.sha256());reopened+=1
        pipelines.append((w,s,models));seed_hashes.append(hashes)
    if reopened!=actual:raise ValueError('All actual nonzero original models must reopen')
    parent_meta=u.read(study.data.parent.REPORT/'manifest.json');parent=independent.legacy.Independent('qual',parent_meta);stored=np.memmap(path,mode='r',dtype='<f4',shape=(2817131,))
    ops=[hashlib.sha256(),hashlib.sha256()];normalized=[hashlib.sha256() for _ in range(5)];fold_hashes=[hashlib.sha256() for _ in range(5)]
    try:
        for lo in range(0,2817131,8192):
            hi=min(lo+8192,2817131)
            parent_z,parent_f=parent.block(lo,hi)
            z,f=independent.extend(parent_z,parent_f,data['retained'],'qual',lo,hi)
            ops[0].update(z.tobytes());ops[1].update(f.tobytes());total=np.zeros(hi-lo,dtype=np.float64)
            for fold,(w,s,models) in enumerate(pipelines):
                baseline=np.clip(z@w,1.05,4.95);x=np.empty((hi-lo,m['nn_inputs']),dtype=np.float32)
                x[:,:-1]=f;x[:,-1]=baseline
                x=np.asarray((np.clip(x,s['lo'],s['hi'])-s['mean'])/s['std'],dtype=np.float32,order='C')
                normalized[fold].update(x.tobytes());correction=np.zeros(hi-lo,dtype=np.float64)
                for slot,model in enumerate(models):
                    if model is None:continue
                    raw=study.mps.predict(model,x);seed_hashes[fold][slot].update(raw.tobytes())
                    divided=np.divide(raw,np.float32(3),dtype=np.float32)
                    correction=np.add(correction,divided.astype(np.float64))
                folded=np.asarray(np.clip(baseline+correction,1.05,4.95),dtype='<f4')
                fold_hashes[fold].update(folded.tobytes());total=np.add(total,folded.astype(np.float64))
            expected=np.asarray(total/5,dtype='<f4')
            if not np.isfinite(expected).all() or not np.array_equal(expected.view('u4'),stored[lo:hi].view('u4')):
                raise ValueError('Every actual complete qualifying bag bit must replay')
            u.write(production.REPORT/'replay-status.json',dict(state='all_five_actual_pipelines_replayed',complete_rows=hi))
    finally:parent.close();del pipelines;torch.mps.empty_cache()
    seeds=[[h.hexdigest() if h is not None else None for h in row] for row in seed_hashes]
    streams=[h.hexdigest() for h in ops]
    if (streams!=data['virtual_c_order_sha256']['qual'] or streams!=complete['operators_sha256'] or
        [h.hexdigest() for h in normalized]!=complete['normalized_sha256'] or
        [h.hexdigest() for h in fold_hashes]!=complete['fold_sha256'] or seeds!=complete['seed_sha256']):
        raise ValueError('Every full operator/normalization/seed/fold stream must independently match')
    study.data.verify_bindings(bindings)
    if study.data.ops.sha(path)!=complete['prediction_sha256']:raise ValueError('Actual bag changed during replay')
    u.write(output,dict(state='complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag',
        plan_sha256=study.data.ops.sha(production.REPORT/'plan.json'),inference_completion_sha256=study.data.ops.sha(production.REPORT/'inference-completion.json'),
        verifier_sha256=study.data.ops.sha(Path(__file__)),source_sha256=bindings,actual_models_reopened=reopened,
        seed_slots=15,rows=2817131,all_original_model_and_input_streams_replayed=True,
        uniform_five_fold_bag_bitwise_equal=True,prediction_sha256=study.data.ops.sha(path),
        matched_B_raw_operators_on_both_queries=True,known_OOF_and_qualifying_bag_constructions_distinct=True,
        qualifying_labels_read=False,models_fitted=0,qualifying_score_claimed=False))
    print('Every matched391 qualifying input/model/seed/fold/bag bit independently replayed',flush=True)


if __name__=='__main__':main()
