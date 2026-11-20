#!/usr/bin/env python3
"""Stream all original complete-trajectory fold pipelines into one qualifying bag."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import datetime as dt
import hashlib
from pathlib import Path
import numpy as np
import torch
import upstream_complete_trajectory_mps as study

u=study.u
RUN=u.ROOT/'artifacts/upstream-complete-trajectory-fold-bag'
REPORT=u.ROOT/'experiments/upstream-complete-trajectory-fold-bag'
DECL=u.ROOT/'OWN_COMPLETE_TRAJECTORY_FOLD_BAG_PLAN.md'
ROWS,CHUNK=2817131,8192


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def fold_values(baseline,slots):
    if len(slots)!=3:raise ValueError('All three original seed slots required')
    correction=np.zeros(len(baseline),dtype=np.float64)
    for prediction in slots:
        if prediction is None:continue
        if prediction.dtype!=np.dtype('float32') or prediction.shape!=baseline.shape or not np.isfinite(prediction).all():
            raise ValueError('Complete finite float32 seed prediction required')
        correction+=prediction/3
    return np.clip(baseline+correction,1.05,4.95).astype('<f4')


def sources():
    trained,report=study.paths('h128');mp=report/'manifest.json';vp=report/'independent-verification.json';cp=report/'independent-canonical-verification.json'
    m,v,c=map(u.read,[mp,vp,cp]);data=u.read(study.data.REPORT/'manifest.json')
    if (c['state']!='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores' or
        v['state']!='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions' or
        c['manifest_sha256']!=study.data.ops.sha(mp) or v['manifest_sha256']!=study.data.ops.sha(mp) or
        c['complete_actual_model_replay_sha256']!=study.data.ops.sha(vp) or c['source_sha256']!=m['source_sha256'] or
        c['nn_rmse']!=v['nn_rmse'] or c['fixed_half_tree154_rmse']!=v['fixed_half_tree154_rmse'] or
        c['declared_global_reference_rmse']!=m['declared_global_reference_rmse'] or
        study.data.ops.sha(trained/'nn.f32')!=v['nn_sha256'] or
        study.data.ops.sha(trained/'fixed-half-tree154.f32')!=v['fixed_half_sha256'] or
        c['prediction_sha256']!=v['fixed_half_sha256'] or
        m['config']!=study.CONFIGS['h128'] or m['code_sha256']!=study.hashes() or m['versions']!=study.n.versions() or
        not c['neural_inputs_matched_B_on_both_queries'] or c['qualifying_labels_read'] is not False or
        not v['complete_original_statistics_bitwise_equal'] or not v['all_original_models_preserved_without_refitting'] or
        [r['fold'] for r in v['records']]!=list(range(5)) or
        not all(r['both_complete_eligible_scalers_independently_recomputed'] and r['all_normalized_input_bits_equal'] and
            r['actual_saved_pipeline_bitwise_equal'] for r in v['records'])):
        raise ValueError('Complete verified original H128 models/statistics/canonical score required')
    bindings=dict(m['source_sha256'])
    def bind(path,digest=None):
        key=str(path.relative_to(u.ROOT));value=study.data.ops.sha(path) if digest is None else digest
        if key in bindings and bindings[key]!=value:raise ValueError('Original fitted source expectation conflicts')
        bindings[key]=value
    actual=0
    for row in v['records']:
        path=u.ROOT/row['receipt_path'];receipt=u.read(path);counts=receipt['chosen_iterations']
        if (study.data.ops.sha(path)!=row['receipt_sha256'] or receipt['outputs']!=row['outputs'] or
            receipt['manifest_sha256']!=study.data.ops.sha(mp) or receipt['fold']!=row['fold'] or len(counts)!=3 or
            any(type(count)!=int or not 0<=count<=40 for count in counts) or
            row['actual_models_reopened']!=sum(count>0 for count in counts)):
            raise ValueError('Every original selected budget and saved pipeline required')
        bind(path,row['receipt_sha256']);actual+=row['actual_models_reopened']
        folder=trained/'nn'/str(row['fold'])
        expected={'ridge.npz','scalers.npz','prediction.f32'}|{'nn-%s.pt'%seed for seed,count in zip(m['config']['seeds'],counts) if count}
        if set(receipt['outputs'])!=expected or {p.name for p in folder.iterdir()}!=expected:
            raise ValueError('Complete actual pipeline files and zero-budget slots required')
        for name,digest in receipt['outputs'].items():bind(folder/name,digest)
    if actual!=c['actual_models_rehashed']:raise ValueError('All actual original models must be accounted for')
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_complete_trajectory_fold_bag.py',DECL,mp,vp,cp,
        study.data.REPORT/'manifest.json',study.data.REPORT/'independent-verification.json']:
        bind(path)
    for name in ['nn.f32','fixed-half-tree154.f32']:bind(trained/name)
    qual_path=u.FRESH/'data/qual/ratings.npy'
    native_plan=u.read(u.ROOT/'experiments/upstream-original1200/execution-plan.json')
    qual_key=str(qual_path.relative_to(u.ROOT))
    if qual_key not in native_plan['bindings']:raise ValueError('Original admitted zero-target binding required')
    bind(qual_path,native_plan['bindings'][qual_key])
    study.data.verify_bindings(bindings)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying fitting targets must be zero')
    return m,data,bindings,actual


def load_pipelines(m):
    trained,report=study.paths('h128');study.mps.MODEL_FACTORY=study.model_factory;result=[]
    for fold in range(5):
        root=trained/'nn'/str(fold);receipt=u.read(report/'nn'/('%s.json'%fold))
        with np.load(root/'ridge.npz') as a:weights=a['full'].copy()
        with np.load(root/'scalers.npz') as a:scaler={key:a['full_'+key].copy() for key in ['lo','hi','mean','std']}
        if weights.shape!=(m['design_columns'],) or not np.isfinite(weights).all() or any(a.shape!=(m['nn_inputs'],) or not np.isfinite(a).all() for a in scaler.values()):
            raise ValueError('Original full complete ridge and scaler required')
        models=[]
        for seed,count in zip(m['config']['seeds'],receipt['chosen_iterations']):
            if not count:models.append(None);continue
            path=root/('nn-%s.pt'%seed);saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['feature_width']!=m['nn_inputs'] or saved['seed']!=seed+fold or saved['epochs']!=count or saved['config']!=m['config']:
                raise ValueError('Actual original model seed/width/epoch/config differs')
            models.append(study.mps.load_model(path,m['config']))
        result.append((weights,scaler,models))
    return result


def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing matched391 qualifying inference retained')
    m,data,bindings,actual=sources();RUN.mkdir();REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,created=stamp(),rows=ROWS,chunk_rows=CHUNK,
        config=m['config'],folds=list(range(5)),seed_slots=m['config']['seeds'],actual_models=actual,
        qualifying_labels_read=False,models_fitted=0,matched_B_both_queries=True))
    pipelines=load_pipelines(m);parent=u.read(study.data.parent.REPORT/'manifest.json');inputs=study.data.Inputs('qual',parent,data['retained'])
    operators=[hashlib.sha256(),hashlib.sha256()]
    normalized=[hashlib.sha256() for _ in range(5)];folds=[hashlib.sha256() for _ in range(5)]
    seeds=[[hashlib.sha256() if model is not None else None for model in pipeline[2]] for pipeline in pipelines]
    try:
        with (RUN/'nn-bag.f32').open('xb') as output:
            for lo in range(0,ROWS,CHUNK):
                hi=min(lo+CHUNK,ROWS);z,f=inputs.block(lo,hi)
                operators[0].update(z.tobytes());operators[1].update(f.tobytes());bag=np.zeros(hi-lo,dtype=np.float64)
                for fold,(weights,scaler,models) in enumerate(pipelines):
                    baseline=np.clip(z@weights,1.05,4.95);x=study.scaling.append_baseline(f,baseline)
                    study.scaling.apply_scaler(x,scaler,output=x);normalized[fold].update(x.tobytes())
                    slots=[]
                    for slot,model in enumerate(models):
                        if model is None:slots.append(None);continue
                        prediction=study.mps.predict(model,x)
                        seeds[fold][slot].update(prediction.tobytes());slots.append(prediction)
                    values=fold_values(baseline,slots);folds[fold].update(values.tobytes());bag+=values.astype(np.float64)
                    if not np.isfinite(values).all():raise ValueError('Finite complete fold predictions required')
                (bag/5).astype('<f4').tofile(output)
                u.write(REPORT/'status.json',dict(state='all_original_five_fold_pipelines_streamed',complete_rows=hi,total_rows=ROWS,observed=stamp()))
    finally:inputs.close();del pipelines;torch.mps.empty_cache()
    if [h.hexdigest() for h in operators]!=data['virtual_c_order_sha256']['qual']:
        raise ValueError('Every complete original qualifying operator bit must match')
    study.data.verify_bindings(bindings)
    if (RUN/'nn-bag.f32').stat().st_size!=ROWS*4:raise ValueError('Complete original query population required')
    u.write(REPORT/'inference-completion.json',dict(state='complete_all_trajectory_streamed_qualifying_bag_pending_independent_replay',
        plan_sha256=study.data.ops.sha(REPORT/'plan.json'),operators_sha256=[h.hexdigest() for h in operators],
        normalized_sha256=[h.hexdigest() for h in normalized],fold_sha256=[h.hexdigest() for h in folds],
        seed_sha256=[[h.hexdigest() if h else None for h in row] for row in seeds],
        prediction_sha256=study.data.ops.sha(RUN/'nn-bag.f32'),actual_models=actual,models_fitted=0,qualifying_labels_read=False))
    print('Complete matched391 qualifying bag; independent actual-model replay remains',flush=True)


if __name__=='__main__':main()
