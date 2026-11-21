#!/usr/bin/env python3
"""Full original-budget H128 neural study on every complete own trajectory."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_weekend_trajectory_data as data
import upstream_own391_paired_mps as established
import upstream_exact_weekend_scaler as exact_stats
import upstream_own352_temporal_mps as previous

u,n,mps,scaling=data.u,previous.n,previous.mps,previous.scaling
CONTEXT=list(established.CONTEXT)
CONFIGS={'h128':dict(established.CONFIGS['h128'],architecture='context-weighted-residual-weekend-trajectory-h128-v1')}


def signal_indices(width):
    if not 618<=width<=813:raise ValueError('Complete dynamically expanded input width required')
    return list(established.SIGNALS[:-1])+list(range(617,width-1))+[width-1]


def dimensions():
    manifest=u.read(data.REPORT/'manifest.json');width=manifest['nn_inputs'];added=width-618
    if (not 0<=added<=195 or manifest['raw_predictions']!=391+added or
        manifest['design_columns']!=467+added or manifest['nonlinear_columns']!=617+added):
        raise ValueError('Mechanically determined complete input widths required')
    return width


def paths(variant='h128'):
    if variant!='h128':raise ValueError('Only the previously selected original H128 arm')
    return u.ROOT/'artifacts/upstream-weekend-trajectory-h128-mps',u.ROOT/'experiments/upstream-weekend-trajectory-h128-mps'


def model_factory(width,cfg):
    import torch
    signals=signal_indices(width)
    if cfg not in CONFIGS.values() or sorted(signals+CONTEXT)!=list(range(width)):
        raise ValueError('Complete dynamic-width original H128 recipe required')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(signals));self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(signals),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals));weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def hashes():
    files=[Path(__file__),Path(data.__file__),Path(exact_stats.__file__),Path(previous.__file__),Path(mps.__file__),
        Path(scaling.__file__),Path(n.__file__),Path(established.__file__),Path(data.ops.__file__),u.ROOT/'scripts/verify_upstream_weekend_trajectory_mps.py',data.DECL]
    files.append(u.ROOT/'scripts/verify_upstream_weekend_trajectory_canonical.py')
    return {str(p.relative_to(u.ROOT)):data.ops.sha(p) for p in files}


def bind_local_reference(bindings):
    root=u.ROOT/'experiments/upstream-complete-trajectory-h128-mps'
    trained=u.ROOT/'artifacts/upstream-complete-trajectory-h128-mps'
    mp,vp,cp=[root/name for name in ['manifest.json','independent-verification.json','independent-canonical-verification.json']]
    manifest,actual,canonical=map(u.read,[mp,vp,cp])
    if (canonical['state']!='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores' or
        canonical['manifest_sha256']!=data.ops.sha(mp) or canonical['complete_actual_model_replay_sha256']!=data.ops.sha(vp) or
        actual['state']!='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions' or
        actual['manifest_sha256']!=data.ops.sha(mp) or canonical['source_sha256']!=manifest['source_sha256'] or
        any(record['qualifying_labels_read'] is not False for record in [manifest,actual,canonical]) or
        not canonical['all_five_original_folds_verified'] or not canonical['every_nn_and_fixed_half_float32_bit_equal'] or
        not actual['complete_original_statistics_bitwise_equal'] or not actual['all_original_models_preserved_without_refitting'] or
        not actual['matched_B_known_and_qualifying'] or not canonical['neural_inputs_matched_B_on_both_queries'] or
        actual['fixed_half_sha256']!=canonical['prediction_sha256'] or
        actual['fixed_half_tree154_rmse']!=canonical['fixed_half_tree154_rmse'] or actual['nn_rmse']!=canonical['nn_rmse'] or
        [r['fold'] for r in actual['records']]!=list(range(5)) or
        not all(r['both_complete_eligible_scalers_independently_recomputed'] and r['all_normalized_input_bits_equal'] and
                r['actual_saved_pipeline_bitwise_equal'] for r in actual['records']) or
        not math.isfinite(canonical['fixed_half_tree154_rmse'])):
        raise ValueError('Full preceding local trajectory known-only reference required')
    data.ops.merge_bindings(bindings,manifest['source_sha256'])
    data.ops.merge_bindings(bindings,manifest['code_sha256'])
    expected={str(p.relative_to(u.ROOT)):data.ops.sha(p) for p in [mp,vp,cp]}
    expected[str((trained/'nn.f32').relative_to(u.ROOT))]=actual['nn_sha256']
    expected[str((trained/'fixed-half-tree154.f32').relative_to(u.ROOT))]=canonical['prediction_sha256']
    models=0
    for record in actual['records']:
        path=u.ROOT/record['receipt_path'];receipt=u.read(path);counts=receipt['chosen_iterations']
        if (data.ops.sha(path)!=record['receipt_sha256'] or receipt['manifest_sha256']!=data.ops.sha(mp) or
            receipt['outputs']!=record['outputs'] or len(counts)!=3 or
            any(type(count)!=int or not 0<=count<=40 for count in counts) or
            sum(count>0 for count in counts)!=record['actual_models_reopened']):
            raise ValueError('Every actual preceding model/zero-budget slot must retain its original expectation')
        expected[record['receipt_path']]=record['receipt_sha256'];models+=record['actual_models_reopened']
        for name,digest in record['outputs'].items():
            expected[str((trained/'nn'/str(record['fold'])/name).relative_to(u.ROOT))]=digest
    if models!=canonical['actual_models_rehashed']:raise ValueError('All actual preceding models must be accounted for')
    data.ops.merge_bindings(bindings,expected)
    return dict(path=str(cp.relative_to(u.ROOT)),sha256=data.ops.sha(cp),
        fixed_half_rmse=canonical['fixed_half_tree154_rmse'],actual_models=models)


def prepare(variant):
    run,report=paths(variant)
    if run.exists() or report.exists():raise ValueError('Original complete or partial paired neural arm retained')
    mp=data.REPORT/'manifest.json';vp=data.REPORT/'independent-verification.json';m=u.read(mp);v=u.read(vp)
    if (v['state']!='complete_independent_all_trajectory_operators_statistics_ridge_and_stored_predictions' or
        v['manifest_sha256']!=data.ops.sha(mp) or not v['matched_B_known_and_qualifying'] or
        not v['every_full_operator_statistic_solve_and_prediction_bit_equal'] or v['qualifying_labels_read'] is not False or
        v['neural_models_fitted']!=0 or v['source_sha256']!=m['source_sha256'] or v['nn_inputs']!=dimensions()):
        raise ValueError('Full complete-trajectory data/scalar/ridge replay required')
    bindings=dict(m['source_sha256'])
    reference_path=u.ROOT/'experiments/upstream-own391-paired-h128-mps/independent-canonical-verification.json'
    reference=u.read(reference_path)
    if (data.ops.sha(reference_path)!='ce7ae633d328f8e0a79055146b0cb15efa5af18bf6a4760f328640557fbe79e7' or
        reference['state']!='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores'):
        raise ValueError('Original full canonical development reference required')
    data.ops.merge_bindings(bindings,{str(reference_path.relative_to(u.ROOT)):data.ops.sha(reference_path)})
    local_reference=bind_local_reference(bindings)
    tree=next(row for row in u.read(previous.REPORT/'manifest.json')['controls'] if row['id']=='tree154')
    if tree['prediction'] not in bindings or data.ops.sha(u.ROOT/tree['prediction'])!=bindings[tree['prediction']]:
        raise ValueError('Original fully replayed tree154 OOF control required')
    for collection in [hashes(),{str(mp.relative_to(u.ROOT)):data.ops.sha(mp),str(vp.relative_to(u.ROOT)):data.ops.sha(vp)}]:
        for key,value in collection.items():
            if key in bindings and bindings[key]!=value:raise ValueError('Original matched neural source expectation conflicts')
            bindings[key]=value
    data.ops.merge_bindings(bindings,{str((data.RUN/name).relative_to(u.ROOT)):digest for name,digest in m['prepared_sha256'].items()})
    data.verify_bindings(bindings);run.mkdir();report.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','stats.npz']:u.link_or_copy(data.RUN/name,run/name)
    u.write(report/'manifest.json',dict(state='declared_original_budget_complete_trajectory_neural_arm',variant=variant,
        config=CONFIGS[variant],versions=n.versions(),code_sha256=hashes(),source_sha256=bindings,
        input_manifest_sha256=data.ops.sha(mp),raw_predictions=m['raw_predictions'],design_columns=m['design_columns'],nonlinear_columns=m['nonlinear_columns'],nn_inputs=dimensions(),
        signal_indices=signal_indices(dimensions()),context_indices=CONTEXT,matched_B_known_and_qualifying=True,
        controls=[tree],declared_global_reference_rmse=min(reference['fixed_half_tree154_rmse'],local_reference['fixed_half_rmse']),
        declared_reference_scores=dict(original391=reference['fixed_half_tree154_rmse'],preceding_local_trajectory=local_reference['fixed_half_rmse']),
        preceding_local_trajectory_reference=local_reference,fixed_tree_weight=.5,
        qualifying_labels_read=False,rule='All391 original scalars plus EVERY novel complete CF/native pair; all90 learned contexts and225 context positions. Original5userfolds/3seeds/max40pat4/batch8192/AdamW. Exact full eligible statistics with bounded scratch. No nonlinear OOF component inputs. Known-only architecture selection before any new qualifying candidate.'))


def inputs():
    m=u.read(data.REPORT/'manifest.json')
    parent=u.read(data.parent.REPORT/'manifest.json')
    return data.Inputs('probex',parent,m['retained'])


def originals(inner_weights,full_weights):
    reader=inputs();inner=np.empty((1128164,dimensions()),dtype=np.float32);full=np.empty_like(inner)
    ib=np.empty(1128164);fb=np.empty(1128164);digests=[hashlib.sha256(),hashlib.sha256()]
    try:
        for lo in range(0,len(inner),8192):
            hi=min(lo+8192,len(inner));z,f=reader.block(lo,hi);digests[0].update(z.tobytes());digests[1].update(f.tobytes())
            ib[lo:hi]=np.clip(z@inner_weights,1.05,4.95);fb[lo:hi]=np.clip(z@full_weights,1.05,4.95)
            inner[lo:hi,:-1]=f;full[lo:hi,:-1]=f;inner[lo:hi,-1]=ib[lo:hi];full[lo:hi,-1]=fb[lo:hi]
    finally:reader.close()
    if [h.hexdigest() for h in digests]!=u.read(data.REPORT/'manifest.json')['virtual_c_order_sha256']['probex']:
        raise ValueError('Every complete expanded paired input bit must match')
    return inner,full,ib,fb


def worker(variant):
    import torch
    run,report=paths(variant);cfg=CONFIGS[variant];manifest=u.read(report/'manifest.json');mps.MODEL_FACTORY=model_factory
    if manifest['config']!=cfg or manifest['code_sha256']!=hashes() or manifest['versions']!=n.versions():raise ValueError('Original matched recipe/runtime changed')
    if (report/'fit-completion.json').exists():raise ValueError('Existing original complete fit retained')
    data.verify_bindings(manifest['source_sha256']);y=np.load(run/'labels.npy');groups=np.load(run/'groups.npy')
    with np.load(run/'stats.npz') as saved:grams,rhs=saved['gram'],saved['rhs']
    for fold in range(5):
        root=run/'nn'/str(fold);root.mkdir(parents=True,exist_ok=False)
        train=groups//2!=fold;held=~train;fit=train&(groups%2==0);valid=train&(groups%2==1);eligible=np.arange(10)//2!=fold
        iw=n.ridge_weights(grams,rhs,eligible&(np.arange(10)%2==0),cfg['ridge_lambda']);fw=n.ridge_weights(grams,rhs,eligible,cfg['ridge_lambda'])
        inner,full,ib,fb=originals(iw,fw)
        si=exact_stats.fit_scaler(inner,fit);scaling.apply_scaler(inner,si,output=inner)
        sf=exact_stats.fit_scaler(full,train);scaling.apply_scaler(full,sf,output=full)
        np.savez(root/'ridge.npz',inner=iw,full=fw);np.savez(root/'scalers.npz',**{'inner_'+k:v for k,v in si.items()},**{'full_'+k:v for k,v in sf.items()})
        correction=np.zeros(int(held.sum()));counts=[]
        for seed in cfg['seeds']:
            actual=seed+fold;n.event(stage='select',variant=variant,fold=fold,seed=seed)
            selected,count=mps.train_nn(inner[fit],y[fit]-ib[fit],cfg,actual,cfg['nn_epochs'],valid=(inner[valid],y[valid]-ib[valid]))
            del selected;torch.mps.empty_cache()
            if count:
                model,_=mps.train_nn(full[train],y[train]-fb[train],cfg,actual,count)
                correction+=mps.predict(model,full[held])/3
                torch.save(dict(state=model.state_dict(),config=cfg,feature_width=dimensions(),seed=actual,epochs=count),root/('nn-%s.pt'%seed))
                del model;torch.mps.empty_cache()
            counts.append(count);n.event(stage='refit_complete',variant=variant,fold=fold,seed=seed,iterations=count)
        prediction=np.clip(fb[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
        u.write(report/'nn'/('%s.json'%fold),dict(fold=fold,canonical_rows=len(prediction),chosen_iterations=counts,
            manifest_sha256=data.ops.sha(report/'manifest.json'),outputs={p.name:data.ops.sha(p) for p in root.iterdir()},rmse=data.parent.score(prediction,y[held])))
        del inner,full,ib,fb;gc.collect();torch.mps.empty_cache()
    data.verify_bindings(manifest['source_sha256']);u.write(report/'fit-completion.json',dict(state='complete_all_five_original_budget_complete_trajectory_neural_fits',variant=variant,qualifying_labels_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=CONFIGS,default='h128');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker(args.variant) if args.worker else prepare(args.variant)
