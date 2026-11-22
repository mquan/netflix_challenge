"""Independent whole qualifying-bag replay from actual saved fold pipelines."""
from contextlib import ExitStack
import gc
import hashlib
from pathlib import Path
import numpy as np
import upstream_rbm400_bag as bag
import upstream_rbm400_neural_study as study
import verify_upstream_rbm400_neural_study as models

STATE='complete_independent_selected_RBM400_neural_full_qualifying_model_operator_and_vector_replay'


def verify(guard):
    output=bag.REPORT/'independent-verification.json'
    if output.exists():raise ValueError('Original complete independent bag proof retained')
    source=bag.sources(guard);torch=study.runtime();manifest=source['manifest']
    plan_path=bag.REPORT/'plan.json';done_path=bag.REPORT/'inference-completion.json'
    plan,done=[bag.execution.read(p) for p in (plan_path,done_path)]
    if (plan['source_sha256']!=source['source_sha256'] or plan['arm']!=source['arm']
            or plan['rows']!=bag.ROWS or plan['chunk_rows']!=8192 or plan['roles']!=source['roles']
            or plan['selection_sha256']!=source['selection_sha256'] or plan['config']!=manifest['config']
            or plan['models_fitted']!=0 or plan['qualifying_labels_read'] is not False
            or done['state']!=bag.STATE or done['plan_sha256']!=bag.execution.inputs.sha(plan_path)
            or done['arm']!=source['arm'] or done['rows']!=bag.ROWS or done['models_fitted']!=0
            or done['qualifying_labels_read'] is not False or done['actual_models_reopened']!=source['actual_models']
            or done['known_NN_provider']!=source['roles']['known_NN_provider']
            or done['known_fixed_half_control']!=source['roles']['known_fixed_half_control']):
        raise ValueError('Complete original selected-arm inference and NN/control roles required')
    bindings={str(p.relative_to(bag.ROOT)):bag.execution.inputs.sha(p) for p in (plan_path,done_path)}
    for field,name in [('NN','NN-bag.npy'),('fixed_half','fixed-half-tree154.npy')]:
        row=done[field];p=Path(row['path'])
        if p!=bag.ART/name or p.resolve()!=p or p.stat().st_size!=row['bytes']:raise ValueError('Exact complete original inference payload required')
        study.inputs.context.bind(bindings,p,row['sha256'])
    records=study.inputs.context.provider.original.verify_bindings(bindings,guard)
    relative=lambda p:str(p.relative_to(bag.ROOT));width=manifest['dimensions']['nn_inputs'];d=manifest['dimensions']['design_columns']
    operators=[hashlib.sha256(),hashlib.sha256()]
    raw=[hashlib.sha256() for _ in range(5)];normalized=[hashlib.sha256() for _ in range(5)];folds=[hashlib.sha256() for _ in range(5)]
    seeds=[[hashlib.sha256() if count else None for count in r['chosen_iterations']] for r in source['receipts']]
    model_count=0;seen=0
    try:
        with ExitStack() as stack:
            nn=stack.enter_context(study.cache.mapped(bag.ART/'NN-bag.npy',records[relative(bag.ART/'NN-bag.npy')],'<f4',(bag.ROWS,),guard))
            half=stack.enter_context(study.cache.mapped(bag.ART/'fixed-half-tree154.npy',records[relative(bag.ART/'fixed-half-tree154.npy')],'<f4',(bag.ROWS,),guard))
            tree=stack.enter_context(bag.raw_vector(bag.ROOT/source['tree_qualifying'],source['records'][source['tree_qualifying']],bag.ROWS,guard))
            run,_=study.paths(source['arm']);pipelines=[]
            for fold,receipt in enumerate(source['receipts']):
                root=run/'nn'/str(fold);p=root/'full-weights.npy'
                weights=stack.enter_context(study.cache.mapped(p,source['records'][relative(p)],'<f8',(d,),guard));scaler={}
                for key in ('lo','hi','mean','std'):
                    p=root/('full-'+key+'.npy');scaler[key]=stack.enter_context(study.cache.mapped(p,source['records'][relative(p)],'<f8',(width,),guard))
                actual=[]
                for seed,count in zip(manifest['config']['seeds'],receipt['chosen_iterations']):
                    if not count:actual.append(None);continue
                    p=root/('nn-%s.pt'%seed)
                    actual.append(stack.enter_context(models.actual_model(p,source['records'][relative(p)],manifest,fold,seed,count,guard)));model_count+=1
                pipelines.append((weights,scaler,actual))
            with study.inputs.context.open_query('qual',source['declaration'],source['records'],guard,
                    retain_scalar=source['data_manifest']['alias']['retain_scalar'],latent=source['arm']=='latent') as reader:
                if len(reader.prediction)!=bag.ROWS:raise ValueError('Every canonical qualifying row required')
                for start in range(0,bag.ROWS,bag.CHUNK):
                    guard();stop=min(start+bag.CHUNK,bag.ROWS);z,f=reader.block(start,stop)
                    operators[0].update(z.tobytes());operators[1].update(f.tobytes());total=np.zeros(stop-start,dtype=np.float64)
                    for fold,(weights,scaler,actual) in enumerate(pipelines):
                        baseline=np.clip(np.matmul(z,weights),1.05,4.95)
                        values=np.empty((stop-start,width),dtype=np.float32);values[:,:-1]=f;values[:,-1]=baseline
                        raw[fold].update(values.tobytes())
                        values[:]=np.asarray((np.clip(values,scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std'],dtype=np.float32,order='C')
                        normalized[fold].update(values.tobytes());correction=np.zeros(stop-start,dtype=np.float64)
                        for slot,model in enumerate(actual):
                            if model is None:continue
                            predicted=study.backend.predict(model,values)
                            if predicted.dtype!=np.float32 or predicted.shape!=baseline.shape or not np.isfinite(predicted).all():
                                raise ValueError('Every actual selected-model query prediction required')
                            seeds[fold][slot].update(predicted.tobytes())
                            correction+=np.divide(predicted,np.float32(3))
                        folded=np.minimum(np.maximum(baseline+correction,1.05),4.95).astype('<f4')
                        folds[fold].update(folded.tobytes());total+=folded.astype(np.float64)
                    expected=(total/5).astype('<f4')
                    if expected.tobytes()!=nn[start:stop].tobytes():raise ValueError('Every independently reconstructed NN-only bag bit required')
                    if not np.isfinite(tree[start:stop]).all():raise ValueError('Complete finite original qualifying tree control required')
                    control=np.clip((expected.astype(np.float64)+tree[start:stop].astype(np.float64))/2,1.05,4.95).astype('<f4')
                    if control.tobytes()!=half[start:stop].tobytes():raise ValueError('Every independent fixed-half control bit required')
                    seen+=stop-start
            if seen!=bag.ROWS or model_count!=source['actual_models']:raise ValueError('All actual selected models and full query population required')
            expected=source['data_manifest']['virtual_c_order_sha256']['qual']['arm_operator_sha256'][source['arm']]
            computed=dict(operators_sha256=[h.hexdigest() for h in operators],raw_neural_input_sha256=[h.hexdigest() for h in raw],
                normalized_sha256=[h.hexdigest() for h in normalized],fold_sha256=[h.hexdigest() for h in folds],
                seed_sha256=[[h.hexdigest() if h else None for h in row] for row in seeds])
            if computed['operators_sha256']!=expected or any(computed[name]!=done[name] for name in computed):
                raise ValueError('Every complete operator/raw/normalized/seed/fold stream hash must independently match')
        result=dict(state=STATE,arm=source['arm'],rows=bag.ROWS,inference_sha256=bag.execution.inputs.sha(done_path),
            plan_sha256=bag.execution.inputs.sha(plan_path),source_sha256=source['source_sha256'],
            actual_models_reopened=model_count,all_five_folds_and_three_seed_slots=True,
            every_full_query_operator_raw_normalized_seed_fold_and_output_bit_verified=True,
            actual_MPS_predictions_replayed=True,provider_is_NN_only_on_both_queries=True,
            known_NN_provider=source['roles']['known_NN_provider'],qualifying_NN_provider=done['NN'],
            known_fixed_half_control=source['roles']['known_fixed_half_control'],qualifying_fixed_half_control=done['fixed_half'],
            models_fitted=0,qualifying_labels_read=False,remaining='All17 original families known-only calibration/freeze/preaudit and fixed recovered audits')
    finally:
        try:
            if study.inputs.context.provider.original.verify_bindings(source['source_sha256'],guard)!=source['records']:
                raise ValueError('Original selected model/source identities changed during qualifying replay')
        finally:
            try:
                if study.inputs.context.provider.original.verify_bindings(bindings,guard)!=records:
                    raise ValueError('Original inference outputs changed during complete independent query replay')
            finally:gc.collect();torch.mps.empty_cache()
    study.write_json(output,result,guard);return result
