"""Complete selected-arm qualifying bag; no fitting, scoring, CLI or owner.

The calibrated NN provider is the NN-only known vector paired with this NN-only
bag. The fixed-half vectors are explicitly separate ranking/audit controls.
"""
from contextlib import contextmanager,ExitStack
import gc
import hashlib
import os
from pathlib import Path
import numpy as np
import upstream_rbm400_neural_study_reference_v2 as study
import upstream_rbm400_neural_canonical_reference_v2 as canonical
import verify_upstream_rbm400_neural_study_reference_v2 as replay
import upstream_rbm400_neural_execution_reference_v2 as execution
import upstream_rbm400_neural_scaler as scaling
import upstream_factored_rbm_admission as admission

ROOT=study.ROOT
REPORT=ROOT/'experiments/upstream-rbm400-selected-bag-reference-v2'
ART=ROOT/'artifacts/upstream-rbm400-selected-bag-reference-v2'
ROWS=2817131
CHUNK=8192
DECL=ROOT/'OWN_RBM400_SELECTED_BAG_REFERENCE_V2_PLAN.md'
STATE='complete_selected_RBM400_neural_qualifying_bag_pending_independent_full_replay'


def fold_values(baseline,slots):
    if (baseline.ndim!=1 or baseline.dtype!=np.float64 or not np.isfinite(baseline).all()
            or len(slots)!=3):raise ValueError('Finite complete baseline and all three original slots required')
    correction=np.zeros(len(baseline),dtype=np.float64)
    for value in slots:
        if value is None:continue
        if value.dtype!=np.float32 or value.shape!=baseline.shape or not np.isfinite(value).all():
            raise ValueError('Every finite float32 original seed prediction required')
        correction+=value/3
    return np.clip(baseline+correction,1.05,4.95).astype('<f4')


def bag_values(folds):
    if len(folds)!=5 or any(v.dtype!=np.float32 or v.shape!=folds[0].shape or not np.isfinite(v).all() for v in folds):
        raise ValueError('All five complete finite float32 fold predictions required')
    total=np.zeros(len(folds[0]),dtype=np.float64)
    for value in folds:total+=value.astype(np.float64)
    return (total/5).astype('<f4')


def roles(selection,replayed):
    if selection['selected_known_vector']!=replayed['fixed_half']:
        raise ValueError('Selection must bind its declared fixed-half ranking vector')
    if replayed['nn']['path']==replayed['fixed_half']['path']:
        raise ValueError('NN provider and fixed-half control must retain distinct declared identities')
    return dict(known_NN_provider=replayed['nn'],known_fixed_half_control=replayed['fixed_half'],
                known_ranking_vector=selection['selected_known_vector'],
                provider_kind='NN_only',control_kind='fixed_half_NN_tree154')


def sources(guard):
    execution.check_sources(execution.read(execution.REPORT/'plan.json'))
    completion_path=execution.REPORT/'completion.json'
    selected_path=ROOT/'experiments/upstream-own-rbm400-neural-selection-reference-v2/selection.json'
    done,selected=[execution.read(p) for p in (completion_path,selected_path)]
    if (done['state']!='complete_both_original_budget_RBM400_neural_arms_replayed_canonical_and_known_selection'
            or done['verification_sha256']!=execution.inputs.sha(selected_path) or done['qualifying_labels_read'] is not False
            or selected['state']!=canonical.SELECTION_STATE or selected['qualifying_labels_read'] is not False
            or selected['qualifying_labels_used_for_selection'] is not False
            or selected['both_arms_complete_before_selection'] is not True
            or selected['selected_arm']!=canonical.choose(selected['known_fixed_half_scores'])):
        raise ValueError('Both complete actual neural arms/replays/canonical known-only selection required')
    arm=selected['selected_arm']
    manifest,declaration,data_manifest,_,v,receipts,_,_=canonical.evidence(arm,guard)
    vector_roles=roles(selected,v);bindings=dict(selected['source_sha256'])
    for name,digest in declaration['source_sha256'].items():study.inputs.context.bind(bindings,ROOT/name,digest)
    for candidate in study.ARMS:
        path=study.paths(candidate)[1]/'independent-canonical-verification.json'
        key=str(path.relative_to(ROOT))
        if bindings.get(key)!=selected['canonical_sha256'][candidate]:raise ValueError('Both original canonical score expectations required')
    # This is the same fully trained tree provider used by own16. Its known
    # vector must equal the selected study's tree OOF control; no new target
    # or freshly derived expected hash can introduce a different tree.
    previous=study.inputs.context.provider.original
    cpath=previous.REPORT/'calibration.json';c=execution.read(cpath)
    if c['family_ids']!=previous.BASE_IDS+[previous.FAMILY_ID] or len(c['families'])!=16:
        raise ValueError('All original own16 families in their original order required')
    trees=[r for r in c['families'] if r['id']=='tree154']
    if len(trees)!=1 or trees[0]['known_prediction']!=manifest['controls'][0]['prediction']:
        raise ValueError('Original matched known/qualifying tree154 pair required')
    tree_qual=trees[0]['qualifying_prediction']
    if tree_qual not in bindings:raise ValueError('Tree qualifying digest must already be inherited from complete own source closure')
    for p in (completion_path,selected_path,execution.REPORT/'plan.json',cpath,Path(__file__),DECL,
              ROOT/'scripts/verify_upstream_rbm400_bag_reference_v2.py'):study.inputs.context.bind(bindings,p)
    import upstream_rbm400_bag_execution_reference_v2 as owner
    owner.check_sources(owner.read(owner.REPORT/'plan.json'))
    for name in ('plan.json','monitor.json','first-scientific-baseline.json','prerequisites.json',
                 'infer-gate.json','infer-ownership.json'):
        study.inputs.context.bind(bindings,owner.REPORT/name)
    study.inputs.context.bind(bindings,ROOT/'experiments/upstream-rbm400-bag-execution-protocol-reference-v2/verification.json')
    records=previous.verify_bindings(bindings,guard);declaration['source_sha256']=bindings
    return dict(arm=arm,manifest=manifest,declaration=declaration,data_manifest=data_manifest,
        receipts=receipts,source_sha256=bindings,records=records,roles=vector_roles,tree_qualifying=tree_qual,
        actual_models=v['actual_saved_models_reopened'],selection_sha256=execution.inputs.sha(selected_path))


@contextmanager
def raw_vector(path,record,rows,guard):
    guard()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if admission.file_identity(before)!=record['identity'] or before.st_size!=rows*4:
            raise ValueError('Authenticated complete original raw float32 vector required')
        array=np.memmap(stream,dtype='<f4',mode='r',shape=(rows,))
        try:yield array
        finally:
            array._mmap.close();guard()
            if (admission.file_identity(before)!=admission.file_identity(os.fstat(stream.fileno()))
                    or admission.file_identity(before)!=admission.file_identity(path.stat())):
                raise ValueError('Original raw vector changed during query inference')


@contextmanager
def pipelines(source,guard):
    manifest=source['manifest'];width=manifest['dimensions']['nn_inputs'];d=manifest['dimensions']['design_columns']
    run,_=study.paths(source['arm']);relative=lambda p:str(p.relative_to(ROOT))
    with ExitStack() as stack:
        result=[]
        for fold,receipt in enumerate(source['receipts']):
            root=run/'nn'/str(fold)
            weights=stack.enter_context(study.cache.mapped(root/'full-weights.npy',source['records'][relative(root/'full-weights.npy')],'<f8',(d,),guard))
            scaler={}
            for name in ('lo','hi','mean','std'):
                p=root/('full-'+name+'.npy');scaler[name]=stack.enter_context(study.cache.mapped(p,source['records'][relative(p)],'<f8',(width,),guard))
            models=[]
            for seed,count in zip(manifest['config']['seeds'],receipt['chosen_iterations']):
                if not count:models.append(None);continue
                p=root/('nn-%s.pt'%seed)
                models.append(stack.enter_context(replay.actual_model(p,source['records'][relative(p)],manifest,fold,seed,count,guard)))
            result.append((weights,scaler,models))
        if sum(model is not None for row in result for model in row[2])!=source['actual_models']:
            raise ValueError('Every selected actual saved model must reopen exactly once')
        yield result


def infer(guard):
    if REPORT.exists() or ART.exists():raise ValueError('Existing complete/partial qualifying inference preserved')
    source=sources(guard);torch=study.runtime();manifest=source['manifest'];keep=source['data_manifest']['alias']['retain_scalar']
    study.terminal.durable_directory(REPORT,ROOT,guard);study.terminal.durable_directory(ART,ROOT,guard)
    plan=dict(source_sha256=source['source_sha256'],arm=source['arm'],rows=ROWS,chunk_rows=CHUNK,
        roles=source['roles'],selection_sha256=source['selection_sha256'],config=manifest['config'],
        actual_models=source['actual_models'],folds=list(range(5)),seed_slots=manifest['config']['seeds'],
        models_fitted=0,qualifying_labels_read=False)
    study.write_json(REPORT/'plan.json',plan,guard)
    prediction=np.empty(ROWS,dtype='<f4');half=np.empty_like(prediction)
    operators=[hashlib.sha256(),hashlib.sha256()]
    raw=[hashlib.sha256() for _ in range(5)];normalized=[hashlib.sha256() for _ in range(5)]
    folds=[hashlib.sha256() for _ in range(5)]
    seeds=[[hashlib.sha256() if count else None for count in receipt['chosen_iterations']] for receipt in source['receipts']]
    try:
        with pipelines(source,guard) as all_pipelines,raw_vector(ROOT/source['tree_qualifying'],source['records'][source['tree_qualifying']],ROWS,guard) as tree:
            with study.inputs.context.open_query('qual',source['declaration'],source['records'],guard,retain_scalar=keep,latent=source['arm']=='latent') as reader:
                if len(reader.prediction)!=ROWS:raise ValueError('Complete original qualifying query population required')
                for lo in range(0,ROWS,CHUNK):
                    guard();hi=min(lo+CHUNK,ROWS);z,f=reader.block(lo,hi);operators[0].update(z.tobytes());operators[1].update(f.tobytes());values=[]
                    for fold,(weights,scaler,models) in enumerate(all_pipelines):
                        baseline=np.clip(z@weights,1.05,4.95);x=np.empty((hi-lo,manifest['dimensions']['nn_inputs']),dtype='<f4')
                        x[:,:-1]=f;x[:,-1]=baseline;raw[fold].update(x.tobytes())
                        scaling.apply_scaler(x,scaler,guard=guard,output=x);normalized[fold].update(x.tobytes());slots=[]
                        for slot,model in enumerate(models):
                            if model is None:slots.append(None);continue
                            value=study.backend.predict(model,x);seeds[fold][slot].update(value.tobytes());slots.append(value)
                        value=fold_values(baseline,slots);folds[fold].update(value.tobytes());values.append(value)
                    prediction[lo:hi]=bag_values(values)
                    if not np.isfinite(tree[lo:hi]).all():raise ValueError('Finite complete original tree control required')
                    half[lo:hi]=np.clip(.5*prediction[lo:hi].astype(np.float64)+.5*tree[lo:hi].astype(np.float64),1.05,4.95).astype('<f4')
        expected=source['data_manifest']['virtual_c_order_sha256']['qual']['arm_operator_sha256'][source['arm']]
        if [h.hexdigest() for h in operators]!=expected:raise ValueError('Every original qualifying operator bit required')
        if not np.isfinite(prediction).all() or not np.isfinite(half).all():raise ValueError('Every complete finite qualifying output row required')
        pred=study.save_array(ART/'NN-bag.npy',prediction,guard);control=study.save_array(ART/'fixed-half-tree154.npy',half,guard)
        result=dict(state=STATE,plan_sha256=execution.inputs.sha(REPORT/'plan.json'),arm=source['arm'],rows=ROWS,
            operators_sha256=[h.hexdigest() for h in operators],raw_neural_input_sha256=[h.hexdigest() for h in raw],
            normalized_sha256=[h.hexdigest() for h in normalized],fold_sha256=[h.hexdigest() for h in folds],
            seed_sha256=[[h.hexdigest() if h else None for h in row] for row in seeds],
            NN=pred,fixed_half=control,known_NN_provider=source['roles']['known_NN_provider'],
            known_fixed_half_control=source['roles']['known_fixed_half_control'],
            actual_models_reopened=source['actual_models'],models_fitted=0,qualifying_labels_read=False,
            remaining='Separate complete qualifying source/operator/scaler/model/seed/fold/output replay')
    finally:
        try:
            if study.inputs.context.provider.original.verify_bindings(source['source_sha256'],guard)!=source['records']:
                raise ValueError('Every original source/model identity must survive qualifying inference')
        finally:
            gc.collect();torch.mps.empty_cache()
    study.write_json(REPORT/'inference-completion.json',result,guard);return result
