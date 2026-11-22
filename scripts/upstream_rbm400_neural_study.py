"""Durable full-budget scalar/context neural studies; no CLI or queue owner.

Production callers must supply the separately reviewed live resource guard.
Preparation requires actual complete CPU data proof; fitting requires MPS.
Complete actual model/preprocessing replay and arm selection are separate work.
"""
from contextlib import contextmanager,ExitStack
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
import upstream_rbm400_neural_inputs as inputs
import upstream_rbm400_neural_fold as folds
import upstream_rbm400_latent_neural as architecture
import upstream_rbm400_latent_cache as cache
import upstream_factored_rbm_terminal_metal as terminal
import upstream_mps_context as backend

ROOT=inputs.ROOT
ARMS=('scalar','latent')
ROWS=1128164
TREE_MANIFEST=ROOT/'experiments/upstream-complete-trajectory-h128-mps/manifest.json'
DECL=ROOT/'OWN_RBM400_NEURAL_EXECUTION_PLAN.md'
PREPARED_STATE='declared_complete_original_budget_RBM400_neural_arm_pending_fits'
FIT_STATE='complete_all_five_original_budget_RBM400_neural_folds_pending_actual_model_replay'


def paths(arm):
    if arm not in ARMS:raise ValueError('Exact predeclared scalar/latent arm required')
    stem='upstream-own-rbm400-'+arm+'-h128-mps'
    return ROOT/'artifacts'/stem,ROOT/'experiments'/stem


def code_bindings():
    files=[Path(__file__),Path(inputs.__file__),Path(folds.__file__),Path(folds.scaling.__file__),
           Path(architecture.__file__),Path(architecture.ops.__file__),Path(backend.__file__),
           Path(architecture.original.__file__),Path(backend.n.__file__),DECL]
    return {str(p.relative_to(ROOT)):inputs.cpu.inputs.sha(p) for p in files}


def versions():return {name:importlib.metadata.version(name) for name in ('numpy','torch','scipy')}


def write_json(path,value,guard):
    return terminal.durable_payload(path,lambda stream:stream.write((json.dumps(value,indent=2)+'\n').encode()),ROOT,guard)


def save_array(path,value,guard):
    value=np.ascontiguousarray(value)
    return terminal.durable_payload(path,lambda stream:np.save(stream,value,allow_pickle=False),ROOT,guard)


def prepared_sources(guard):
    declaration,manifest,_=inputs.sources(guard)
    bindings=dict(declaration['source_sha256'])
    for name,digest in code_bindings().items():inputs.context.bind(bindings,ROOT/name,digest)
    tree_manifest=inputs.cpu.read(TREE_MANIFEST)
    controls=[row for row in tree_manifest['controls'] if row['id']=='tree154']
    if len(controls)!=1:raise ValueError('Original single tree154 OOF control required')
    tree=controls[0];relative=tree['prediction']
    if relative not in bindings:raise ValueError('Tree154 must already belong to the complete inherited own source closure')
    inputs.context.bind(bindings,TREE_MANIFEST)
    records=inputs.context.provider.original.verify_bindings(bindings,guard)
    declaration['source_sha256']=bindings
    return declaration,manifest,records,tree


def prepare(guard):
    if not callable(guard):raise ValueError('External owned prepare guard required')
    for arm in ARMS:
        if any(path.exists() for path in paths(arm)):raise ValueError('Existing complete/partial arm namespace preserved; no retry')
    declaration,data_manifest,records,tree=prepared_sources(guard)
    keep=data_manifest['alias']['retain_scalar'];receipts={}
    for arm in ARMS:
        run,report=paths(arm);terminal.durable_directory(run,ROOT,guard);terminal.durable_directory(report,ROOT,guard)
        layout=architecture.ops.layout(architecture.original.signal_indices(713),architecture.original.CONTEXT,
                                       retain_scalar=keep,latent=arm=='latent')
        manifest=dict(state=PREPARED_STATE,arm=arm,config=architecture.CONFIGS[arm],dimensions=data_manifest['arms'][arm],
            layout=layout,versions=versions(),code_sha256=code_bindings(),source_sha256=declaration['source_sha256'],
            parent_CPU_manifest_sha256=inputs.cpu.inputs.sha(inputs.data.REPORT/'manifest.json'),
            parent_CPU_verification_sha256=inputs.cpu.inputs.sha(inputs.data.REPORT/'independent-verification.json'),
            controls=[tree],tree_provenance='Retain historical tree154 A/B realization caveat',fixed_tree_weight=.5,
            matched_known_and_qualifying_operators=True,rows=ROWS,qualifying_rows=2817131,
            schedule='all_original_independent_inner_selections_then_fresh_outer_refits',
            single_full_matrix_memory_construction=True,actual_peak_RAM_verified=False,
            nonlinear_component_OOF_predictions_used_as_features=False,qualifying_labels_read=False,neural_models_fitted=0)
        receipts[arm]=write_json(report/'manifest.json',manifest,guard)
    if inputs.context.provider.original.verify_bindings(declaration['source_sha256'],guard)!=records:
        raise ValueError('Original source identity changed during preparation')
    return receipts


def load(arm,guard):
    _,report=paths(arm);manifest=inputs.cpu.read(report/'manifest.json')
    if (manifest['state']!=PREPARED_STATE or manifest['arm']!=arm or manifest['config']!=architecture.CONFIGS[arm]
            or manifest['code_sha256']!=code_bindings() or manifest['versions']!=versions()
            or manifest['qualifying_labels_read'] is not False
            or manifest['fixed_tree_weight']!=.5
            or manifest['matched_known_and_qualifying_operators'] is not True
            or manifest['schedule']!='all_original_independent_inner_selections_then_fresh_outer_refits'
            or manifest['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or manifest['rows']!=ROWS or manifest['qualifying_rows']!=2817131):
        raise ValueError('Exact original known-only neural preparation/source/runtime required')
    declaration,data_manifest,records,tree=prepared_sources(guard)
    expected_layout=architecture.ops.layout(architecture.original.signal_indices(713),architecture.original.CONTEXT,
        retain_scalar=data_manifest['alias']['retain_scalar'],latent=arm=='latent')
    if (manifest['source_sha256']!=declaration['source_sha256'] or manifest['controls']!=[tree]
            or manifest['layout']!=expected_layout
            or manifest['dimensions']!=data_manifest['arms'][arm]
            or manifest['parent_CPU_manifest_sha256']!=inputs.cpu.inputs.sha(inputs.data.REPORT/'manifest.json')
            or manifest['parent_CPU_verification_sha256']!=inputs.cpu.inputs.sha(inputs.data.REPORT/'independent-verification.json')):
        raise ValueError('Complete original data/alias/tree/source proofs must remain unchanged')
    return manifest,declaration,data_manifest,records


@contextmanager
def known_arrays(data_manifest,records,guard):
    d=data_manifest['arms']['scalar']['design_columns']
    schema={'labels.npy':('<f8',(ROWS,)),'users.npy':('<i4',(ROWS,)),'groups.npy':('i1',(ROWS,)),
            'gram.npy':('<f8',(10,d,d)),'rhs.npy':('<f8',(10,d))}
    with ExitStack() as stack:
        arrays={}
        for name,(dtype,shape) in schema.items():
            path=Path(data_manifest['prepared'][name]['path'])
            if path!=inputs.data.ART/name:raise ValueError('Original complete owned CPU prepared file required')
            arrays[name]=stack.enter_context(cache.mapped(path,records[str(path.relative_to(ROOT))],dtype,shape,guard))
        yield arrays


def runtime():
    if any(os.environ.get(k)!=v for k,v in inputs.context.provider.rbm.ENV.items()):
        raise ValueError('Original source-bound runtime environment required')
    import torch
    if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS required; no CPU substitution')
    backend.MODEL_FACTORY=architecture.model_factory
    return torch


def checkpoint(path,model,arm,fold,seed,count,width,guard):
    if (arm not in ARMS or type(count) is not int or not 1<=count<=40
            or type(fold) is not int or not 0<=fold<5 or seed not in architecture.CONFIGS[arm]['seeds']
            or width not in [architecture.ops.dimensions(keep,arm=='latent')['nn_inputs'] for keep in (False,True)]):
        raise ValueError('Actual nonzero original known-selected model slot required')
    if any(p.device.type!='mps' for p in model.parameters()):raise ValueError('Actual original MPS model required for checkpoint')
    if not hasattr(model,'base') or model.base[0].in_features!=width:
        raise ValueError('Actual model input width must match checkpoint metadata')
    guard()
    state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    value=dict(state=state,config=architecture.CONFIGS[arm],feature_width=width,seed=seed+fold,
        seed_base=seed,epochs=count,fold=fold,arm=arm,backend='torch-mps-float32',
        optimizer_state_retained=False,training_resume_supported=False)
    import torch
    if any(not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Every finite actual predictive tensor required')
    keep=width==architecture.ops.dimensions(True,arm=='latent')['nn_inputs']
    layout=architecture.ops.layout(architecture.original.signal_indices(713),architecture.original.CONTEXT,
                                    retain_scalar=keep,latent=arm=='latent')
    h,r=architecture.CONFIGS[arm]['nn_hidden'],architecture.CONFIGS[arm]['context_rank']
    shapes={'signals':(len(layout['signals']),),'context':(len(layout['context']),),
        'base.0.weight':(h,width),'base.0.bias':(h,),'base.3.weight':(h,h),'base.3.bias':(h,),
        'base.6.weight':(1,h),'base.6.bias':(1,),'project.weight':(r,len(layout['signals'])),
        'gate.0.weight':(h,len(layout['context'])),'gate.0.bias':(h,),'gate.3.weight':(r,h),'gate.3.bias':(r,)}
    if set(state)!=set(shapes) or any(tuple(state[k].shape)!=shape or state[k].dtype!=(torch.int64 if k in ('signals','context') else torch.float32)
                                     for k,shape in shapes.items()):
        raise ValueError('Every exact complete original predictive tensor shape/dtype required')
    if any(not torch.equal(state[k],torch.tensor(layout[k],dtype=torch.int64)) for k in ('signals','context')):
        raise ValueError('Actual signal/context buffers must match the full declared layout')
    return terminal.durable_payload(path,lambda stream:torch.save(value,stream),ROOT,guard)


def fit(arm,guard):
    torch=runtime();manifest,declaration,data_manifest,records=load(arm,guard)
    run,report=paths(arm);completion=report/'fit-completion.json'
    if completion.exists() or (run/'nn').exists():raise ValueError('Original complete/partial neural fit preserved; no restart')
    manifest_sha=inputs.cpu.inputs.sha(report/'manifest.json');width=manifest['dimensions']['nn_inputs'];fold_proofs={}
    with known_arrays(data_manifest,records,guard) as arrays:
        labels,users,groups=[arrays[name] for name in ('labels.npy','users.npy','groups.npy')]
        gram,rhs=arrays['gram.npy'],arrays['rhs.npy'];cfg=architecture.CONFIGS[arm]
        for fold in range(5):
            root=run/'nn'/str(fold);terminal.durable_directory(root,ROOT,guard)
            if any(root.iterdir()):raise ValueError('Original partial fold retained')
            eligible=np.arange(10)//2!=fold
            inner=backend.n.ridge_weights(gram,rhs,eligible&(np.arange(10)%2==0),cfg['ridge_lambda'])
            full=backend.n.ridge_weights(gram,rhs,eligible,cfg['ridge_lambda'])
            outputs={}
            def build(stage,weights):return inputs.originals(arm,weights,declaration,data_manifest,records,guard)
            def save_preprocessing(stage,weights,scaler,record):
                outputs[stage+'-weights.npy']=save_array(root/(stage+'-weights.npy'),weights,guard)
                for name,value in scaler.items():outputs[stage+'-'+name+'.npy']=save_array(root/(stage+'-'+name+'.npy'),value,guard)
                outputs[stage+'-pipeline.json']=write_json(root/(stage+'-pipeline.json'),record,guard)
            def save_model(model,seed,count):
                name='nn-%s.pt'%seed
                outputs[name]=checkpoint(root/name,model,arm,fold,seed,count,width,guard)
            def release():gc.collect();torch.mps.empty_cache()
            result=folds.run(arm,fold,labels,users,groups,inner,full,build=build,train_nn=backend.train_nn,
                predict=backend.predict,save_preprocessing=save_preprocessing,save_model=save_model,release=release,guard=guard)
            prediction=result.pop('prediction');outputs['prediction.npy']=save_array(root/'prediction.npy',prediction,guard)
            model_names={'nn-%s.pt'%s for s,c in zip(cfg['seeds'],result['chosen_iterations']) if c}
            required={stage+'-'+leaf for stage in ('inner','full') for leaf in
                      ('weights.npy','lo.npy','hi.npy','mean.npy','std.npy','pipeline.json')}|{'prediction.npy'}|model_names
            if set(outputs)!=required or {p.name for p in root.iterdir()}!=required:raise ValueError('Complete exact actual fold inventory required')
            receipt=dict(result,manifest_sha256=manifest_sha,outputs=outputs,actual_MPS_fit=True,
                         actual_saved_model_count=len(model_names),qualifying_labels_read=False)
            path=report/'nn'/('%s.json'%fold);fold_proofs[str(path.relative_to(ROOT))]=write_json(path,receipt,guard)['sha256']
    if inputs.context.provider.original.verify_bindings(declaration['source_sha256'],guard)!=records:
        raise ValueError('Full original source identity changed during neural fitting')
    result=dict(state=FIT_STATE,arm=arm,manifest_sha256=manifest_sha,fold_proof_sha256=fold_proofs,
                all_five_folds_complete=True,all_three_seed_slots_each_fold=True,actual_MPS_fit=True,
                selected_zero_iteration_slots_retained=True,qualifying_labels_read=False,
                remaining='Full independent actual-model/scaler/input/prediction replay and canonical known-only scoring')
    write_json(completion,result,guard);return result
