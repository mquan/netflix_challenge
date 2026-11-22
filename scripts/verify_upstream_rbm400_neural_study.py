"""Full independent raw/scaler/checkpoint/held-prediction replay; no CLI/owner."""
from contextlib import contextmanager,ExitStack
import gc
import hashlib
import os
from pathlib import Path
import numpy as np
import upstream_rbm400_neural_study as study
import upstream_factored_rbm_admission as admission

STATE='complete_independent_RBM400_neural_actual_models_all_eligible_scalers_inputs_and_predictions'


def hash_rows(value,guard):
    h=hashlib.sha256()
    for lo in range(0,len(value),8192):guard();h.update(value[lo:lo+8192].tobytes())
    return h.hexdigest()


def independent_scaler(raw,eligible,guard):
    if (raw.dtype!=np.float32 or not raw.flags.c_contiguous or raw.shape[1] not in (713,714,863,864)
            or eligible.dtype!=np.bool_ or eligible.shape!=(len(raw),) or not eligible.any()):
        raise ValueError('Complete declared raw matrix and eligible rows required')
    result={key:np.empty(raw.shape[1]) for key in ('lo','hi','mean','std')}
    count=int(eligible.sum())
    for column in range(raw.shape[1]):
        guard();values=raw[eligible,column]
        if not np.isfinite(values).all():raise ValueError('All finite eligible values required')
        lo,hi=np.percentile(values,[1,99]);result['lo'][column]=lo;result['hi'][column]=hi
        clipped=np.clip(values.astype(np.float64),lo,hi);clipped[0]+=np.float64(0.)
        mean=np.add.accumulate(clipped)[-1]/count
        squared=np.clip(values.astype(np.float64),lo,hi)-mean;np.multiply(squared,squared,out=squared);squared[0]+=np.float64(0.)
        result['mean'][column]=mean;result['std'][column]=max(float(np.sqrt(np.add.accumulate(squared)[-1]/count)),1e-6)
    return result


def original_matrix(arm,weights,declaration,data_manifest,records,guard):
    keep=data_manifest['alias']['retain_scalar'];dims=data_manifest['arms'][arm]
    features=np.empty((study.ROWS,dims['nn_inputs']),dtype='<f4');baseline=np.empty(study.ROWS,dtype=np.float64)
    digests=[hashlib.sha256(),hashlib.sha256()]
    with study.inputs.context.open_query('probex',declaration,records,guard,retain_scalar=keep,latent=arm=='latent') as reader:
        if len(reader.prediction)!=study.ROWS:raise ValueError('Every canonical known row required')
        for start in range(0,study.ROWS,8192):
            guard();stop=min(start+8192,study.ROWS);z,f=reader.block(start,stop)
            digests[0].update(z.tobytes());digests[1].update(f.tobytes())
            baseline[start:stop]=np.clip(np.matmul(z,weights),1.05,4.95)
            features[start:stop,:-1]=f;features[start:stop,-1]=baseline[start:stop]
    expected=data_manifest['virtual_c_order_sha256']['probex']['arm_operator_sha256'][arm]
    if [h.hexdigest() for h in digests]!=expected:raise ValueError('All independently reconstructed operator bits required')
    return features,baseline,dict(operator_sha256=expected,raw_neural_input_sha256=hash_rows(features,guard),
        rows=study.ROWS,width=dims['nn_inputs'],baseline_stored_in_input_as_float32=True)


def receipt_bindings(arm,manifest):
    run,report=study.paths(arm);complete_path=report/'fit-completion.json';complete=study.inputs.cpu.read(complete_path)
    manifest_sha=study.inputs.cpu.inputs.sha(report/'manifest.json')
    expected={str((report/'nn'/('%s.json'%fold)).relative_to(study.ROOT)) for fold in range(5)}
    if (complete['state']!=study.FIT_STATE or complete['arm']!=arm or complete['manifest_sha256']!=manifest_sha
            or complete['all_five_folds_complete'] is not True or complete['all_three_seed_slots_each_fold'] is not True
            or complete['actual_MPS_fit'] is not True or complete['qualifying_labels_read'] is not False
            or set(complete['fold_proof_sha256'])!=expected):
        raise ValueError('All complete original fitted folds and seed slots required')
    bindings=dict(complete['fold_proof_sha256']);receipts=[];seeds=manifest['config']['seeds']
    for fold in range(5):
        path=report/'nn'/('%s.json'%fold);receipt=study.inputs.cpu.read(path);counts=receipt['chosen_iterations']
        if (receipt['fold']!=fold or receipt['manifest_sha256']!=manifest_sha or receipt['seed_order']!=seeds
                or receipt['seed_offset']!=fold or receipt['actual_MPS_fit'] is not True
                or receipt['qualifying_labels_read'] is not False or len(counts)!=3
                or any(type(v) is not int or not 0<=v<=40 for v in counts)
                or receipt['schedule']!='all_original_independent_inner_selections_then_fresh_outer_refits'):
            raise ValueError('Every original actual known-selected slot/count/seed required')
        files={stage+'-'+name for stage in ('inner','full') for name in
               ('weights.npy','lo.npy','hi.npy','mean.npy','std.npy','pipeline.json')}|{'prediction.npy'}
        files|={'nn-%s.pt'%seed for seed,count in zip(seeds,counts) if count}
        root=run/'nn'/str(fold)
        if (set(receipt['outputs'])!=files or {p.name for p in root.iterdir()}!=files
                or receipt['actual_saved_model_count']!=sum(count>0 for count in counts)):
            raise ValueError('Every complete original saved model/scaler/prediction file required')
        for name,row in receipt['outputs'].items():
            p=Path(row['path'])
            if p!=root/name or p.resolve()!=p or p.stat().st_size!=row['bytes']:
                raise ValueError('Exact regular owned original pipeline payload required')
            study.inputs.context.bind(bindings,p,row['sha256'])
        receipts.append(receipt)
    for path in (complete_path,report/'manifest.json'):study.inputs.context.bind(bindings,path)
    return receipts,bindings


@contextmanager
def actual_model(path,record,manifest,fold,seed,count,guard):
    import torch
    guard()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if admission.file_identity(before)!=record['identity']:raise ValueError('Authenticated original model FD required')
        saved=torch.load(stream,map_location='cpu',weights_only=True)
        if (saved['config']!=manifest['config'] or saved['feature_width']!=manifest['dimensions']['nn_inputs']
                or saved['arm']!=manifest['arm'] or saved['fold']!=fold or saved['seed_base']!=seed
                or saved['seed']!=seed+fold or saved['epochs']!=count or saved['backend']!='torch-mps-float32'
                or saved['optimizer_state_retained'] is not False or saved['training_resume_supported'] is not False):
            raise ValueError('Actual original complete predictive checkpoint metadata required')
        torch.set_num_threads(manifest['config']['threads'])
        model=study.architecture.model_factory(saved['feature_width'],manifest['config']);initial=model.state_dict();state=saved['state']
        if set(state)!=set(initial) or any(value.device.type!='cpu' or value.dtype!=initial[name].dtype
                or value.shape!=initial[name].shape or not bool(torch.isfinite(value).all()) for name,value in state.items()):
            raise ValueError('Every exact finite original predictive tensor required without coercion')
        for name in ('signals','context'):
            expected=torch.tensor(manifest['layout'][name],dtype=torch.int64)
            if not torch.equal(state[name],initial[name]) or not torch.equal(state[name],expected):
                raise ValueError('Every actual signal/context position must match the declaration')
        model.load_state_dict(state,strict=True);model=model.to('mps')
        try:yield model
        finally:
            guard()
            if (admission.file_identity(before)!=admission.file_identity(os.fstat(stream.fileno()))
                    or admission.file_identity(before)!=admission.file_identity(path.stat())):
                raise ValueError('Original actual model changed during full replay')


def verify(arm,guard):
    torch=study.runtime();manifest,declaration,data_manifest,source_records=study.load(arm,guard)
    run,report=study.paths(arm);output=report/'independent-verification.json'
    if output.exists() or (run/'nn.npy').exists() or (run/'fixed-half-tree154.npy').exists():
        raise ValueError('Existing complete/partial independent replay preserved; no automatic retry')
    receipts,bindings=receipt_bindings(arm,manifest)
    records=study.inputs.context.provider.original.verify_bindings(bindings,guard)
    relative=lambda path:str(path.relative_to(study.ROOT))
    dimensions=manifest['dimensions'];d=dimensions['design_columns'];width=dimensions['nn_inputs'];fold_records=[]
    prediction=np.empty(study.ROWS,dtype='<f4');seen=np.zeros(study.ROWS,dtype=bool)
    try:
        with study.known_arrays(data_manifest,source_records,guard) as arrays:
            labels,users,groups=[arrays[name] for name in ('labels.npy','users.npy','groups.npy')]
            order=np.argsort(users,kind='stable')
            if (set(np.unique(groups))!=set(range(10)) or not np.isin(labels,[1,2,3,4,5]).all()
                    or np.any((users[order][1:]==users[order][:-1])&(groups[order][1:]!=groups[order][:-1]))):
                raise ValueError('Complete original known outcomes and disjoint inner/outer user groups required')
            gram,rhs=arrays['gram.npy'],arrays['rhs.npy'];regularizer=np.diag(np.r_[np.full(d-1,1000.),0.])
            for fold,receipt in enumerate(receipts):
                root=run/'nn'/str(fold);train=groups//2!=fold;held=~train;fit=train&(groups%2==0);eligible=np.arange(10)//2!=fold
                counts=receipt['chosen_iterations'];reopened=0;pipeline={};full_baseline=None;correction=np.zeros(int(held.sum()))
                for stage,selector,mask in [('inner',eligible&(np.arange(10)%2==0),fit),('full',eligible,train)]:
                    guard();weights=np.linalg.solve(gram[selector].sum(axis=0)+regularizer,rhs[selector].sum(axis=0))
                    with ExitStack() as stack:
                        saved_weights=stack.enter_context(study.cache.mapped(root/(stage+'-weights.npy'),records[relative(root/(stage+'-weights.npy'))],'<f8',(d,),guard))
                        if saved_weights.tobytes()!=weights.tobytes():raise ValueError('All independently solved eligible ridge weight bits required')
                        raw,baseline,raw_proof=original_matrix(arm,weights,declaration,data_manifest,source_records,guard)
                        scaler=independent_scaler(raw,mask,guard)
                        for name,value in scaler.items():
                            path=root/(stage+'-'+name+'.npy')
                            saved=stack.enter_context(study.cache.mapped(path,records[relative(path)],'<f8',(width,),guard))
                            if saved.tobytes()!=value.tobytes():raise ValueError('Every complete independent eligible scaler bit required')
                        pipe=study.inputs.cpu.read(root/(stage+'-pipeline.json'))
                        if pipe['raw']!=raw_proof or pipe['eligible_rows']!=int(mask.sum()):raise ValueError('Every raw input hash and eligible row count required')
                        for lo in range(0,len(raw),8192):
                            guard();hi=min(lo+8192,len(raw))
                            raw[lo:hi]=np.asarray((np.clip(raw[lo:hi],scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std'],dtype=np.float32,order='C')
                        normalized=hash_rows(raw,guard)
                        if normalized!=pipe['normalized_input_sha256']:raise ValueError('Every complete normalized input bit required')
                        pipeline[stage]=dict(raw=raw_proof,normalized_input_sha256=normalized,eligible_rows=int(mask.sum()))
                        if stage=='full':
                            for seed,count in zip(manifest['config']['seeds'],counts):
                                if count:
                                    path=root/('nn-%s.pt'%seed)
                                    with actual_model(path,records[relative(path)],manifest,fold,seed,count,guard) as model:
                                        values=study.backend.predict(model,raw[held])
                                        if values.dtype!=np.float32 or values.shape!=correction.shape or not np.isfinite(values).all():
                                            raise ValueError('Every finite actual-model held prediction required')
                                        correction+=values/3
                                    del model;gc.collect();torch.mps.empty_cache();reopened+=1
                            full_baseline=baseline[held].copy()
                        del raw,baseline;gc.collect();torch.mps.empty_cache()
                expected=np.clip(full_baseline+correction,1.05,4.95).astype('<f4')
                path=root/'prediction.npy'
                with study.cache.mapped(path,records[relative(path)],'<f4',(int(held.sum()),),guard) as saved:
                    if saved.tobytes()!=expected.tobytes() or np.any(seen[held]):raise ValueError('Every actual complete saved held-prediction bit required')
                prediction[held]=expected;seen[held]=True
                if receipt['preprocessing']!=pipeline:raise ValueError('Both actual eligible pipeline receipts must match')
                fold_records.append(dict(fold=fold,actual_models_reopened=reopened,chosen_iterations=counts,
                    receipt_sha256=bindings[relative(report/'nn'/('%s.json'%fold))],
                    both_complete_eligible_scalers_recomputed=True,all_raw_and_normalized_input_bits_equal=True,
                    actual_saved_pipeline_bitwise_equal=True))
            if not seen.all() or not np.isfinite(prediction).all():raise ValueError('Every complete held-user row required')
        tree_path=study.ROOT/manifest['controls'][0]['prediction']
        with tree_path.open('rb') as stream:
            if admission.file_identity(os.fstat(stream.fileno()))!=source_records[relative(tree_path)]['identity']:
                raise ValueError('Original fully authenticated tree OOF file required')
            tree=np.frombuffer(stream.read(),dtype='<f4')
        if tree.shape!=prediction.shape or not np.isfinite(tree).all():raise ValueError('Complete original tree154 OOF control required')
        half=np.clip(.5*prediction.astype(np.float64)+.5*tree.astype(np.float64),1.05,4.95).astype('<f4')
        nn_receipt=study.save_array(run/'nn.npy',prediction,guard);half_receipt=study.save_array(run/'fixed-half-tree154.npy',half,guard)
        result=dict(state=STATE,arm=arm,manifest_sha256=study.inputs.cpu.inputs.sha(report/'manifest.json'),records=fold_records,
            actual_saved_models_reopened=sum(row['actual_models_reopened'] for row in fold_records),
            all_five_original_folds_replayed=True,all_three_seed_slots_each_fold=True,
            all_actual_models_preserved_without_refitting=True,complete_original_group_statistics_inherited=True,
            all_eligible_ridge_weights_scalers_raw_and_normalized_inputs_bitwise_equal=True,
            every_held_prediction_bit_equal=True,actual_MPS_predictions_replayed=True,
            nn=nn_receipt,fixed_half=half_receipt,source_sha256=manifest['source_sha256'],
            fit_payload_sha256=bindings,qualifying_labels_read=False,
            tree_control_retains_historical_A_B_realization_caveat=True,
            remaining='Separate canonical known scores for both arms and known-only arm selection')
    finally:
        try:
            if study.inputs.context.provider.original.verify_bindings(declaration['source_sha256'],guard)!=source_records:
                raise ValueError('Complete original source identity changed during actual model replay')
        finally:
            if study.inputs.context.provider.original.verify_bindings(bindings,guard)!=records:
                raise ValueError('Actual original fit files changed during independent replay')
    study.write_json(output,result,guard);return result
