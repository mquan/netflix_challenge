"""Separate compensated known scoring and predeclared two-arm selection.

No model fitting or qualifying outcome access. Both actual MPS model/pipeline
replays must complete before either score can participate in selection.
"""
from contextlib import ExitStack
import math
import os
from pathlib import Path
import numpy as np
import upstream_rbm400_neural_study as study
import verify_upstream_rbm400_neural_study as replay
import upstream_factored_rbm_admission as admission

STATE='complete_separate_RBM400_neural_canonical_known_scores'
SELECTION_STATE='complete_known_only_RBM400_neural_arm_selection_before_qualifying_bag'
TIE=1e-12


def score(values,labels):
    if (values.shape!=labels.shape or values.ndim!=1 or not len(values)
            or values.dtype!=np.dtype('<f4') or not np.isfinite(values).all()
            or not np.isin(labels,[1,2,3,4,5]).all()):
        raise ValueError('Complete finite stored vectors and original known outcomes required')
    clipped=np.clip(values.astype(np.float64),1.05,4.95)
    return math.sqrt(math.fsum(float(v) for v in (clipped-labels)**2)/len(labels))


def choose(scores):
    if (set(scores)!=set(study.ARMS) or any(type(v) is not float or not math.isfinite(v) or v<0 for v in scores.values())):
        raise ValueError('Exactly two complete finite canonical known arm scores required')
    return 'latent' if scores['latent']<scores['scalar']-TIE else 'scalar'


def evidence(arm,guard):
    manifest,declaration,data_manifest,source_records=study.load(arm,guard)
    run,report=study.paths(arm);path=report/'independent-verification.json';v=study.inputs.cpu.read(path)
    if (v['state']!=replay.STATE or v['arm']!=arm or v['manifest_sha256']!=study.inputs.cpu.inputs.sha(report/'manifest.json')
            or v['all_five_original_folds_replayed'] is not True or v['all_three_seed_slots_each_fold'] is not True
            or v['all_actual_models_preserved_without_refitting'] is not True
            or v['all_eligible_ridge_weights_scalers_raw_and_normalized_inputs_bitwise_equal'] is not True
            or v['every_held_prediction_bit_equal'] is not True or v['actual_MPS_predictions_replayed'] is not True
            or v['qualifying_labels_read'] is not False or v['source_sha256']!=manifest['source_sha256']
            or [r['fold'] for r in v['records']]!=list(range(5))):
        raise ValueError('Both complete eligible pipelines and every actual held model replay required')
    receipts,bindings=replay.receipt_bindings(arm,manifest)
    if bindings!=v['fit_payload_sha256']:raise ValueError('Exact independently replayed actual fit payloads required')
    for fold,(receipt,row) in enumerate(zip(receipts,v['records'])):
        if (row['chosen_iterations']!=receipt['chosen_iterations']
                or row['actual_models_reopened']!=sum(c>0 for c in receipt['chosen_iterations'])
                or row['both_complete_eligible_scalers_recomputed'] is not True
                or row['all_raw_and_normalized_input_bits_equal'] is not True
                or row['actual_saved_pipeline_bitwise_equal'] is not True
                or row['receipt_sha256']!=bindings[str((report/'nn'/('%s.json'%fold)).relative_to(study.ROOT))]):
            raise ValueError('Every original fold/seed/actual model count must match')
    if v['actual_saved_models_reopened']!=sum(row['actual_models_reopened'] for row in v['records']):
        raise ValueError('Complete actual saved model population required')
    for field,name in [('nn','nn.npy'),('fixed_half','fixed-half-tree154.npy')]:
        row=v[field];p=Path(row['path'])
        if p!=run/name or p.resolve()!=p or p.stat().st_size!=row['bytes']:raise ValueError('Exact complete verified output path required')
        study.inputs.context.bind(bindings,p,row['sha256'])
    study.inputs.context.bind(bindings,path)
    records=study.inputs.context.provider.original.verify_bindings(bindings,guard)
    return manifest,declaration,data_manifest,source_records,v,receipts,bindings,records


def canonical(arm,guard):
    run,report=study.paths(arm);output=report/'independent-canonical-verification.json'
    if output.exists():raise ValueError('Original separate canonical proof retained; no overwrite')
    manifest,declaration,data_manifest,source_records,v,receipts,bindings,records=evidence(arm,guard)
    relative=lambda path:str(path.relative_to(study.ROOT));scores=[]
    try:
        with study.known_arrays(data_manifest,source_records,guard) as arrays,ExitStack() as stack:
            labels,groups=arrays['labels.npy'],arrays['groups.npy']
            nn=stack.enter_context(study.cache.mapped(run/'nn.npy',records[relative(run/'nn.npy')],'<f4',(study.ROWS,),guard))
            half=stack.enter_context(study.cache.mapped(run/'fixed-half-tree154.npy',records[relative(run/'fixed-half-tree154.npy')],'<f4',(study.ROWS,),guard))
            reference=np.empty(study.ROWS,dtype='<f4');seen=np.zeros(study.ROWS,dtype=bool)
            if set(np.unique(groups))!=set(range(10)):raise ValueError('Complete original user-fold groups required')
            for fold in range(5):
                guard();held=groups//2==fold;p=run/'nn'/str(fold)/'prediction.npy'
                values=stack.enter_context(study.cache.mapped(p,records[relative(p)],'<f4',(int(held.sum()),),guard))
                if np.any(seen[held]):raise ValueError('Every original known row belongs to one held fold')
                reference[held]=values;seen[held]=True
            if not seen.all() or reference.tobytes()!=nn.tobytes():raise ValueError('Every canonical NN float32 bit must match held fits')
            tree_path=study.ROOT/manifest['controls'][0]['prediction']
            with tree_path.open('rb') as stream:
                before=os.fstat(stream.fileno())
                if admission.file_identity(before)!=source_records[relative(tree_path)]['identity']:raise ValueError('Authenticated original tree descriptor required')
                tree=np.frombuffer(stream.read(),dtype='<f4')
                if (admission.file_identity(before)!=admission.file_identity(os.fstat(stream.fileno()))
                        or admission.file_identity(before)!=admission.file_identity(tree_path.stat())):raise ValueError('Original tree file changed during scoring')
            if tree.shape!=nn.shape or not np.isfinite(tree).all():raise ValueError('Complete original tree OOF control required')
            expected=np.clip(.5*nn.astype(np.float64)+.5*tree.astype(np.float64),1.05,4.95).astype('<f4')
            if expected.tobytes()!=half.tobytes():raise ValueError('Every canonical fixed-half bit must independently match')
            scores=[score(nn,labels),score(half,labels)]
        result=dict(state=STATE,arm=arm,manifest_sha256=study.inputs.cpu.inputs.sha(report/'manifest.json'),
            actual_model_replay_sha256=study.inputs.cpu.inputs.sha(report/'independent-verification.json'),
            actual_models_rehashed=v['actual_saved_models_reopened'],all_five_original_folds_verified=True,
            every_nn_and_fixed_half_float32_bit_equal=True,nn_rmse=scores[0],fixed_half_tree154_rmse=scores[1],
            nn=v['nn'],fixed_half=v['fixed_half'],source_sha256=manifest['source_sha256'],
            fit_and_replay_payload_sha256=bindings,qualifying_labels_read=False,
            qualifying_labels_used_for_fitting_or_selection=False,known_score_is_five_user_fold_CV=True,
            tree_control_retains_historical_A_B_realization_caveat=True,goal_complete=False)
    finally:
        try:
            if study.inputs.context.provider.original.verify_bindings(declaration['source_sha256'],guard)!=source_records:
                raise ValueError('Source closure identity changed during separate canonical scoring')
        finally:
            if study.inputs.context.provider.original.verify_bindings(bindings,guard)!=records:
                raise ValueError('Actual models/replays/vectors changed during separate canonical scoring')
    study.write_json(output,result,guard);return result


def select(guard):
    report=study.ROOT/'experiments/upstream-own-rbm400-neural-selection';output=report/'selection.json'
    if report.exists():raise ValueError('Existing complete/partial known selection preserved')
    scores={};bindings={};canonical_hashes={};vectors={};model_counts={}
    for arm in study.ARMS:
        manifest,declaration,data_manifest,source_records,v,_,fit_bindings,fit_records=evidence(arm,guard)
        run,arm_report=study.paths(arm);path=arm_report/'independent-canonical-verification.json';c=study.inputs.cpu.read(path)
        if (c['state']!=STATE or c['arm']!=arm or c['manifest_sha256']!=study.inputs.cpu.inputs.sha(arm_report/'manifest.json')
                or c['actual_model_replay_sha256']!=study.inputs.cpu.inputs.sha(arm_report/'independent-verification.json')
                or c['source_sha256']!=manifest['source_sha256'] or c['fit_and_replay_payload_sha256']!=fit_bindings
                or c['every_nn_and_fixed_half_float32_bit_equal'] is not True
                or c['all_five_original_folds_verified'] is not True
                or c['qualifying_labels_read'] is not False or c['qualifying_labels_used_for_fitting_or_selection'] is not False
                or c['nn']!=v['nn'] or c['fixed_half']!=v['fixed_half']
                or c['actual_models_rehashed']!=v['actual_saved_models_reopened']):
            raise ValueError('Both complete separate canonical known scores required before selection')
        with study.known_arrays(data_manifest,source_records,guard) as arrays:
            for field,filename,key in [('nn','nn.npy','nn_rmse'),('fixed_half','fixed-half-tree154.npy','fixed_half_tree154_rmse')]:
                p=run/filename;relative=str(p.relative_to(study.ROOT))
                with study.cache.mapped(p,fit_records[relative],'<f4',(study.ROWS,),guard) as values:
                    actual=score(values,arrays['labels.npy'])
                if abs(actual-c[key])>1e-12:raise ValueError('Each complete known score must independently reproduce before selection')
                if field=='fixed_half':scores[arm]=actual
        for collection in (declaration['source_sha256'],fit_bindings):
            for name,digest in collection.items():study.inputs.context.bind(bindings,study.ROOT/name,digest)
        study.inputs.context.bind(bindings,path);canonical_hashes[arm]=study.inputs.cpu.inputs.sha(path)
        vectors[arm]=v['fixed_half'];model_counts[arm]=v['actual_saved_models_reopened']
        if (study.inputs.context.provider.original.verify_bindings(declaration['source_sha256'],guard)!=source_records
                or study.inputs.context.provider.original.verify_bindings(fit_bindings,guard)!=fit_records):
            raise ValueError('Both complete arm source/model/vector identities must remain unchanged')
    selected=choose(scores);records=study.inputs.context.provider.original.verify_bindings(bindings,guard)
    result=dict(state=SELECTION_STATE,selected_arm=selected,known_fixed_half_scores=scores,tie_tolerance=TIE,
        tie_rule='scalar',both_arms_complete_before_selection=True,canonical_sha256=canonical_hashes,
        selected_known_vector=vectors[selected],actual_saved_model_counts=model_counts,source_sha256=bindings,
        qualifying_labels_read=False,qualifying_labels_used_for_selection=False,
        selected_using_only_complete_known_fixed_half_scores=True,goal_complete=False,
        remaining='Complete selected-arm qualifying bag/replay and own17 freeze/preaudit/fixed audits')
    if study.inputs.context.provider.original.verify_bindings(bindings,guard)!=records:raise ValueError('Selection source identity changed')
    study.terminal.durable_directory(report,study.ROOT,guard);study.write_json(output,result,guard);return result
