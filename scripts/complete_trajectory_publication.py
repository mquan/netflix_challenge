"""Publication admission for the original full trajectory neural study only."""
import hashlib
import json
from pathlib import Path

QUEUE_SHA='a063d39ca733345884cf430a59b07067dd99209cbba747ebc418ee02ad498968'
REFERENCE_SHA='ce7ae633d328f8e0a79055146b0cb15efa5af18bf6a4760f328640557fbe79e7'
REPORT='experiments/upstream-complete-trajectory-h128-mps'
RUN='artifacts/upstream-complete-trajectory-h128-mps'
DATA='experiments/upstream-complete-trajectory-data'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def validate(root,record,current_path,current,model_path,model,*,require_improvement=True):
    """Check original declaration, full data/replay/arithmetic and actual files.

    This consumes completed scientific proofs. It does not fit, replay a model,
    infer qualifying scores or promote a pending result.
    """
    root=Path(root);report=root/REPORT;run=root/RUN
    manifest_path=report/'manifest.json';manifest=read(manifest_path)
    require(current_path==report/'independent-canonical-verification.json' and
        model_path==report/'independent-verification.json','Exact complete trajectory proof paths required')
    queue=root/'experiments/upstream-complete-trajectory-neural-queue/plan.json'
    require(digest(queue)==QUEUE_SHA,'Original predeclared full neural queue required')
    for name,expected in read(queue)['source_sha256'].items():
        require(digest(root/name)==expected,'Original declared scientific source changed: '+name)
    reference_path=root/'experiments/upstream-own391-paired-h128-mps/independent-canonical-verification.json'
    require(digest(reference_path)==REFERENCE_SHA,'Original verified matched391 benchmark required')
    reference=read(reference_path);prior_manifest_path=reference_path.parent/'manifest.json'
    require(digest(prior_manifest_path)==reference['manifest_sha256'],'Original reference recipe changed')
    prior_manifest=read(prior_manifest_path)
    expected_config=dict(prior_manifest['config'],architecture='context-weighted-residual-complete-trajectory-h128-v1')
    reference_score=reference['fixed_half_tree154_rmse'];records=model['records']
    require(current['state']=='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores' and
        model['state']=='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions' and
        manifest['state']=='declared_original_budget_complete_trajectory_neural_arm' and
        current['variant']==model['variant']==manifest['variant']=='h128' and manifest['config']==expected_config and
        expected_config['nn_hidden']==128 and expected_config['context_rank']==32 and
        expected_config['seeds']==[2026,7,13] and expected_config['nn_epochs']==40 and
        expected_config['nn_patience']==4 and expected_config['nn_batch']==8192,
        'Full original H128 recipe and both completed proofs required')
    score=current['fixed_half_tree154_rmse']
    require(digest(current_path)==record['independent_verification_sha256']==record['verified_development_report_sha256'] and
        score==record['best_development_rmse'] and current['complete_actual_model_replay_sha256']==digest(model_path) and
        current['manifest_sha256']==model['manifest_sha256']==digest(manifest_path) and
        current['verifier_sha256']==digest(root/'scripts/verify_upstream_complete_trajectory_canonical.py') and
        model['verifier_sha256']==digest(root/'scripts/verify_upstream_complete_trajectory_mps.py') and
        current['source_sha256']==manifest['source_sha256'] and
        current['declared_global_reference_rmse']==model['declared_global_reference_rmse']==manifest['declared_global_reference_rmse']==reference_score and
        current['improves_previous']==model['fixed_half_below_declared_global_reference']==(score<reference_score) and
        (not require_improvement or score<reference_score),
        'Exact source-bound complete canonical score and reference comparison required')
    require(all(current[k] is True for k in ('all_five_original_folds_verified','every_nn_and_fixed_half_float32_bit_equal',
            'tree_control_retains_historical_A_B_realization_caveat','neural_inputs_matched_B_on_both_queries')) and
        all(model[k] is True for k in ('complete_original_statistics_bitwise_equal','all_original_models_preserved_without_refitting','matched_B_known_and_qualifying')) and
        current['qualifying_labels_read'] is False and model['qualifying_labels_read'] is False and
        current['goal_complete'] is False and model['goal_complete'] is False and model['production_started'] is False and
        model['nn_rmse']==current['nn_rmse'] and model['fixed_half_tree154_rmse']==score and
        model['fixed_half_sha256']==current['prediction_sha256'] and
        [r['fold'] for r in records]==list(range(5)) and sum(r['rows'] for r in records)==1128164 and
        current['actual_models_rehashed']==15 and all(r['actual_models_reopened']==3 and
            all(r[k] is True for k in ('both_complete_eligible_scalers_independently_recomputed','all_normalized_input_bits_equal','actual_saved_pipeline_bitwise_equal')) for r in records),
        'All15 actual models, full eligible preprocessing and separate arithmetic required')
    data_path=root/DATA/'manifest.json';data=read(data_path)
    dp=root/DATA/'independent-verification.json';dv=read(dp)
    ip=root/DATA/'input-verification.json';iv=read(ip);added=iv['retained_additions'];width=618+added
    im_path=root/DATA/'input-manifest.json';im=read(im_path)
    require(dv['state']=='complete_independent_all_trajectory_operators_statistics_ridge_and_stored_predictions' and
        iv['state']=='complete_independent_all519_scalar_pairs_and_both_operators' and
        dv['manifest_sha256']==manifest['input_manifest_sha256']==digest(data_path) and
        dv['input_verification_sha256']==data['input_verification_sha256']==digest(ip) and
        data['input_manifest_sha256']==iv['input_manifest_sha256']==digest(im_path) and
        im['retained']==data['retained'] and im['qualifying_labels_read'] is False and
        dv['source_sha256']==iv['source_sha256']==im['source_sha256']==data['source_sha256'] and
        all(iv[k] is True for k in ('original391_retained','all128_candidates_replayed','every_complete_parent_operator_bit_equal')) and
        dv['every_full_operator_statistic_solve_and_prediction_bit_equal'] is True and
        dv['all_original_paired_B391_bits_preserved'] is True and dv['matched_B_known_and_qualifying'] is True and
        dv['qualifying_labels_read'] is False and iv['qualifying_labels_read'] is False and
        dv['neural_models_fitted']==0 and 0<=added<=128 and added+iv['exact_paired_aliases']==128 and
        len(data['retained'])==added and
        all([row[k] for k in ('raw_predictions','design_columns','nonlinear_columns','nn_inputs')]==
            [391+added,467+added,617+added,width] for row in (manifest,data,dv)) and
        manifest['context_indices']==prior_manifest['context_indices'] and len(manifest['context_indices'])==225 and
        manifest['signal_indices']==prior_manifest['signal_indices'][:-1]+list(range(617,width-1))+[width-1] and
        len(manifest['signal_indices'])==width-225 and
        sorted(manifest['signal_indices']+manifest['context_indices'])==list(range(width)) and
        manifest['fixed_tree_weight']==.5 and manifest['matched_B_known_and_qualifying'] is True and
        manifest['qualifying_labels_read'] is False,
        'Complete519-pair admission, exact aliases and full dynamic input dimensions required')
    bindings=manifest['source_sha256']
    require(bool(data['source_sha256']) and all(bindings.get(name)==value for name,value in data['source_sha256'].items()),
        'Complete upstream data source closure must be inherited')
    for path in (data_path,dp,reference_path):
        name=str(path.relative_to(root));require(bindings.get(name)==digest(path),'Required original proof binding missing/changed: '+name)
    for name,expected in manifest['code_sha256'].items():
        require(bindings.get(name)==expected and digest(root/name)==expected,'Original complete neural recipe/helper changed: '+name)
    for name,expected in bindings.items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts,'External scientific source binding refused')
        if name.endswith('.py') or name.startswith((REPORT+'/',RUN+'/',DATA+'/', 'artifacts/upstream-complete-trajectory-data/')):
            require(digest(root/path)==expected,'Original publication source/model/data changed: '+name)
    require(read(report/'fit-completion.json')['state']=='complete_all_five_original_budget_complete_trajectory_neural_fits',
        'All original full fits required before publication')
    for row in records:
        path=root/row['receipt_path'];receipt=read(path)
        require(path==report/'nn'/('%d.json'%row['fold']) and digest(path)==row['receipt_sha256'] and
            receipt['manifest_sha256']==digest(manifest_path) and receipt['fold']==row['fold'] and
            receipt['canonical_rows']==row['rows'] and len(receipt['chosen_iterations'])==3 and
            all(type(c) is int and 1<=c<=40 for c in receipt['chosen_iterations']) and
            receipt['outputs']==row['outputs'] and set(row['outputs'])==
                {'ridge.npz','scalers.npz','prediction.f32','nn-2026.pt','nn-7.pt','nn-13.pt'},
            'Every actual original fold/seed budget and receipt required')
        for name,expected in row['outputs'].items():
            require(digest(run/'nn'/str(row['fold'])/name)==expected,'Actual saved model/scaler/prediction changed')
    require(len(manifest['controls'])==1 and manifest['controls'][0]['id']=='tree154','Original fixed tree154 required')
    tree=manifest['controls'][0]['prediction']
    require(digest(root/tree)==bindings[tree],'Original tree154 control changed')
    for name,expected in [('nn.f32',model['nn_sha256']),('fixed-half-tree154.f32',current['prediction_sha256'])]:
        require(digest(run/name)==expected,'Complete canonical stored neural/half vector changed')
