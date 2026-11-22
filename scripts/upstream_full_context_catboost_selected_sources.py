"""Read-only full known-selection and qualifying-source matching component.

No query inference, fitting, output publication, queue or outcome audit occurs.
Qualifying operator/data replay and actual producer exit still belong to the
future owned bag pipeline. The two known arms must already be fully complete.
"""
import hashlib
from pathlib import Path
import upstream_full_context_catboost_execution_v1 as owner
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer
import upstream_full_context_catboost_canonical as canonical
import verify_upstream_full_context_catboost_study as replay


def merge(target,values):
    for name,digest in values.items():
        if name in target and target[name]!=digest:raise ValueError('Conflicting original source/artifact binding')
        target[name]=digest


def json_record(path,digest,bindings,guard):
    relative=str(path.relative_to(study.ROOT));merge(bindings,{relative:digest})
    return consumer.verified_json(path,digest,study.ROOT,guard)


def load(selection_sha256,guard):
    consumer.no_torch();root=study.ROOT;bindings={}
    if not callable(guard):raise ValueError('External owned whole-family guard required')
    plan_path=owner.REPORT/'plan.json';plan=owner.read(plan_path);owner.check_sources(plan)
    plan_sha=study.sha(plan_path);merge(bindings,plan['source_sha256']);merge(bindings,{str(plan_path.relative_to(root)):plan_sha})
    baseline=owner.REPORT/'first-scientific-baseline.json'
    json_record(baseline,study.BASELINE_SHA,bindings,guard)
    done_path=owner.REPORT/'completion.json';done=json_record(done_path,study.sha(done_path),bindings,guard)
    if (done['state']!=owner.COMPLETE or done['verification_sha256']!=selection_sha256 or done['qualifying_labels_read'] is not False):
        raise ValueError('Actual complete owned two-arm tree studies and independent selection required')
    phase_files={str((owner.RESULTS/(p+'.json')).relative_to(root)) for p in owner.PHASES}
    peak_files={str((owner.REPORT/(p+'-resource-peaks.json')).relative_to(root)) for p in owner.PHASES}
    completed=done['completed_phase_and_observed_resource_proof_sha256']
    if set(completed)!=phase_files|peak_files:raise ValueError('All nine original scientific and observed resource receipts required')
    for phase in owner.PHASES:
        path=owner.RESULTS/(phase+'.json');value=json_record(path,completed[str(path.relative_to(root))],bindings,guard)
        owner.validate_result(phase,value,plan_sha)
        path=owner.REPORT/(phase+'-resource-peaks.json');value=json_record(path,completed[str(path.relative_to(root))],bindings,guard)
        if (value['state']!='complete_observed_native_original_family_resource_samples' or value['phase']!=phase
                or value['continuous_peak_claimed'] is not False or value['original_baseline_sha256']!=study.BASELINE_SHA):
            raise ValueError('Original observed native sample scope and unchanged baseline required')
    path=root/'experiments/upstream-full-context-catboost-selection/selection.json'
    selection=json_record(path,selection_sha256,bindings,guard)
    if (selection['state']!=canonical.SELECTION_STATE or set(selection['canonical_sha256'])!=set(study.ARMS)
            or selection['selection_uses_only_complete_known_CV'] is not True or selection['qualifying_labels_read'] is not False
            or selection['tie_tolerance']!=1e-12 or selection['tie_rule']!='scalar'
            or selection['production_study_admitted'] is not False):
        raise ValueError('Both complete original known-only canonical arms and original tie rule required')
    arms={};scores={}
    for arm in study.ARMS:
        run,report=study.paths(arm);path=report/'independent-canonical-verification.json'
        c=json_record(path,selection['canonical_sha256'][arm],bindings,guard)
        if (c['state']!=canonical.STATE or c['arm']!=arm or c['rows']!=study.ROWS
                or c['qualifying_labels_read'] is not False or c['score_is_complete_five_user_fold_CV'] is not True
                or c['canonical_code_sha256']!=canonical.code_bindings()):
            raise ValueError('Separate complete original compensated known CV proof required')
        m,metadata,contract,fit,folds,outputs=replay.bindings(arm,c['manifest_sha256'],c['fit_completion_sha256'],guard)
        merge(bindings,outputs);merge(bindings,c['canonical_code_sha256'])
        for row in m['outputs'].values():merge(bindings,{str(Path(row['path']).relative_to(root)):row['sha256']})
        for row in m['authorized_prepare_bindings'].values():merge(bindings,{str(Path(row['path']).relative_to(root)):row['sha256']})
        for kind in ('fit','replay'):
            path=report/(kind+'-pipeline-completion.json');pipe=json_record(path,study.sha(path),bindings,guard)
            science=report/('fit-completion.json' if kind=='fit' else 'independent-verification.json')
            if (pipe['state']!='complete_actual_isolated_full_context_tree_pipeline_component' or pipe['mode']!=kind
                    or pipe['arm']!=arm or pipe['manifest_sha256']!=c['manifest_sha256']
                    or pipe['scientific_proof_sha256']!=study.sha(science)
                    or pipe['producer_reaped_zero_before_any_tree_fit_or_saved_model_replay'] is not True
                    or pipe['qualifying_labels_read'] is not False):
                raise ValueError('Actual complete original fit/replay producer pipeline required')
            exit_proof=json_record(report/(kind+'-producer-exit.json'),pipe['actual_producer_exit_sha256'],bindings,guard)
            if (exit_proof['state']!='complete_owned_original_tree_producer_zero_exit' or exit_proof['exit_code']!=0
                    or exit_proof['observed_by_actual_parent_waitpid'] is not True
                    or exit_proof['source_admission_sha256']!=m['source_metadata_sha256']
                    or exit_proof['contract_sha256']!=hashlib.sha256(study.wire.encoded(contract)).hexdigest()
                    or exit_proof['qualifying_labels_read'] is not False):
                raise ValueError('Actual successful original source exit and exact stream binding required')
            json_record(report/(kind+'-producer-admission.json'),pipe['producer_admission_sha256'],bindings,guard)
            external=pipe['external_authorization'];json_record(Path(external['path']),external['sha256'],bindings,guard)
        p=json_record(report/'independent-verification.json',c['independent_replay_sha256'],bindings,guard)
        prediction=str((run/'prediction.npy').relative_to(root))
        if (p['state']!=replay.STATE or p['arm']!=arm or p['rows']!=study.ROWS or p['qualifying_rows']!=study.QUALIFYING_ROWS
                or p['manifest_sha256']!=c['manifest_sha256'] or p['fit_completion_sha256']!=c['fit_completion_sha256']
                or p['original_seed_slots']!=15 or p['ridge_equations']!=10 or p['qualifying_labels_read'] is not False
                or any(p[key] is not True for key in ('every_original_raw_population_and_held_prediction_bit_verified',
                    'every_full_OOF_join_bit_verified','every_eligible_ridge_equation_and_coefficient_bit_verified'))
                or p['verifier_code_sha256']!=replay.code_bindings()
                or any(p['output_sha256'].get(k)!=v for k,v in outputs.items())
                or p['output_sha256'][prediction]!=c['prediction_sha256']):
            raise ValueError('Complete original full-stream/ridge/saved-model/OOF replay closure required')
        merge(bindings,p['output_sha256']);merge(bindings,p['verifier_code_sha256'])
        known={str(Path(v['path']).relative_to(root)):v['sha256'] for v in metadata['known_arrays'].values()}
        merge(bindings,known)
        # Recheck exact arithmetic from the full immutable original vector and
        # known outcomes. No qualifying outcomes or query rows are opened.
        records=consumer.verify_files(root,{**known,prediction:c['prediction_sha256']},guard)
        with consumer.cache.mapped(run/'prediction.npy',records[prediction],'<f4',(study.ROWS,),guard) as values:
            with consumer.known_arrays(metadata,records,root,guard) as (labels,users,groups):
                score=canonical.score(values,labels)
        if score!=c['tree_only_CV_rmse'] or selection['tree_only_known_CV_scores'].get(arm)!=score:
            raise ValueError('Both independently rechecked original complete known scores must match selection')
        scores[arm]=score;arms[arm]=dict(manifest=m,metadata=metadata,contract=contract,folds=folds,canonical=c)
    selected=canonical.choose(scores)
    if selection['selected_arm']!=selected or selection['selected_known_prediction_sha256']!=arms[selected]['canonical']['prediction_sha256']:
        raise ValueError('Original known-only choice and matched exact known prediction vector required')
    records=consumer.verify_files(root,bindings,guard)
    return dict(arm=selected,selection=selection,selection_sha256=selection_sha256,arms=arms,
        source_sha256=bindings,records=records,production_data_admitted=False,qualifying_labels_read=False)


def qualifying_metadata(known,metadata_path,metadata_sha256,guard):
    """Match a separately produced Q stream to this exact selected known arm."""
    consumer.no_torch();root=study.ROOT
    q=consumer.verified_json(Path(metadata_path),metadata_sha256,root,guard)
    arm=known['arm'];source=known['arms'][arm];m=source['manifest'];original=source['metadata']
    original_sha=hashlib.sha256(study.wire.encoded(original)).hexdigest()
    code=root/'scripts/upstream_full_context_catboost_query_producer.py'
    expected_code={str(code.relative_to(root)):study.sha(code)}
    expected_bindings=dict(original['source_sha256']);merge(expected_bindings,expected_code)
    if (q['format']!='own_full_context_tree_original_qualifying_source_v1' or q['query']!='qual' or q['arm']!=arm
            or q['rows']!=study.QUALIFYING_ROWS or q['known_rows']!=study.ROWS
            or q['dimensions']!=m['dimensions'] or q['retain_scalar']!=original['retain_scalar']
            or q['original_known_source_metadata']!=original or q['original_known_source_metadata_sha256']!=original_sha
            or original_sha!=m['source_metadata_sha256'] or q['weights_sha256']!=original['weights_sha256']
            or q['query_code_sha256']!=expected_code or q['source_sha256']!=expected_bindings
            or q['models_fitted']!=0 or q['qualifying_labels_read'] is not False
            or q['selection_authorized'] is not False or q['production_data_admitted'] is not False):
        raise ValueError('Exact matched selected known/qualifying source, original coefficients and all columns required')
    parent=root/consumer.DATA/'manifest.json'
    cpu=consumer.verified_json(parent,original['parent_CPU_manifest_sha256'],root,guard)
    if cpu['virtual_c_order_sha256']['qual']['arm_operator_sha256'][arm]!=q['operator_sha256']:
        raise ValueError('Complete original independently verified selected qualifying operator hashes required')
    contract=study.wire.contract(study.QUALIFYING_ROWS,retain_scalar=q['retain_scalar'],latent=arm=='latent',
        weights_sha256=q['weights_sha256'],admission_sha256=hashlib.sha256(study.wire.encoded(q)).hexdigest(),operator_sha256=q['operator_sha256'])
    # Data bytes will be authenticated by the isolated full source producer;
    # original CPU/source proof and stream/exit replay remain mandatory.
    consumer.verify_files(root,expected_code,guard)
    return q,contract
