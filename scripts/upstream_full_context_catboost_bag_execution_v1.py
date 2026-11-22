"""Owned full selected-tree qualifying pair; no work starts on import.

Both complete known arms and their known-only choice precede every query phase.
Original whole-family resource supervision, lanes and same-frame preservation
are retained. This controller fits no models and reads no qualifying outcomes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_rbm400_neural_execution_reference_v4 as base
import upstream_full_context_catboost_execution_v1 as known_owner
import upstream_full_context_catboost_bag_pipeline_v1 as pipeline
import upstream_full_context_catboost_supervision as supervision

bag=pipeline.bag
study=bag.study
consumer=bag.consumer
selected=bag.selected
replay=pipeline.replay
ROOT=study.ROOT
REPORT=ROOT/'experiments/upstream-full-context-catboost-bag-execution-v1'
RESULTS=ROOT/'experiments/upstream-full-context-catboost-bag-execution-results-v1'
SOURCE_REPORT=ROOT/'experiments/upstream-full-context-catboost-bag-source-v1'
DECL=ROOT/'OWN_FULL_CONTEXT_CATBOOST_OWNED_BAG_PLAN_V1.md'
PROTOCOL=ROOT/'experiments/upstream-full-context-catboost-owned-bag-protocol-v1/verification.json'
WORKER=ROOT/'scripts/upstream_full_context_catboost_bag_worker_v1.py'
SELECTION=ROOT/'experiments/upstream-full-context-catboost-selection/selection.json'
PHASES=('prepare','infer','replay','admit')
ENV=dict(base.ENV)
FAMILY_LIMIT=16*2**30
OUTPUT_INVENTORY_MiB=dict(prediction_vectors=128,source_metadata_and_receipts=512,staging=64,general_reserve=64)
OUTPUT_RESERVE=sum(OUTPUT_INVENTORY_MiB.values())*2**20
PLAN_STATE='declared_owned_complete_selected_tree_query_pair_and_full_replay_v1'
COMPLETE='complete_owned_selected_tree_matched_known_qualifying_pair_and_full_independent_replay_v1'
PREPARED='prepared_original_selected_tree_qualifying_source_pending_owned_inference'
ADMITTED='complete_selected_tree_pair_source_models_vectors_and_owned_pipeline_checks'
read=base.read
stamp=base.stamp
status=base.status
identity=base.identity
family=supervision.family


def metadata_guard(extra_disk_bytes=0):
    return base.metadata_guard(extra_disk_bytes)


def save(path,value,guard=metadata_guard):
    return base.durable_payload(Path(path),lambda s:s.write((json.dumps(value,indent=2)+'\n').encode()),ROOT,guard)


def policy():
    return dict(family_bytes=FAMILY_LIMIT,total_physical_and_RSS_bytes=24*2**30,compressor_bytes=5*2**30,
        admission_compressor_bytes=int(4.5*2**30),maximum_swap_growth_bytes=128*2**20,
        disk_floor_bytes=int(18.75*2**30),full_remaining_output_reservation=OUTPUT_RESERVE,output_inventory_MiB=OUTPUT_INVENTORY_MiB,
        stable_samples=3,sample_spacing_seconds=10,both_original_lanes=True,same_process_group_frames=True,
        original_baseline_sha256=study.BASELINE_SHA)



def command(phase):
    if phase not in PHASES:raise ValueError('Exact original selected-pair phase required')
    return [sys.executable,'-u',str(WORKER),'--phase',phase]


def producer_command(phase,pid):
    if phase not in ('infer','replay') or type(pid) is not int or pid<=0:
        raise ValueError('Exact original query stream phase/root required')
    return command(phase)+['--source-root',str(pid)]


def check_sources(plan):
    if (plan['state']!=PLAN_STATE or plan['phases']!=list(PHASES) or plan['environment']!=ENV
            or plan['policy']!=policy() or plan['libraries']!=study.versions() or plan['python']!=sys.version
            or plan['known_rows']!=1128164 or plan['qualifying_rows']!=2817131
            or plan['fold_slots']!=5 or plan['seed_slots']!=15 or plan['block_rows']!=8192
            or plan['models_fitted']!=0 or plan['qualifying_labels_read'] is not False
            or plan['known_only_selection_required'] is not True):
        raise ValueError('Complete original selected-pair runtime, dimensions and resource policy required')
    for name,digest in plan['source_sha256'].items():
        if study.sha(ROOT/name)!=digest:raise ValueError('Frozen original selected-pair source changed: '+name)


def source_files():
    # Verify the actual saved known-owner snapshot. Never regenerate its glob
    # after downstream modules have been added.
    old=known_owner.REPORT/'plan.json';previous=read(old);known_owner.check_sources(previous)
    required={Path(__file__),WORKER,DECL,old}
    required.update(ROOT/p for p in previous['source_sha256'])
    required.update((ROOT/'scripts').glob('*upstream_full_context_catboost*.py'))
    for name in ('upstream-full-context-catboost-selected-bag-protocol-v1',
            'upstream-full-context-catboost-query-pipeline-protocol-v1'):
        path=ROOT/'experiments'/name/'verification.json';proof=read(path)
        for name,digest in proof['source_sha256'].items():
            if study.sha(ROOT/name)!=digest:raise ValueError('Original component review expectation changed: '+name)
            required.add(ROOT/name)
        required.add(path)
    return required


def frozen_plan():
    required=source_files();proof=read(PROTOCOL)
    if (proof['state']!='reviewed_owned_complete_selected_tree_query_pair_v1_pending_real_run'
            or proof['test_exit_code']!=0 or proof['test_count']<104
            or proof['production_source_admission_run'] is not False or proof['production_models_fitted']!=0
            or proof['qualifying_labels_read'] is not False or proof['review']!='manual_source_and_full_regression_complete'):
        raise ValueError('Reviewed complete original owned query controller required')
    for p in required:
        if proof['source_sha256'].get(str(p.relative_to(ROOT)))!=study.sha(p):
            raise ValueError('Owned selected-pair review source changed: '+str(p))
    required.add(PROTOCOL)
    return dict(state=PLAN_STATE,created=stamp(),phases=list(PHASES),environment=ENV,policy=policy(),
        libraries=study.versions(),python=sys.version,known_rows=1128164,qualifying_rows=2817131,
        fold_slots=5,seed_slots=15,block_rows=8192,models_fitted=0,qualifying_labels_read=False,
        known_only_selection_required=True,source_sha256={str(p.relative_to(ROOT)):study.sha(p) for p in sorted(required)})


def prerequisites():
    not_needed=known_owner.REPORT/'not-needed.json'
    if not_needed.exists():
        # Require the original complete own17 primary audit as well as the
        # original known-owner decision. No comparison candidate can skip work.
        previous=known_owner.prerequisites()
        if previous['start_followup'] is not False or read(not_needed)['state']!='own17_primary_already_meets_original_target_no_tree_study_started':
            raise ValueError('Original full primary target audit required before skipping tree pair')
        bindings=dict(previous['source_sha256']);selected.merge(bindings,{str(not_needed.relative_to(ROOT)):study.sha(not_needed)})
        return dict(start_followup=False,source_sha256=bindings,qualifying_outcomes_read=False)
    known=selected.load(study.sha(SELECTION),metadata_guard)
    return dict(start_followup=True,selection_sha256=known['selection_sha256'],selected_arm=known['arm'],
        source_sha256=known['source_sha256'],qualifying_outcomes_read=False,
        condition='both_complete_known_tree_arms_replayed_and_known_only_selected')


def query_inputs(guard):
    pre=read(REPORT/'prerequisites.json');selection_sha=pre['selection_sha256']
    known=selected.load(selection_sha,guard)
    path=SOURCE_REPORT/'metadata.json';record=read(SOURCE_REPORT/'preparation.json')
    if (record['state']!=PREPARED or record['selection_sha256']!=selection_sha or record['arm']!=known['arm']
            or record['metadata_path']!=str(path) or record['metadata_sha256']!=study.sha(path)
            or record['qualifying_labels_read'] is not False or record['models_fitted']!=0):
        raise ValueError('Complete original selected qualifying source preparation required')
    metadata,expected=selected.qualifying_metadata(known,path,record['metadata_sha256'],guard)
    if record['logical_source_admission_sha256']!=expected['admission_sha256']:
        raise ValueError('Original logical source-admission and metadata-file hashes must remain distinct')
    return known,path,record['metadata_sha256'],metadata,expected


def validate_source_request(phase,captured,request,guard):
    if phase not in ('infer','replay'):raise ValueError('Only full inference/replay admits a query source')
    known,path,digest,metadata,expected=query_inputs(guard)
    actual=identity.process(request['producer']['pid'])
    completion=study.sha(bag.paths()[1]/'completion.json') if phase=='replay' else None
    if (request['mode']!=phase or request['arm']!=known['arm']
            or request['command']!=producer_command(phase,captured['pid'])
            or not family.matches(captured,request['scientific_root'],captured['pid'])
            or not same_role(actual,request['producer']) or actual['ppid']!=captured['pid'] or actual['pgid']!=captured['pid']
            or request['selection_sha256']!=known['selection_sha256'] or request['completion_sha256']!=completion
            or request['query_metadata_path']!=str(path) or request['query_metadata_sha256']!=digest
            or request['source_metadata_sha256']!=expected['admission_sha256']
            or request['contract_sha256']!=hashlib.sha256(study.wire.encoded(expected)).hexdigest()
            or request['qualifying_labels_read'] is not False):
        raise ValueError('Exact original query source command/lifetime/selection/metadata required')
    existing=roles(phase,captured)
    if not same_role(existing.get('producer'),request['producer']):raise ValueError('Actual original early query-source capture required')
    bindings=dict(known['source_sha256']);selected.merge(bindings,metadata['source_sha256'])
    selected.merge(bindings,{str(path.relative_to(ROOT)):digest})
    selected.merge(bindings,{str(p.relative_to(ROOT)):study.sha(p) for p in [*pipeline.code_paths(),WORKER]})
    if phase=='replay':selected.merge(bindings,{str((bag.paths()[1]/'completion.json').relative_to(ROOT)):completion})
    if request['source_sha256']!=bindings:raise ValueError('Exact complete original query source closure required')
    return existing


def admit_source(phase,captured,request,guard):
    if phase not in ('infer','replay'):raise ValueError('Only full inference/replay admits a query source')
    authorize(phase)
    # Popen/capture can beat the fresh child's immutable bootstrap write.
    # Wait in this same frame while the exact captured source remains live;
    # an observation delay must not become a failed/restarted experiment.
    while not (REPORT/(phase+'-source-bootstrap.json')).exists():
        guard();actual=identity.process(request['producer']['pid'])
        if (not same_role(request['producer'],actual) or actual['state'].startswith('Z')
                or actual['ppid']!=captured['pid'] or actual['pgid']!=captured['pid']):
            raise ValueError('Original live source required while waiting for its bootstrap registration')
        time.sleep(.05)
    existing=validate_source_request(phase,captured,request,guard)
    save(REPORT/(phase+'-roles.json'),dict(scientific_root_pid=captured['pid'],roles=existing),guard)
    path=REPORT/(phase+'-source-authorization.json')
    save(path,dict(state='authorized_original_full_context_tree_query_producer_role',request=request),guard)
    return dict(path=str(path),sha256=study.sha(path))


def scientific_path(phase):
    if phase=='prepare':return SOURCE_REPORT/'preparation.json',PREPARED
    if phase=='infer':return bag.paths()[1]/'completion.json',bag.STATE
    if phase=='replay':return bag.paths()[1]/'independent-verification.json',replay.STATE
    if phase=='admit':return SOURCE_REPORT/'pair-admission.json',ADMITTED
    raise ValueError('Exact complete pair phase required')


def validate_result(phase,value,plan_sha):
    path,state=scientific_path(phase);actual=read(path)
    if (value['state']!='complete_owned_original_tree_pair_'+phase or value['execution_plan_sha256']!=plan_sha
            or value['scientific_proof_sha256']!=study.sha(path) or value['qualifying_labels_read'] is not False
            or actual['state']!=state or actual['qualifying_labels_read'] is not False or actual['models_fitted']!=0):
        raise ValueError('Exact complete original selected-pair scientific result required')
    if phase in ('infer','replay'):
        if (actual.get('known_rows')!=study.ROWS or actual.get('qualifying_rows')!=study.QUALIFYING_ROWS):
            raise ValueError('Both full original populations required')
        pipe=read(pipeline.paths()/(phase+'-pipeline-completion.json'))
        if pipe['scientific_proof_sha256']!=study.sha(path) or pipe['producer_reaped_zero_before_any_pair_publication'] is not True:
            raise ValueError('Original actual successful query pipeline required')
    if phase=='infer' and (actual['original_fold_slots']!=5 or actual['original_seed_slots']!=15
            or actual['every_original_qualifying_row_inferred'] is not True):raise ValueError('Every original fold/seed/query row required')
    if phase=='replay' and (actual['fold_slots']!=5 or actual['seed_slots']!=15 or actual['ridge_equations']!=10
            or actual['every_known_and_all_five_qualifying_fold_and_bag_bit_verified'] is not True
            or actual['every_original_full_raw_stream_and_baseline_bit_replayed'] is not True):
        raise ValueError('Independent complete original raw/ridge/model/vector replay required')


def admit_pair(guard):
    """Authenticate the full pair and both actual producer pipelines.

    This proof still requires all four owned phase/resource receipts and the
    external owner's terminal-family completion before an own18 provider can
    admit the pair. No fitting, score or qualifying outcome is consumed here.
    """
    consumer.no_torch();known,path,digest,metadata,expected=query_inputs(guard)
    art,report=bag.paths();bindings=dict(known['source_sha256'])
    def record(path,digest=None):
        digest=digest or study.sha(path)
        selected.merge(bindings,{str(path.relative_to(ROOT)):digest})
        return consumer.verified_json(path,digest,ROOT,guard)
    selected.merge(bindings,metadata['source_sha256']);record(path,digest)
    plan=record(REPORT/'plan.json');check_sources(plan);selected.merge(bindings,plan['source_sha256'])
    plan_sha=study.sha(REPORT/'plan.json')
    for phase in ('prepare','infer','replay'):
        value=record(RESULTS/(phase+'.json'));validate_result(phase,value,plan_sha)
        peak=record(REPORT/(phase+'-resource-peaks.json'))
        if (peak['state']!='complete_observed_native_original_family_resource_samples' or peak['phase']!=phase
                or peak['continuous_peak_claimed'] is not False or peak['original_baseline_sha256']!=study.BASELINE_SHA):
            raise ValueError('Original observed resource sample scope and baseline required')
    complete=record(report/'completion.json');completion_sha=study.sha(report/'completion.json')
    proof=record(report/'independent-verification.json')
    if (complete['state']!=bag.STATE or complete['arm']!=known['arm']
            or complete['selection_sha256']!=known['selection_sha256']
            or complete['query_metadata_path']!=str(path) or complete['query_metadata_sha256']!=digest
            or complete['query_source_admission_sha256']!=expected['admission_sha256']
            or complete['zero_slots_retained_in_denominator'] is not True
            or proof['state']!=replay.STATE or proof['arm']!=known['arm']
            or proof['selection_sha256']!=known['selection_sha256'] or proof['completion_sha256']!=completion_sha
            or proof['verifier_code_sha256']!=replay.code_bindings()
            or proof['independent_arithmetic_does_not_call_production_predict'] is not True
            or proof['actual_independent_producer_zero_exit_verified'] is not True
            or complete['production_data_admitted'] is not False or proof['production_data_admitted'] is not False
            or set(complete['outputs'])!={'known','qualifying',*('fold-'+str(i) for i in range(5))}):
        raise ValueError('Full original independently replayed matched selected-tree pair required')
    selected.merge(bindings,complete['source_sha256']);selected.merge(bindings,proof['source_sha256'])
    selected.merge(bindings,proof['verifier_code_sha256'])
    for name,row in complete['outputs'].items():
        output=art/('known-prediction.f32' if name=='known' else 'prediction.f32' if name=='qualifying' else name+'.f32')
        rows=study.ROWS if name=='known' else study.QUALIFYING_ROWS
        relative=str(output.relative_to(ROOT))
        if (Path(row['path'])!=output or row['bytes']!=4*rows
                or proof['source_sha256'].get(relative)!=row['sha256']):
            raise ValueError('Exact complete independently verified original prediction vectors required')
        selected.merge(bindings,{relative:row['sha256']})
    producers=[]
    for phase in ('infer','replay'):
        pipe=record(pipeline.paths()/(phase+'-pipeline-completion.json'))
        scientific=report/('completion.json' if phase=='infer' else 'independent-verification.json')
        captured=record(REPORT/(phase+'-ownership.json'))
        stored_roles=record(REPORT/(phase+'-roles.json'))
        producer=stored_roles['roles']['producer'];source=pipe['producer'];root=pipe['scientific_root']
        if (pipe['state']!='complete_actual_isolated_selected_tree_query_pipeline_component'
                or pipe['mode']!=phase or pipe['arm']!=known['arm']
                or pipe['selection_sha256']!=known['selection_sha256']
                or pipe['completion_sha256']!=(completion_sha if phase=='replay' else None)
                or pipe['query_metadata_sha256']!=digest
                or pipe['source_metadata_sha256']!=expected['admission_sha256']
                or pipe['scientific_proof_sha256']!=study.sha(scientific)
                or pipe['producer_reaped_zero_before_any_pair_publication'] is not True
                or pipe['production_data_admitted'] is not False or pipe['qualifying_labels_read'] is not False
                or stored_roles['scientific_root_pid']!=captured['pid']
                or not family.matches(captured,root,captured['pid']) or not same_role(producer,source)
                or source['ppid']!=root['pid'] or source['pgid']!=root['pid']):
            raise ValueError('Both actual original owned root/source lifetimes and full pipelines required')
        external=pipe['external_authorization'];authorization=record(Path(external['path']),external['sha256'])
        request=authorization['request']
        original_bindings=dict(known['source_sha256']);selected.merge(original_bindings,metadata['source_sha256'])
        selected.merge(original_bindings,{str(path.relative_to(ROOT)):digest})
        selected.merge(original_bindings,{str(p.relative_to(ROOT)):study.sha(p) for p in [*pipeline.code_paths(),WORKER]})
        if phase=='replay':selected.merge(original_bindings,{str((report/'completion.json').relative_to(ROOT)):completion_sha})
        exact_request=dict(mode=phase,arm=known['arm'],command=producer_command(phase,captured['pid']),
            scientific_root=root,producer=source,selection_sha256=known['selection_sha256'],
            completion_sha256=completion_sha if phase=='replay' else None,query_metadata_path=str(path),query_metadata_sha256=digest,
            source_metadata_sha256=expected['admission_sha256'],
            contract_sha256=hashlib.sha256(study.wire.encoded(expected)).hexdigest(),
            source_sha256=original_bindings,qualifying_labels_read=False)
        admission_path=pipeline.paths()/(phase+'-producer-admission.json')
        admission=record(admission_path,pipe['producer_admission_sha256'])
        if (authorization['state']!='authorized_original_full_context_tree_query_producer_role' or request!=exact_request
                or Path(external['path'])!=REPORT/(phase+'-source-authorization.json')
                or admission!=dict(request=exact_request,external_authorization=external,
                    production_data_admitted=False,qualifying_labels_read=False)):
            raise ValueError('Exact original source authorization and immutable admission closure required')
        closure=dict(original_bindings)
        selected.merge(closure,{str(Path(external['path']).relative_to(ROOT)):external['sha256'],
            str(admission_path.relative_to(ROOT)):pipe['producer_admission_sha256']})
        exit_path=pipeline.paths()/(phase+'-producer-exit.json')
        exit_proof=record(exit_path,pipe['actual_producer_exit_sha256'])
        consumer.producer_exit(exit_path,pipe['actual_producer_exit_sha256'],source,expected,expected['admission_sha256'],ROOT,guard)
        if (pipe['source_sha256']!=closure or exit_proof['source_sha256']!=closure
                or not same_role(root,exit_proof['owner']) or exit_proof['scientific_group']!=root['pid']
                or proof['source_sha256'].get(str(exit_path.relative_to(ROOT)))!=pipe['actual_producer_exit_sha256']):
            raise ValueError('Exact source/authorization closure and independently checked actual source exits required')
        if phase=='infer':
            expected_exit=dict(path=str(exit_path),sha256=pipe['actual_producer_exit_sha256'],producer=source)
            if complete['actual_producer_exit_receipt']!=expected_exit:
                raise ValueError('Original inference source exit must match its saved pair')
        producers.append(source)
        selected.merge(bindings,closure)
    if same_role(producers[0],producers[1]):raise ValueError('Independent replay requires a fresh actual source lifetime')
    for p in (REPORT/'monitor.json',REPORT/'first-scientific-baseline.json',REPORT/'prerequisites.json',
            REPORT/'admit-gate.json',REPORT/'admit-ownership.json',SOURCE_REPORT/'preparation.json'):
        record(p,study.BASELINE_SHA if p.name=='first-scientific-baseline.json' else None)
    records=consumer.verify_files(ROOT,bindings,guard)
    value=dict(state=ADMITTED,arm=known['arm'],selection_sha256=known['selection_sha256'],
        completion_sha256=completion_sha,independent_replay_sha256=study.sha(report/'independent-verification.json'),
        known_rows=study.ROWS,qualifying_rows=study.QUALIFYING_ROWS,fold_slots=5,seed_slots=15,
        outputs={name:complete['outputs'][name] for name in ('known','qualifying')},
        every_original_raw_ridge_model_fold_bag_and_matched_known_bit_independently_verified=True,
        both_actual_owned_source_exit_and_authorization_closures_verified=True,
        source_sha256=bindings,models_fitted=0,qualifying_labels_read=False,production_data_admitted=False,
        remaining='External owner must bind all four phase/resource receipts and terminal family; full own18 calibration/audits remain')
    if consumer.verify_files(ROOT,bindings,guard)!=records:raise ValueError('Every original pair input/output identity must survive final checks')
    study.save_json(SOURCE_REPORT/'pair-admission.json',value,guard);return value


def roles(phase,captured):
    result={}
    for name in (phase+'-source-bootstrap.json',phase+'-roles.json'):
        path=REPORT/name
        if not path.exists():continue
        value=read(path)
        if value['scientific_root_pid']!=captured['pid']:raise ValueError('Same original root role table required')
        for role,row in value['roles'].items():
            if role in result and result[role]!=row:raise ValueError('Original producer lifetime cannot be replaced')
            result[role]=row
    return result


def phase_check(phase,captured):
    plan=read(REPORT/'plan.json');check_sources(plan)
    gate=read(REPORT/(phase+'-gate.json'));owner=read(REPORT/'monitor.json')
    if (gate['phase']!=phase or gate['pid']!=captured['pid'] or gate['command']!=command(phase)
            or gate['execution_plan_sha256']!=study.sha(REPORT/'plan.json')
            or gate['owner_sha256']!=study.sha(REPORT/'monitor.json') or gate['baseline_sha256']!=study.BASELINE_SHA
            or not identity.same_owner(owner,identity.process(owner['pid']))
            or not identity.same_owner(gate['watchdog'],identity.process(gate['watchdog']['pid']))):
        raise ValueError('Same live original owner/watchdog/source/phase gate required')
    for table in (read(REPORT/'prerequisites.json')['source_sha256'],gate['predecessor_sha256']):
        for name,digest in table.items():
            if study.sha(ROOT/name)!=digest:raise ValueError('Original completed prerequisite/predecessor changed')
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('Original environment required before numerical imports')
    return gate,owner


def control(phase,captured,role='root'):
    gate,owner=phase_check(phase,captured)
    return supervision.Control(captured,lambda:roles(phase,captured),repo=ROOT,baseline_path=REPORT/'first-scientific-baseline.json',
        owner=owner,watchdog=gate['watchdog'],check_sources=lambda:phase_check(phase,captured),remaining_output=lambda:OUTPUT_RESERVE,
        emergency_reserve=OUTPUT_RESERVE,request_path=REPORT/(phase+'-'+role+'-resource-request.json'))


def authorize(phase):
    if phase not in PHASES:raise ValueError('Declared original phase required')
    captured=read(REPORT/(phase+'-ownership.json'));phase_check(phase,captured)
    actual=identity.process(os.getpid())
    if not family.matches(captured,actual,captured['pid']) or actual['command']!=' '.join(command(phase)):
        raise ValueError('Actual original dedicated numerical root required')
    return captured


def bootstrap(phase,owner_pid):
    plan=read(REPORT/'plan.json');check_sources(plan);owner=read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Original actual owner required')
    captured=identity.process(os.getpid());captured['worker_command']=' '.join(command(phase))
    if captured['pid']!=captured['pgid']:raise ValueError('Dedicated original scientific root group required')
    save(REPORT/(phase+'-ownership.json'),captured)
    status(REPORT/'child-ownership.json',dict(phase=phase,root=captured))
    path=REPORT/(phase+'-gate.json')
    while True:
        if not identity.same_owner(owner,identity.process(owner_pid)):
            family.stop_family(captured,roles(phase,captured));continue
        if path.exists():
            phase_check(phase,captured)
            env=os.environ.copy();env.update(ENV);os.execvpe(sys.executable,command(phase),env)
        time.sleep(.1)


def same_role(left,right):
    return bool(left and right and all(left[k]==right[k] for k in ('pid','ppid','pgid','started','command')))


def source_bootstrap(phase,root_pid):
    captured=read(REPORT/(phase+'-ownership.json'));phase_check(phase,captured)
    actual=identity.process(os.getpid());parent=identity.process(root_pid)
    if (captured['pid']!=root_pid or not family.matches(captured,parent,root_pid)
            or actual['ppid']!=root_pid or actual['pgid']!=root_pid
            or actual['command']!=' '.join(producer_command(phase,root_pid))):
        raise ValueError('Actual original root/source bootstrap identities required')
    save(REPORT/(phase+'-source-bootstrap.json'),dict(scientific_root_pid=root_pid,roles={'producer':actual}))
    if sys.stdin.buffer.read(1)!=b'G':
        family.stop_family(captured,{'producer':actual});raise RuntimeError('Source authorization never released')
    phase_check(phase,captured)
    if not (REPORT/(phase+'-roles.json')).exists() or not same_role(roles(phase,captured).get('producer'),actual):
        raise ValueError('Registered same original source required')
    auth=read(REPORT/(phase+'-source-authorization.json'))
    if (auth.get('state')!='authorized_original_full_context_tree_query_producer_role'
            or not same_role(auth['request']['producer'],actual) or auth['request']['scientific_root']['pid']!=root_pid):
        raise ValueError('Actual original producer request authorization required')
    validate_source_request(phase,captured,auth['request'],metadata_guard)
    return captured


def active_family():
    path=REPORT/'child-ownership.json'
    if not path.exists():return None
    value=read(path);return value['root'],roles(value['phase'],value['root'])


def watchdog(owner_pid,fds):
    owner=read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Original watchdog owner required')
    supervision.watch(owner,active_family,lambda:(REPORT/'owner-release.json').exists(),fds,
        record=lambda v:status(REPORT/'watchdog-status.json',v))


def observe_peaks(peaks,sample):
    values=dict(family_physical_bytes=sample['family_usage']['bytes'],family_RSS_bytes=sample['family_usage']['rss_bytes'],
        aggregate_physical_bytes=sum(v['bytes'] for v in sample['workers'].values()),
        aggregate_RSS_bytes=sum(v['rss_bytes'] for v in sample['workers'].values()))
    if any(type(v) is not int or v<0 for v in values.values()):raise ValueError('Actual nonnegative native domain samples required')
    for key,value in values.items():peaks[key]=max(peaks.get(key,0),value)
    peaks['samples']=peaks.get('samples',0)+1
    return peaks


def main():
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('Original environment required before declaring owner')
    plan=frozen_plan();check_sources(plan)
    roots=[REPORT,RESULTS,SOURCE_REPORT,pipeline.paths(),*bag.paths()]
    if any(p.exists() for p in roots):raise ValueError('Existing full/partial owned tree namespace preserved; no retry')
    base.durable_directory(REPORT,ROOT,metadata_guard);save(REPORT/'plan.json',plan);save(REPORT/'monitor.json',identity.process(os.getpid()))
    base.recovery.copy_baseline(REPORT/'first-scientific-baseline.json')
    primary=(ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a');gpu=(ROOT/'experiments/upstream-source-extension/run.lock').open('a')
    child=captured=watch=None;completed={}
    def check():
        check_sources(plan)
        if watch is not None and watch.poll() is not None:raise RuntimeError('Original watchdog unavailable')
    def interrupted(signum,frame):raise KeyboardInterrupt('Original tree owner interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--watch',str(os.getpid()),'--keep-fds',str(primary.fileno()),str(gpu.fileno())],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(primary.fileno(),gpu.fileno()))
            while True:
                check()
                try:pre=prerequisites();break
                except (FileNotFoundError,json.JSONDecodeError):
                    status(REPORT/'status.json',dict(state='waiting_both_complete_tree_arms_and_known_only_selection',observed=stamp()));time.sleep(10)
            save(REPORT/'prerequisites.json',pre)
            if not pre['start_followup']:
                save(REPORT/'not-needed.json',dict(state='own17_primary_already_meets_original_target_no_tree_pair_started'))
                save(REPORT/'owner-release.json',dict(state='no_scientific_root_started'));return
            while True:
                check();got=False
                try:
                    fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB);got=True;fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if got:fcntl.flock(primary,fcntl.LOCK_UN)
                    status(REPORT/'status.json',dict(state='waiting_both_original_lanes',observed=stamp()));time.sleep(10)
            for phase in PHASES:
                peaks={};good=0
                while good<3:
                    check()
                    try:ok=base.ready(base.resources.footprints(os.getpid()),base.memory.memory(),read(REPORT/'first-scientific-baseline.json'),
                        base.shutil.disk_usage(ROOT).free,FAMILY_LIMIT,OUTPUT_RESERVE)
                    except Exception:ok=False
                    good=good+1 if ok else 0
                    status(REPORT/'status.json',dict(state='waiting_original_full_family_headroom',phase=phase,stable_samples=good,observed=stamp()))
                    if good<3:time.sleep(10)
                with (REPORT/(phase+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--bootstrap',str(os.getpid()),'--phase',phase],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**ENV))
                    ownership=REPORT/(phase+'-ownership.json')
                    while not ownership.exists():
                        check()
                        if child.poll() is not None:raise RuntimeError('Original tree bootstrap failed')
                        time.sleep(.1)
                    captured=read(ownership)
                    if captured['pid']!=child.pid or not family.matches(captured,identity.process(child.pid),child.pid):raise ValueError('Actual original bootstrap required')
                    save(REPORT/(phase+'-gate.json'),dict(pid=child.pid,phase=phase,command=command(phase),execution_plan_sha256=study.sha(REPORT/'plan.json'),
                        owner_sha256=study.sha(REPORT/'monitor.json'),baseline_sha256=study.BASELINE_SHA,watchdog=identity.process(watch.pid),predecessor_sha256=completed))
                    c=control(phase,captured)
                    while child.poll() is None:
                        check()
                        try:
                            allowed,sample=c.sample();actual=identity.process(child.pid);observe_peaks(peaks,sample)
                            status(REPORT/(phase+'-resource-sample.json'),dict(observed=stamp(),observed_peaks=peaks,**sample))
                        except Exception:allowed=False;actual=identity.process(child.pid)
                        if child.poll() is not None:break
                        if not allowed or (actual and 'T' in actual['state']):
                            request=REPORT/(phase+'-root-resource-request.json')
                            if request.exists() and 'original_source_pipeline_failure' in read(request).get('reason',{}):
                                raise RuntimeError('Original failed source pipeline remains preserved without retry')
                            c.stop(dict(supervisor_policy_or_preservation_request=True));c.wait_then_resume(OUTPUT_RESERVE)
                        status(REPORT/'status.json',dict(state=phase,pid=child.pid,observed=stamp()));time.sleep(3)
                    if child.returncode!=0:raise RuntimeError('Original scientific phase failed without retry: '+phase)
                if family.live_roles(captured,roles(phase,captured)):raise RuntimeError('Original scientific source remains live after root exit')
                value=read(RESULTS/(phase+'.json'));validate_result(phase,value,study.sha(REPORT/'plan.json'))
                completed[str((RESULTS/(phase+'.json')).relative_to(ROOT))]=study.sha(RESULTS/(phase+'.json'))
                peak_path=REPORT/(phase+'-resource-peaks.json')
                save(peak_path,dict(state='complete_observed_native_original_family_resource_samples',phase=phase,observed_peaks=peaks,
                    nominal_sample_spacing_seconds=3,continuous_peak_claimed=False,original_baseline_sha256=study.BASELINE_SHA))
                completed[str(peak_path.relative_to(ROOT))]=study.sha(peak_path)
                captured=child=None
            final=SOURCE_REPORT/'pair-admission.json'
            save(REPORT/'completion.json',dict(state=COMPLETE,verification_sha256=study.sha(final),qualifying_labels_read=False,
                selection_sha256=pre['selection_sha256'],selected_arm=pre['selected_arm'],
                known_rows=study.ROWS,qualifying_rows=study.QUALIFYING_ROWS,models_fitted=0,production_pair_admitted=True,
                completed_phase_and_observed_resource_proof_sha256=completed,
                remaining='Full own18 calibration, independent preaudit and fixed recovered canonical audits required; not goal completion'))
            save(REPORT/'owner-release.json',dict(state='all_original_scientific_roles_terminal_after_complete_selected_pair_replay'))
    except BaseException as error:
        try:
            current=active_family();alive=bool(current and family.live_roles(*current))
            if alive:family.stop_family(*current)
            status(REPORT/'failure.json',dict(error=repr(error),actual_family_preserved=alive,no_automatic_retry=True,observed=stamp()))
            if not alive:save(REPORT/'owner-release.json',dict(state='no_live_original_scientific_roles_at_failure'))
        except Exception:pass  # Watchdog keeps locks on uncertain role metadata.
        raise
    finally:primary.close();gpu.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true')
    parser.add_argument('--watch',type=int);parser.add_argument('--keep-fds',type=int,nargs='*',default=[])
    parser.add_argument('--bootstrap',type=int);parser.add_argument('--phase',choices=PHASES);args=parser.parse_args()
    if args.watch:watchdog(args.watch,args.keep_fds)
    elif args.bootstrap:bootstrap(args.phase,args.bootstrap)
    elif args.start:main()

