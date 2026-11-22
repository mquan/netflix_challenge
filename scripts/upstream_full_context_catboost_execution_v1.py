"""Owned full-budget two-arm tree studies; explicit start only after review.

Known studies/selection only. The complete qualifying bag, own18 and original
target remain required afterward. No production work starts on import.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_rbm400_neural_execution_reference_v4 as base
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_pipeline as pipeline
import upstream_full_context_catboost_supervision as supervision
import upstream_full_context_catboost_canonical as canonical
import verify_upstream_full_context_catboost_study as replay

ROOT=study.ROOT
REPORT=ROOT/'experiments/upstream-full-context-catboost-execution-v1'
RESULTS=ROOT/'experiments/upstream-full-context-catboost-execution-results-v1'
DECL=ROOT/'OWN_FULL_CONTEXT_CATBOOST_OWNED_KNOWN_PLAN_V1.md'
PROTOCOL=ROOT/'experiments/upstream-full-context-catboost-owned-known-protocol-v1/verification.json'
WORKER=ROOT/'scripts/upstream_full_context_catboost_worker_v1.py'
PHASES=tuple(kind+'_'+arm for arm in study.ARMS for kind in ('prepare','fit','replay','canonical'))+('select',)
ENV=dict(base.ENV)
FAMILY_LIMIT=16*2**30
OUTPUT_INVENTORY_MiB=dict(models=30*64,vectors_and_weights=128,metadata=512,model_staging=64,general_reserve=64)
OUTPUT_RESERVE=sum(OUTPUT_INVENTORY_MiB.values())*2**20
PLAN_STATE='declared_full_original_budget_owned_two_tree_arms_and_known_only_selection_v1'
COMPLETE='complete_owned_both_original_tree_arms_independently_replayed_and_known_only_selected_v1'
PRIMARY='rbm400-own17-simplex'
PREDECESSOR=ROOT/'experiments/upstream-rbm400-own17-execution-reference-v4'
PREDECESSOR_STUDY=ROOT/'experiments/upstream-rbm400-own17-calibration-reference-v4'
GROUP_COUNTS={'source0':1408789,'source1':1408342}
read=base.read
stamp=base.stamp
status=base.status
identity=base.identity
family=supervision.family


def metadata_guard(extra_disk_bytes=0):
    return base.metadata_guard(extra_disk_bytes)


def save(path,value,guard=metadata_guard):
    return base.durable_payload(Path(path),lambda s:s.write((json.dumps(value,indent=2)+'\n').encode()),ROOT,guard)


def command(phase):
    if phase not in PHASES:raise ValueError('Exact original known-study phase required')
    return [sys.executable,'-u',str(WORKER),'--phase',phase]


def producer_command(phase,pid):
    if phase not in PHASES or phase.split('_')[0] not in ('fit','replay') or type(pid) is not int or pid<=0:
        raise ValueError('Exact original stream-producing phase/root required')
    return command(phase)+['--source-root',str(pid)]


def policy():
    return dict(family_bytes=FAMILY_LIMIT,total_physical_and_RSS_bytes=24*2**30,compressor_bytes=5*2**30,
        admission_compressor_bytes=int(4.5*2**30),maximum_swap_growth_bytes=128*2**20,
        disk_floor_bytes=int(18.75*2**30),full_remaining_output_reservation=OUTPUT_RESERVE,output_inventory_MiB=OUTPUT_INVENTORY_MiB,
        stable_samples=3,sample_spacing_seconds=10,both_original_lanes=True,same_process_group_frames=True,
        original_baseline_sha256=study.BASELINE_SHA)


def check_sources(plan):
    if (plan['state']!=PLAN_STATE or plan['phases']!=list(PHASES) or plan['environment']!=ENV
            or plan['policy']!=policy() or plan['libraries']!=study.versions() or plan['python']!=sys.version
            or plan['parameters']!=study.backend.PARAMETERS or plan['max_rounds']!=2000 or plan['patience']!=100
            or plan['seeds']!=[2026,7,13] or plan['folds']!=5 or plan['known_rows']!=1128164
            or plan['qualifying_rows']!=2817131 or plan['tie_rule']!='scalar' or plan['tie_tolerance']!=1e-12
            or plan['selection_uses_only_known_CV'] is not True):
        raise ValueError('All original budgets, runtime, source and resource policy must remain exact')
    for name,digest in plan['source_sha256'].items():
        if study.sha(ROOT/name)!=digest:raise ValueError('Frozen original tree source changed: '+name)


def source_files():
    import upstream_rbm400_own17_execution_reference_v4 as previous
    old=previous.REPORT/'plan.json';previous.check_sources(read(old))
    required={Path(__file__),WORKER,DECL,old}
    required.update(ROOT/p for p in read(old)['source_sha256'])
    for pattern in ('*upstream_full_context_catboost*.py',):required.update((ROOT/'scripts').glob(pattern))
    for p in (ROOT/'experiments').glob('upstream-full-context-catboost-*-protocol/verification.json'):
        proof=read(p)
        for name,digest in proof['source_sha256'].items():
            if study.sha(ROOT/name)!=digest:raise ValueError('Original component review expectation changed: '+name)
            required.add(ROOT/name)
        required.add(p)
    required.update((ROOT/'scripts'/name) for name in (
        'upstream_full_context_catboost_execution_v1.py','upstream_full_context_catboost_worker_v1.py',
        'upstream_rbm400_neural_execution_reference_v4.py','upstream_research_footprints_native_original_scope.py'))
    return required


def frozen_plan():
    required=source_files();proof=read(PROTOCOL)
    if (proof['state']!='reviewed_owned_original_two_tree_known_studies_v1_pending_real_run'
            or proof['test_exit_code']!=0 or proof['test_count']<89 or proof['production_source_admission_run'] is not False
            or proof['production_models_fitted']!=0 or proof['qualifying_labels_read'] is not False
            or proof['review']!='manual_source_and_full_regression_complete'):
        raise ValueError('Reviewed complete owned tree source scope required before declaration')
    for p in required:
        if proof['source_sha256'].get(str(p.relative_to(ROOT)))!=study.sha(p):
            raise ValueError('Owned tree review source changed: '+str(p))
    required.add(PROTOCOL)
    return dict(state=PLAN_STATE,created=stamp(),phases=list(PHASES),environment=ENV,policy=policy(),
        libraries=study.versions(),python=sys.version,parameters=study.backend.PARAMETERS,max_rounds=2000,patience=100,
        seeds=[2026,7,13],folds=5,known_rows=1128164,qualifying_rows=2817131,tie_rule='scalar',tie_tolerance=1e-12,
        selection_uses_only_known_CV=True,source_sha256={str(p.relative_to(ROOT)):study.sha(p) for p in sorted(required)})


def prerequisites():
    done=read(PREDECESSOR/'completion.json');path=PREDECESSOR_STUDY/'independent-canonical-verification.json';proof=read(path)
    if (done['state']!='complete_own17_known_only_calibration_full_preaudit_and_separate_fixed_recovered_canonical_audit'
            or done['verification_sha256']!=study.sha(path) or done['qualifying_labels_used_for_fitting_or_selection'] is not False
            or proof['state']!='complete_independent_selected_RBM400_own17_full_source_models_coefficients_and_frozen_group_scores'
            or proof['primary']!=PRIMARY or proof['simplex_supports_examined']!=131071 or proof['ridge_equations']!=18
            or proof['all_six_vectors_independently_verified'] is not True or proof['full_inherited_source_closure_rehashed'] is not True
            or proof['qualifying_labels_used_for_fitting_or_selection'] is not False or proof['official_or_untouched_test'] is not False
            or set(proof['groups'])!=set(GROUP_COUNTS)):
        raise ValueError('Exact completed own17 primary and independent full canonical audit required')
    for name,count in GROUP_COUNTS.items():
        row=proof['groups'][name]
        if row['rows']!=count or not math.isfinite(row['rmse']) or row['rmse']<=0:raise ValueError('Both complete original recovered groups required')
    passed=all(row['rmse']<.85725 for row in proof['groups'].values())
    if proof['both_groups_below_target'] is not passed:raise ValueError('Original strict primary start condition must be self-consistent')
    bindings={str(path.relative_to(ROOT)):study.sha(path),str((PREDECESSOR/'completion.json').relative_to(ROOT)):study.sha(PREDECESSOR/'completion.json')}
    for name,digest in proof['bindings'].items():
        if study.sha(ROOT/name)!=digest:raise ValueError('Original own17 proof dependency changed')
        bindings[name]=digest
    return dict(start_followup=not passed,source_sha256=bindings,condition='predeclared_own17_primary_misses_strict_both_group_target',
        qualifying_outcomes_read=False,score_values_used_for_parameters_or_selection=False)


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


def validate_source_request(phase,captured,request,guard):
    mode,arm=phase.split('_');run,report=study.paths(arm)
    manifest,metadata,expected=study.load(arm,study.sha(report/'manifest.json'),guard)
    actual=identity.process(request['producer']['pid'])
    if (request['mode']!=mode or request['arm']!=arm or request['command']!=producer_command(phase,captured['pid'])
            or not family.matches(captured,request['scientific_root'],captured['pid'])
            or not same_role(actual,request['producer']) or actual['ppid']!=captured['pid'] or actual['pgid']!=captured['pid']
            or request['manifest_sha256']!=study.sha(report/'manifest.json')
            or request['source_metadata_sha256']!=manifest['source_metadata_sha256']
            or request['contract_sha256']!=pipeline.hashlib.sha256(study.wire.encoded(expected)).hexdigest()
            or request['qualifying_labels_read'] is not False):
        raise ValueError('Exact original producer command/lifetime/arm/metadata role request required')
    existing=roles(phase,captured)
    if not same_role(existing.get('producer'),request['producer']):raise ValueError('Actual original early source bootstrap capture required')
    bindings=dict(metadata['source_sha256'])
    for row in manifest['outputs'].values():bindings[str(Path(row['path']).relative_to(ROOT))]=row['sha256']
    bindings[str((report/'manifest.json').relative_to(ROOT))]=study.sha(report/'manifest.json')
    for path in [*pipeline.code_paths(),WORKER]:bindings[str(path.relative_to(ROOT))]=study.sha(path)
    if request['source_sha256']!=bindings:raise ValueError('Exact complete original producer source closure required')
    return existing


def admit_source(phase,captured,request,guard):
    authorize(phase);existing=validate_source_request(phase,captured,request,guard)
    save(REPORT/(phase+'-roles.json'),dict(scientific_root_pid=captured['pid'],roles=existing),guard)
    path=REPORT/(phase+'-source-authorization.json')
    value=dict(state='authorized_original_full_context_tree_producer_role',request=request)
    save(path,value,guard);return dict(path=str(path),sha256=study.sha(path))


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
    if (auth.get('state')!='authorized_original_full_context_tree_producer_role'
            or not same_role(auth['request']['producer'],actual) or auth['request']['scientific_root']['pid']!=root_pid):
        raise ValueError('Actual original producer request authorization required')
    validate_source_request(phase,captured,auth['request'],metadata_guard)
    return captured


def scientific_path(phase):
    if phase=='select':return ROOT/'experiments/upstream-full-context-catboost-selection/selection.json',canonical.SELECTION_STATE
    kind,arm=phase.split('_');report=study.paths(arm)[1]
    names={'prepare':('manifest.json',study.PREPARED),'fit':('fit-completion.json',study.FITTED),
        'replay':('independent-verification.json',replay.STATE),'canonical':('independent-canonical-verification.json',canonical.STATE)}
    name,state=names[kind];return report/name,state


def validate_result(phase,value,plan_sha):
    path,state=scientific_path(phase);actual=read(path)
    if (value['state']!='complete_owned_original_tree_'+phase or value['execution_plan_sha256']!=plan_sha
            or value['scientific_proof_sha256']!=study.sha(path) or value['qualifying_labels_read'] is not False
            or actual['state']!=state):raise ValueError('Exact complete original scientific result required')
    if phase=='select':
        if actual['selected_arm']!=canonical.choose(actual['tree_only_known_CV_scores']) or actual['qualifying_labels_read'] is not False:
            raise ValueError('Both complete canonical known-only arms must determine selection')
    else:
        kind,arm=phase.split('_')
        if actual['arm']!=arm or actual['qualifying_labels_read'] is not False:raise ValueError('Exact complete original arm required')
        if kind=='fit' and (actual['all_five_folds_complete'] is not True or actual['original_seed_slots']!=15):raise ValueError('All original fold/seed slots required')
        if kind=='replay' and (actual['every_full_OOF_join_bit_verified'] is not True or actual['ridge_equations']!=10):raise ValueError('Complete original full replay required')
        if kind in ('fit','replay'):
            chain=read(path.parent/(kind+'-pipeline-completion.json'))
            if chain['scientific_proof_sha256']!=study.sha(path):raise ValueError('Actual successful isolated producer pipeline required')


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
    roots=[REPORT,RESULTS,ROOT/'experiments/upstream-full-context-catboost-selection']+[p for a in study.ARMS for p in study.paths(a)]
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
                    status(REPORT/'status.json',dict(state='waiting_complete_original_own17_primary_canonical_condition',observed=stamp()));time.sleep(10)
            save(REPORT/'prerequisites.json',pre)
            if not pre['start_followup']:
                save(REPORT/'not-needed.json',dict(state='own17_primary_already_meets_original_target_no_tree_study_started'))
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
            final=ROOT/'experiments/upstream-full-context-catboost-selection/selection.json'
            save(REPORT/'completion.json',dict(state=COMPLETE,verification_sha256=study.sha(final),qualifying_labels_read=False,
                completed_phase_and_observed_resource_proof_sha256=completed,
                remaining='Full selected qualifying bag/replay and own18 calibration/fixed audits required; not goal completion'))
            save(REPORT/'owner-release.json',dict(state='all_original_scientific_roles_terminal_after_full_known_selection'))
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
