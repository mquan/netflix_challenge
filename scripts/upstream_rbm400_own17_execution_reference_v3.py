"""Owned seventeen-family known-only calibration, preaudit and fixed recovered audits.

No production queue is started merely by importing this module. Numerical work
requires a frozen plan, actual bootstrap/owner/watchdog gate and original lanes.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_research_footprints_native_original_scope as resources
import upstream_rbm400_metal_declaration as recovery
import upstream_cf96_memory_resume as memory
import upstream_cf96_primary_lane as identity
import upstream_factored_rbm_inputs as inputs
from upstream_factored_rbm_terminal_metal import durable_directory,durable_payload

ROOT=u.ROOT
REPORT=ROOT/'experiments/upstream-rbm400-own17-execution-reference-v3'
STUDY=ROOT/'experiments/upstream-rbm400-own17-execution-results-reference-v3'
ART=ROOT/'artifacts/upstream-rbm400-own17-execution-reservation-reference-v3'
FLOOR=int(18.75*2**30)
OUTPUT_RESERVE=64*2**20
TOTAL_LIMIT=24*2**30
PHASES=('known','freeze','preaudit','audit','canonical')
MPS_PHASES=set()
ENV=dict(OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
         PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0',PYTHONUNBUFFERED='1')
RESULTS={phase:(phase+'.json','complete_owned_RBM400_own17_'+phase) for phase in PHASES}


def stamp():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def metadata_guard(extra_disk_bytes=0):
    if shutil.disk_usage(ROOT).free<FLOOR+max(OUTPUT_RESERVE,extra_disk_bytes):
        raise RuntimeError('Original metadata disk floor/reserve unavailable')


def exclusive_json(path,value,guard=metadata_guard):
    return durable_payload(path,lambda stream:stream.write((json.dumps(value,indent=2)+'\n').encode()),ROOT,guard)


def status(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def cap(phase):
    if phase not in PHASES:raise ValueError('Declared complete neural phase required')
    return (16 if phase in MPS_PHASES else 3)*2**30


def command(phase):
    if phase not in PHASES:raise ValueError('Declared complete neural phase required')
    return [sys.executable,'-u',str(ROOT/'scripts/upstream_rbm400_own17_worker_reference_v3.py'),'--phase',phase]


def same_child(captured,actual):
    return bool(actual and actual['pid']==captured['pid'] and actual['started']==captured['started']
        and actual['pgid']==captured['pid'] and actual['command'] in (captured['command'],captured.get('worker_command')))


def stop_child(captured):
    actual=identity.process(captured['pid'])
    if not same_child(captured,actual):return False
    os.killpg(captured['pid'],signal.SIGSTOP);return True


def safe(workers,current,baseline,free_bytes,reserve=0,disk_reserve=OUTPUT_RESERVE):
    return (sum(w['bytes'] for w in workers.values())+reserve<TOTAL_LIMIT
        and sum(w['rss_bytes'] for w in workers.values())+reserve<TOTAL_LIMIT
        and current['compressor_bytes']<5*2**30
        and (baseline is None or current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20)
        and free_bytes>=FLOOR+disk_reserve)


def ready(workers,current,baseline,free_bytes,reserve,disk_reserve):
    return safe(workers,current,baseline,free_bytes,reserve,disk_reserve) and current['compressor_bytes']<4.5*2**30


def continuation_ready(workers,pid,phase,current,baseline,free_bytes,disk_reserve):
    own=workers.get(str(pid));limit=cap(phase)
    if not own or max(own['bytes'],own['rss_bytes'])>limit:return False
    # Each accounting domain must reserve the full worker cap independently.
    # A high RSS must not erase the reservation for lower physical footprint.
    padded=dict(workers);padded[str(pid)]=dict(own,bytes=limit,rss_bytes=limit)
    return ready(padded,current,baseline,free_bytes,0,disk_reserve)


def best_effort_json(path,value):
    try:exclusive_json(path,value)
    except (OSError,RuntimeError):pass


def check_sources(plan):
    import upstream_rbm400_own17_calibration_reference_v3 as study
    if (plan['state']!='declared_full_own17_known_only_calibration_six_frozen_vectors_and_fixed_audits'
            or plan['phases']!=list(PHASES) or plan['environment']!=ENV
            or plan['known_rows']!=1128164 or plan['qualifying_rows']!=2817131
            or plan['worker_caps']!={p:cap(p) for p in PHASES} or plan['families']!=17
            or plan['simplex_supports']!=131071 or plan['ridge_equations']!=18 or plan['ridge_lambda']!=10.
            or plan['candidates']!=list(study.CANDIDATES) or plan['primary']!=study.CANDIDATES[0]
            or plan['qualifying_labels_used_for_fitting_or_selection'] is not False):
        raise ValueError('All17 original families/full simplex/ridge/sixvectors/fixed primary required')
    for name,digest in plan['source_sha256'].items():
        if inputs.sha(ROOT/name)!=digest:raise ValueError('Frozen own17 source changed: '+name)
    if plan['libraries']!={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch')}:
        raise ValueError('Frozen own17 numerical runtime versions changed')


def validate_result(phase,value,plan_sha256):
    import upstream_rbm400_own17_calibration_reference_v3 as study
    import verify_upstream_rbm400_own17_calibration_reference_v3 as replay
    if (phase not in PHASES or value['state']!=RESULTS[phase][1]
            or value['execution_plan_sha256']!=plan_sha256
            or value['qualifying_labels_read']!=(phase in ('audit','canonical'))):
        raise ValueError('Exact owned full own17 phase receipt required')
    names={'known':('known-completion.json','complete_original17_selected_RBM400_NN_known_only_coefficients'),
        'freeze':('frozen.json','complete_all_three_selected_RBM400_own17_candidates_frozen_before_recovered_audit'),
        'preaudit':('independent-preaudit-verification.json',replay.STATE),
        'audit':('independent-recovered-audit.json','complete_all_three_fixed_selected_RBM400_own17_recovered_audits'),
        'canonical':('independent-canonical-verification.json','complete_independent_selected_RBM400_own17_full_source_models_coefficients_and_frozen_group_scores')}
    name,expected=names[phase];path=study.REPORT/name;result=read(path)
    if value['scientific_proof_sha256']!=inputs.sha(path) or result['state']!=expected:
        raise ValueError('Actual complete scientific phase proof must match the owned receipt')
    if phase in ('known','freeze','preaudit') and result['qualifying_labels_read'] is not False:
        raise ValueError('No qualifying outcomes before independent preaudit')
    if phase=='preaudit' and (result['simplex_supports_examined']!=131071 or result['ridge_equations']!=18
            or result['all_six_vectors_bitwise'] is not True):raise ValueError('Full17-family coefficients/vectors required')
    if phase in ('audit','canonical') and (result['qualifying_labels_used_for_fitting_or_selection'] is not False
            or result['official_or_untouched_test'] is not False or result['primary']!=study.CANDIDATES[0]
            or [r['candidate'] for r in result['results']]!=list(study.CANDIDATES)):
        raise ValueError('All three fixed adaptive audits with original primary required')
    if phase=='canonical' and (result['all_six_vectors_independently_verified'] is not True
            or result['full_inherited_source_closure_rehashed'] is not True
            or result['simplex_supports_examined']!=131071 or result['ridge_equations']!=18):
        raise ValueError('Separate full inherited source/numerical canonical verification required')


def output_remaining():
    import upstream_rbm400_own17_calibration_reference_v3 as study
    expected=256*1024**2;used=0;seen=set()
    for directory in (study.REPORT,study.ART,STUDY):
        if not directory.exists():continue
        for p in directory.rglob('*'):
            if p.is_symlink():raise ValueError('No aliased owned bag payloads permitted')
            if p.is_dir():continue
            if not p.is_file():raise ValueError('Owned regular bag outputs required')
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:seen.add(key);used+=st.st_size
    return max(OUTPUT_RESERVE,expected+OUTPUT_RESERVE-used)


class WorkerGuard:
    """Self-suspend inside the same call stack; never throw/retry for resources."""
    def __init__(self,phase):
        if phase not in PHASES:raise ValueError('Declared worker phase required')
        self.phase=phase;self.limit=cap(phase);self.last=0.;self.counter=0
        self.plan=read(REPORT/'plan.json');check_sources(self.plan)
        self.baseline=read(REPORT/'first-scientific-baseline.json')
        self.baseline_sha=inputs.sha(REPORT/'first-scientific-baseline.json')
        self.captured=read(REPORT/(phase+'-ownership.json'))
        self.predecessors=read(REPORT/(phase+'-gate.json'))['predecessor_sha256']
        if self.captured['pid']!=os.getpid() or not same_child(self.captured,identity.process(os.getpid())):
            raise ValueError('Actual original dedicated worker identity required')

    def __call__(self,extra_disk_bytes=0):
        while True:
            now=time.monotonic();free=0;measurement_error=None
            try:disk_reserve=max(output_remaining(),extra_disk_bytes)
            except Exception as error:
                # Account an unknown remaining-output scan as invalid, retain
                # a conservative full reservation, and self-stop this frame.
                disk_reserve=max(256*2**20+OUTPUT_RESERVE,extra_disk_bytes,getattr(self,'last_disk_reserve',0))
                measurement_error='remaining-output accounting: '+repr(error)
            self.last_disk_reserve=disk_reserve
            try:free=shutil.disk_usage(ROOT).free
            except Exception as error:measurement_error=(measurement_error+'; ' if measurement_error else '')+repr(error)
            torch=sys.modules.get('torch');gpu=0;allocated=0;gpu_error=None
            try:
                if self.phase in MPS_PHASES and torch is not None:
                    gpu=torch.mps.driver_allocated_memory();allocated=torch.mps.current_allocated_memory()
                    if gpu>self.limit and allocated<=self.limit:
                        torch.mps.empty_cache();gpu=torch.mps.driver_allocated_memory()
            except Exception as error:gpu_error=repr(error)
            if not measurement_error and not gpu_error and now-self.last<3 and free>=FLOOR+disk_reserve and gpu<=self.limit:return
            check_sources(self.plan)
            for name,digest in self.predecessors.items():
                if inputs.sha(ROOT/name)!=digest:raise ValueError('Frozen prior scientific phase changed')
            if inputs.sha(REPORT/'first-scientific-baseline.json')!=self.baseline_sha:
                raise ValueError('Original scientific swap baseline changed')
            try:
                workers,current=resources.footprints(os.getpid()),memory.memory();own=workers.get(str(os.getpid()))
                admitted=bool(not measurement_error and not gpu_error and own and max(own['bytes'],own['rss_bytes'],gpu)<=self.limit
                    and safe(workers,current,self.baseline,free,disk_reserve=disk_reserve))
                sample=dict(workers=workers,memory=current,gpu_driver_bytes=gpu,gpu_allocated_bytes=allocated,gpu_accounting_error=gpu_error,disk_accounting_error=measurement_error,free_bytes=free)
            except Exception as error:
                admitted=False;sample=dict(accounting_error=repr(error),gpu_driver_bytes=gpu,gpu_allocated_bytes=allocated,free_bytes=free)
            if admitted:self.last=now;return
            self.counter+=1
            record=dict(phase=self.phase,child=self.captured,extra_disk_reserve=disk_reserve,
                original_baseline_sha256=self.baseline_sha,sample=sample,observed=stamp(),no_automatic_retry=True)
            try:
                status(REPORT/(self.phase+'-worker-resource-request.json'),record)
                path=REPORT/(self.phase+'-self-stop-%06d.json'%self.counter)
                with path.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
            except OSError:pass  # Resource failure must still preserve actual RAM.
            if not stop_child(self.captured):raise RuntimeError('Original worker identity disappeared before resource stop')
            # SIGCONT from the same live supervisor returns HERE, preserving
            # RNG/model/batch/write frames and the original baseline.


def authorize(phase):
    plan=read(REPORT/'plan.json');check_sources(plan)
    captured=read(REPORT/(phase+'-ownership.json'));gate=read(REPORT/(phase+'-gate.json'))
    owner=read(REPORT/'monitor.json');actual=identity.process(os.getpid())
    if (captured['pid']!=os.getpid() or not same_child(captured,actual) or actual['command']!=' '.join(command(phase))
            or gate['pid']!=os.getpid() or gate['command']!=command(phase) or gate['phase']!=phase
            or gate['execution_plan_sha256']!=inputs.sha(REPORT/'plan.json') or gate['owner_sha256']!=inputs.sha(REPORT/'monitor.json')
            or gate['baseline_sha256']!=inputs.sha(REPORT/'first-scientific-baseline.json')
            or not identity.same_owner(owner,identity.process(owner['pid']))
            or not identity.same_owner(gate['watchdog'],identity.process(gate['watchdog']['pid']))):
        raise ValueError('Exact live source-bound owner/watchdog/phase/baseline gate required')
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('Original runtime environment required before Torch import')
    prereq=read(REPORT/'prerequisites.json')
    for name,digest in prereq['source_sha256'].items():
        if inputs.sha(ROOT/name)!=digest:raise ValueError('Completed original prerequisite changed')
    for name,digest in gate['predecessor_sha256'].items():
        if inputs.sha(ROOT/name)!=digest:raise ValueError('Completed predecessor changed before phase')
    return plan


def source_files():
    import upstream_rbm400_own17_calibration_reference_v3 as study
    import verify_upstream_rbm400_own17_calibration_reference_v3
    import verify_upstream_rbm400_own17_arithmetic
    import upstream_rbm400_bag_execution_reference_v3 as previous
    import upstream_rbm400_own17_execution_reference_v2 as legacy
    old_plan=legacy.REPORT/'plan.json';legacy.check_sources(read(old_plan))
    required={Path(__file__).resolve(),study.DECL,ROOT/'scripts/upstream_rbm400_own17_worker_reference_v3.py'}
    for pattern in ('upstream_rbm400_own17*.py','verify_upstream_rbm400_own17*.py','test*upstream_rbm400_own17*.py'):
        required.update((ROOT/'scripts').glob(pattern))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.parent==ROOT/'scripts' and p.suffix=='.py':required.add(p)
    required.add(old_plan);required.update(ROOT/name for name in read(old_plan)['source_sha256'])
    preceding=previous.REPORT/'plan.json';previous.check_sources(read(preceding));required.add(preceding)
    required.update(ROOT/name for name in read(preceding)['source_sha256'])
    return required


def frozen_plan():
    import upstream_rbm400_own17_calibration_reference_v3 as study
    required=source_files();proof_path=ROOT/'experiments/upstream-rbm400-own17-execution-protocol-reference-v3/verification.json';proof=read(proof_path)
    if (proof['state']!='reviewed_full_own17_calibration_and_fixed_audit_execution_pending_actual_NN_pair'
            or proof['test_exit_code']!=0 or proof['test_count']<16
            or proof['actual_MPS_tested'] is not False or proof['production_source_admission_run'] is not False
            or proof['production_models_fitted']!=0 or proof['qualifying_labels_read'] is not False
            or proof['review']!='manual_source_and_full_regression_complete'):
        raise ValueError('Full reviewed CPU fixture protocol required without production admission')
    for path in required:
        if proof['source_sha256'].get(str(path.relative_to(ROOT)))!=inputs.sha(path):raise ValueError('Own17 review source changed: '+str(path))
    required.add(proof_path)
    return dict(state='declared_full_own17_known_only_calibration_six_frozen_vectors_and_fixed_audits',created=stamp(),
        phases=list(PHASES),environment=ENV,known_rows=1128164,qualifying_rows=2817131,families=17,
        simplex_supports=131071,ridge_equations=18,ridge_lambda=10.,candidates=list(study.CANDIDATES),primary=study.CANDIDATES[0],
        worker_caps={p:cap(p) for p in PHASES},qualifying_labels_used_for_fitting_or_selection=False,
        source_sha256={str(p.relative_to(ROOT)):inputs.sha(p) for p in sorted(required)},
        libraries={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch')},
        policy=dict(total_physical_and_rss_bytes=TOTAL_LIMIT,compressor_bytes=5*2**30,admission_compressor_bytes=4.5*2**30,
            swap_growth_bytes=128*2**20,disk_floor_bytes=FLOOR,stable_samples=3,primary_and_GPU_lanes=True,
            same_process_continuation=True,full_remaining_output_reservation=True,
            original_baseline_sha256=inputs.sha(ROOT/'experiments/upstream-factored-rbm-execution-deterministic-metal/first-scientific-baseline.json')))


def prerequisites():
    import upstream_rbm400_bag_execution_reference_v3 as previous
    import upstream_rbm400_bag_reference_v3 as bag
    import verify_upstream_rbm400_bag_reference_v3 as replay
    receipt=previous.REPORT/'completion.json';proof=bag.REPORT/'independent-verification.json'
    done,checked=read(receipt),read(proof)
    if (done['state']!='complete_selected_RBM400_qualifying_bag_and_independent_full_query_replay'
            or done['verification_sha256']!=inputs.sha(proof) or done['qualifying_labels_read'] is not False
            or checked['state']!=replay.STATE or checked['qualifying_labels_read'] is not False
            or checked['models_fitted']!=0 or checked['rows']!=2817131
            or checked['actual_MPS_predictions_replayed'] is not True
            or checked['provider_is_NN_only_on_both_queries'] is not True
            or checked['all_five_folds_and_three_seed_slots'] is not True
            or checked['every_full_query_operator_raw_normalized_seed_fold_and_output_bit_verified'] is not True):
        raise ValueError('Complete actual matched selected NN-only pair and all-query replay required')
    previous.check_sources(read(previous.REPORT/'plan.json'))
    return {str(p.relative_to(ROOT)):inputs.sha(p) for p in (receipt,proof)}


def bootstrap(phase,owner_pid):
    plan=read(REPORT/'plan.json');check_sources(plan);owner=read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact admitted owner required')
    captured=identity.process(os.getpid());captured['worker_command']=' '.join(command(phase))
    if captured['pgid']!=captured['pid']:raise ValueError('Dedicated worker group required')
    exclusive_json(REPORT/(phase+'-ownership.json'),captured)
    status(REPORT/'child-ownership.json',captured)
    gate_path=REPORT/(phase+'-gate.json')
    while True:
        if not identity.same_owner(owner,identity.process(owner_pid)):
            os.killpg(captured['pid'],signal.SIGSTOP);continue
        if gate_path.exists():
            gate=read(gate_path)
            if (gate['pid']!=captured['pid'] or gate['command']!=command(phase)
                    or gate['execution_plan_sha256']!=inputs.sha(REPORT/'plan.json')
                    or not identity.same_owner(gate['watchdog'],identity.process(gate['watchdog']['pid']))):
                raise ValueError('Live original watchdog/source dispatch gate required')
            exclusive_json(REPORT/(phase+'-dispatch.json'),dict(command=command(phase),pid=os.getpid(),observed=stamp()))
            env=os.environ.copy();env.update(ENV)
            os.execvpe(sys.executable,command(phase),env)
        time.sleep(.1)


def watchdog(owner_pid,keep_fds):
    owner=read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Original live supervisor required')
    try:
        while not (REPORT/'owner-release.json').exists():
            if not identity.same_owner(owner,identity.process(owner_pid)):
                p=REPORT/'child-ownership.json';captured=read(p) if p.exists() else None
                stopped=bool(captured and stop_child(captured))
                best_effort_json(REPORT/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,
                    original_lanes_retained=stopped,observed=stamp(),no_automatic_retry=True))
                # Inherited open-file descriptions retain the actual lane locks
                # after owner loss until this exact child becomes terminal.
                while captured and same_child(captured,identity.process(captured['pid'])):time.sleep(2)
                return
            time.sleep(2)
    finally:
        for fd in keep_fds:os.close(fd)


def main():
    plan=frozen_plan();check_sources(plan)
    import upstream_rbm400_own17_calibration_reference_v3 as study
    roots=[REPORT,STUDY,ART,study.REPORT,study.ART]
    if any(p.exists() for p in roots):raise ValueError('Existing complete/partial bag namespace retained; no retry')
    durable_directory(REPORT,ROOT,metadata_guard)
    exclusive_json(REPORT/'plan.json',plan);exclusive_json(REPORT/'monitor.json',identity.process(os.getpid()))
    primary=(ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a')
    gpu=(ROOT/'experiments/upstream-source-extension/run.lock').open('a')
    child=captured=watch=None;baseline=recovery.inherited_baseline();completed={};pause_count=0
    recovery.copy_baseline(REPORT/'first-scientific-baseline.json')
    def check():
        check_sources(plan)
        if watch is not None and watch.poll() is not None:raise RuntimeError('Original watchdog unavailable')
    def interrupted(signum,frame):raise KeyboardInterrupt('Original supervisor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as output:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid()),
                    '--keep-fds',str(primary.fileno()),str(gpu.fileno())],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(primary.fileno(),gpu.fileno()))
            while True:
                check()
                try:pre=prerequisites();break
                except (FileNotFoundError,json.JSONDecodeError):
                    status(REPORT/'status.json',dict(state='waiting_complete_actual_selected_NN_pair_and_full_query_replay',observed=stamp()));time.sleep(10)
            exclusive_json(REPORT/'prerequisites.json',dict(source_sha256=pre,observed=stamp(),score_values_used=False))
            while True:
                check();got=False
                try:
                    fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB);got=True
                    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if got:fcntl.flock(primary,fcntl.LOCK_UN)
                    status(REPORT/'status.json',dict(state='waiting_joint_original_primary_GPU_lanes',observed=stamp()));time.sleep(10)
            for phase in PHASES:
                good=0
                while good<3:
                    check()
                    try:
                        workers,current=resources.footprints(os.getpid()),memory.memory()
                        ok=ready(workers,current,baseline,shutil.disk_usage(ROOT).free,cap(phase),output_remaining())
                    except Exception:ok=False
                    good=good+1 if ok else 0
                    status(REPORT/'status.json',dict(state='waiting_original_headroom',phase=phase,stable_samples=good,observed=stamp()))
                    if good<3:time.sleep(10)
                ownership=REPORT/(phase+'-ownership.json')
                with (REPORT/(phase+'.log')).open('x') as output:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--bootstrap',str(os.getpid()),'--phase',phase],
                        cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    while not ownership.exists():
                        check()
                        if child.poll() is not None:raise RuntimeError('Original worker bootstrap failed')
                        time.sleep(.1)
                    captured=read(ownership)
                    if captured['pid']!=child.pid or not same_child(captured,identity.process(child.pid)):
                        raise ValueError('Exact original bootstrap identity required')
                    exclusive_json(REPORT/(phase+'-gate.json'),dict(pid=child.pid,phase=phase,command=command(phase),
                        execution_plan_sha256=inputs.sha(REPORT/'plan.json'),owner_sha256=inputs.sha(REPORT/'monitor.json'),
                        baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json'),watchdog=identity.process(watch.pid),predecessor_sha256=completed))
                    while child.poll() is None:
                        check();actual=identity.process(child.pid)
                        if actual is None:
                            if child.poll() is not None:break
                            time.sleep(.2);continue
                        try:
                            workers,current=resources.footprints(child.pid),memory.memory();own=workers.get(str(child.pid))
                            allowed=bool(own and max(own['bytes'],own['rss_bytes'])<=cap(phase)
                                and safe(workers,current,baseline,shutil.disk_usage(ROOT).free,disk_reserve=output_remaining()))
                        except Exception:allowed=False
                        if not allowed or 'T' in actual['state']:
                            if not stop_child(captured):continue
                            pause_count+=1;good=0
                            best_effort_json(REPORT/('pause-%06d.json'%pause_count),dict(phase=phase,child=captured,observed=stamp(),baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json')))
                            while good<3:
                                check()
                                if child.poll() is not None:raise RuntimeError('Preserved worker became terminal')
                                try:
                                    workers,current=resources.footprints(child.pid),memory.memory();own=workers.get(str(child.pid))
                                    request=REPORT/(phase+'-worker-resource-request.json');self_stop=read(request) if request.exists() else None
                                    needed=max(output_remaining(),self_stop['extra_disk_reserve'] if self_stop else OUTPUT_RESERVE)
                                    reported_gpu=self_stop['sample'].get('gpu_driver_bytes',0) if self_stop else 0
                                    ok=bool(own and reported_gpu<=cap(phase) and max(own['bytes'],own['rss_bytes'])<=cap(phase)
                                        and continuation_ready(workers,child.pid,phase,current,baseline,shutil.disk_usage(ROOT).free,needed))
                                except Exception:ok=False
                                good=good+1 if ok else 0
                                try:status(REPORT/'status.json',dict(state='same_worker_RAM_preserved_waiting_original_readmission',phase=phase,stable_samples=good,observed=stamp()))
                                except OSError:pass
                                if good<3:time.sleep(10)
                            if not same_child(captured,identity.process(child.pid)):raise ValueError('Same original worker required for continuation')
                            exclusive_json(REPORT/('continuation-%06d.json'%pause_count),dict(child=captured,phase=phase,original_baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json'),observed=stamp()))
                            os.killpg(child.pid,signal.SIGCONT)
                        status(REPORT/'status.json',dict(state=phase,pid=child.pid,observed=stamp()));time.sleep(3)
                    if child.returncode!=0:raise RuntimeError('Scientific phase terminal failure; no retry: '+phase)
                result=STUDY/RESULTS[phase][0];validate_result(phase,read(result),inputs.sha(REPORT/'plan.json'))
                completed[str(result.relative_to(ROOT))]=inputs.sha(result);child=captured=None
            final=study.REPORT/'independent-canonical-verification.json'
            exclusive_json(REPORT/'completion.json',dict(state='complete_own17_known_only_calibration_full_preaudit_and_separate_fixed_recovered_canonical_audit',verification_sha256=inputs.sha(final),qualifying_labels_used_for_fitting_or_selection=False,observed=stamp()))
            exclusive_json(REPORT/'owner-release.json',dict(state='all_original_scientific_children_terminal_after_independent_verification'))
    except BaseException as error:
        stopped=bool(captured and child is not None and child.poll() is None and stop_child(captured))
        status(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=stopped,no_automatic_retry=True,observed=stamp()))
        if child is None or child.poll() is not None:
            exclusive_json(REPORT/'owner-release.json',dict(state='no_live_scientific_child_at_terminal_owner_failure'))
        raise
    finally:primary.close();gpu.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true')
    parser.add_argument('--watch',type=int);parser.add_argument('--keep-fds',type=int,nargs='*',default=[])
    parser.add_argument('--bootstrap',type=int);parser.add_argument('--phase',choices=PHASES);args=parser.parse_args()
    if args.watch:watchdog(args.watch,args.keep_fds)
    elif args.bootstrap:
        if not args.phase:parser.error('Declared phase required')
        bootstrap(args.phase,args.bootstrap)
    elif args.start:main()
    else:parser.error('Explicit --start required; import/checking never launches a job')
