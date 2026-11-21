"""Owned execution protocol for the independent full-budget local GPU RBM.

No production queue is started merely by importing this module. Numerical work
requires a frozen plan, actual bootstrap/owner/watchdog gate and original lanes.
"""
from dataclasses import asdict
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
from upstream_factored_rbm_training_metal import Config
from upstream_factored_rbm_terminal_metal import durable_directory,durable_payload

ROOT=u.ROOT
REPORT=ROOT/'experiments/upstream-factored-rbm-execution-deterministic-metal'
STUDY=ROOT/'experiments/upstream-factored-rbm-gpu400-deterministic-metal'
ART=ROOT/'artifacts/upstream-factored-rbm-gpu400-deterministic-metal'
FLOOR=int(18.75*2**30)
OUTPUT_RESERVE=64*2**20
TOTAL_LIMIT=24*2**30
PHASES=('data','mps_preflight','fit_known','replay_known','fit_qualifying','replay_qualifying','assemble')
ENV=dict(OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
         PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0',PYTHONUNBUFFERED='1')
RESULTS={
    'data':('data-admission.json','complete_original63_payload_headers_both_phase_rows_full_holdout_map_and_zero_targets'),
    'mps_preflight':('mps-preflight.json','complete_device_math_checks_only'),
    'fit_known':('known-fit.json','complete_own_gpu_rbm_phase15_actual_model_and_native_predictions'),
    'replay_known':('known-replay.json','complete_independent_own_gpu_rbm_phase_actual_model_and_all_native_queries'),
    'fit_qualifying':('qualifying-fit.json','complete_own_gpu_rbm_phase15_actual_model_and_native_predictions'),
    'replay_qualifying':('qualifying-replay.json','complete_independent_own_gpu_rbm_phase_actual_model_and_all_native_queries'),
    'assemble':('paired-verification.json','complete_own_gpu_rbm400_full15_15_models_and_paired_queries_no_outcomes'),
}


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
    if phase not in PHASES:raise ValueError('Declared phase required')
    return (3 if phase in ('data','assemble') else 8)*2**30


def command(phase):
    if phase not in PHASES:raise ValueError('Declared phase required')
    return [sys.executable,'-u',str(ROOT/'scripts/upstream_factored_rbm_worker_metal.py'),'--phase',phase]


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
    recovery.check_declaration(plan)
    if plan['configuration']!=asdict(Config()) or plan['phases']!=list(PHASES) or plan['environment']!=ENV:
        raise ValueError('Exact original15+15 configuration/phase/runtime plan required')
    for name,digest in plan['source_sha256'].items():
        if inputs.sha(ROOT/name)!=digest:raise ValueError('Original frozen RBM source changed: '+name)
    if plan['libraries']!={n:importlib.metadata.version(n) for n in ('numpy','torch')}:
        raise ValueError('Frozen numerical runtime versions changed')


def validate_result(phase,value,plan_sha256):
    if value['state']!=RESULTS[phase][1] or value['execution_plan_sha256']!=plan_sha256:
        raise ValueError('Exact original phase completion/plan binding required')
    if value.get('qualifying_labels_read') is not False:raise ValueError('No recovered outcomes permitted')
    if phase=='mps_preflight':
        from upstream_factored_rbm_preflight import REQUIRED_STRESS
        if (value['device']!='mps' or value['actual_MPS_tested'] is not True or value['passed_for_production_device'] is not True
                or value['full_required_stress_tested'] is not True or value['fresh_mps_runtime_import'] is not True
                or value['cases']['stress']['shape']!=REQUIRED_STRESS):
            raise ValueError('Actual fresh MPS/full-production-width preflight required')


def output_remaining():
    path=STUDY/'data-admission.json'
    if not path.exists():return OUTPUT_RESERVE
    data=read(path)
    predicted=sum(v['statistics']['predictive_parameter_bytes'] for v in data['phases'].values())
    vectors=4*(1760494+1128164+2817131)
    used=0;seen=set()
    if ART.exists():
        for p in ART.iterdir():
            if p.is_symlink() or not p.is_file():raise ValueError('Only owned regular artifact outputs permitted')
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);used+=s.st_size
    return max(OUTPUT_RESERVE,predicted+vectors+2*OUTPUT_RESERVE-used)


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
        disk_reserve=max(OUTPUT_RESERVE,extra_disk_bytes)
        while True:
            now=time.monotonic();free=0;measurement_error=None
            try:free=shutil.disk_usage(ROOT).free
            except Exception as error:measurement_error=repr(error)
            torch=sys.modules.get('torch');gpu=0;allocated=0;gpu_error=None
            try:
                if self.phase not in ('data','assemble') and torch is not None:
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


def frozen_plan():
    return recovery.frozen_plan(sys.modules[__name__])


PREREQUISITES=(
    ('experiments/upstream-original1200-exact-build-recovery/completion.json','complete_original1200_full23_23_and_independent58_output_verification','proof_sha256',
     'experiments/upstream-original1200/independent-verification.json','complete_independent_original1200_both23_phases_and_all58_numeric_outputs'),
    ('experiments/upstream-own15-complete-trajectory-queue/completion.json','complete_own15_known_only_calibration_freeze_and_independent_recovered_audit','verification_sha256',
     'experiments/upstream-own15-complete-trajectory-calibration/independent-canonical-verification.json','complete_independent_own15_full_source_models_coefficients_and_frozen_group_scores'),
)


def prerequisites():
    result={}
    for receipt,state,key,proof,pstate in PREREQUISITES:
        r,p=read(ROOT/receipt),read(ROOT/proof)
        if r['state']!=state or r[key]!=inputs.sha(ROOT/proof) or p['state']!=pstate:
            raise ValueError('Complete original independently verified prerequisite required')
        result[receipt]=inputs.sha(ROOT/receipt);result[proof]=inputs.sha(ROOT/proof)
    p=read(ROOT/PREREQUISITES[0][3]);schema=read(ROOT/'experiments/upstream-original1200/execution-plan.json')['exact_expected_outputs']
    expected={'artifacts/upstream-original1200/source/preds_new/'+row['name'] for row in schema}
    expected.add('artifacts/upstream-original1200/source/preds_new/tsvdx5-1200lm.out')
    original=read(ROOT/'experiments/upstream-own14-matched391-bounded-sidecar/native-ownership.json')
    if (p['epochs_each']!=23 or p['factors']!=1200 or p['neighbors']!=100 or len(schema)!=58
            or set(p['outputs'])!=expected or p['qualifying_labels_read'] is not False
            or identity.same_owner(original,identity.process(original['pid']))):
        raise ValueError('Full original1200 terminality and all59 output proof required')
    return result


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
    if REPORT.exists() or STUDY.exists() or ART.exists():raise ValueError('Existing scientific/control namespace retained; no retry')
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
                    status(REPORT/'status.json',dict(state='waiting_original1200_and_full_local_own15_completion',observed=stamp()));time.sleep(10)
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
                                and safe(workers,current,baseline,shutil.disk_usage(ROOT).free))
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
            final=STUDY/RESULTS['assemble'][0]
            exclusive_json(REPORT/'completion.json',dict(state='complete_full_own_gpu_rbm400_original15_15_and_independent_queries',verification_sha256=inputs.sha(final),qualifying_labels_read=False,observed=stamp()))
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
