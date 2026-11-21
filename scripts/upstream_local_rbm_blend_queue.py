"""Owned serial CPU calibration/audit queue after full independent local RBM."""
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
import upstream_local_rbm_blend as study
import verify_upstream_local_rbm_blend as verification

rbm=study.rbm
inputs=study.inputs
ROOT=study.ROOT
REPORT=ROOT/'experiments/upstream-local-rbm-blend-queue'
LIMIT=2*2**30
ENV={k:'2' for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
ENV['PYTHONUNBUFFERED']='1'
PHASES=('known','freeze','preaudit','audit','canonical')
RESULTS={
    'known':('known-completion.json','complete_original16_local_rbm_known_only_coefficients'),
    'freeze':('frozen.json','complete_all_three_local_rbm16_candidates_frozen_before_recovered_audit'),
    'preaudit':('independent-preaudit-verification.json',verification.STATE),
    'audit':('independent-recovered-audit.json','complete_all_three_fixed_local_rbm16_recovered_audits'),
    'canonical':('independent-canonical-verification.json','complete_independent_local_rbm16_full_source_models_coefficients_and_frozen_group_scores'),
}


def command(phase):
    if phase not in PHASES:raise ValueError('Fixed full calibration phase required')
    name='upstream_local_rbm_blend.py' if phase in ('known','freeze') else 'verify_upstream_local_rbm_blend.py'
    return [sys.executable,'-u',str(ROOT/'scripts'/name),phase]


def check_sources(plan):
    if (plan['phases']!=list(PHASES) or plan['environment']!=ENV or plan['family_ids']!=study.BASE_IDS+[study.FAMILY_ID]
            or plan['candidates']!=list(study.CANDIDATES) or plan['primary']!=study.CANDIDATES[0]
            or plan['ridge_lambda']!=10. or plan['worker_cap_bytes']!=LIMIT):
        raise ValueError('Exact original sixteen-family known-only calibration recipe required')
    if plan['libraries']!={name:importlib.metadata.version(name) for name in ('numpy','scipy')}:
        raise ValueError('Original frozen numerical library versions required')
    for name,digest in plan['source_sha256'].items():
        if inputs.sha(ROOT/name)!=digest:raise ValueError('Original source changed: '+name)


def output_remaining():
    used=0;seen=set()
    if study.ART.exists():
        for path in study.ART.iterdir():
            if path.is_symlink() or not path.is_file():raise ValueError('Owned regular output payloads required')
            stat=path.stat();key=(stat.st_dev,stat.st_ino)
            if key not in seen:seen.add(key);used+=stat.st_size
    return max(rbm.OUTPUT_RESERVE,3*4*sum(study.COUNTS.values())+rbm.OUTPUT_RESERVE-used)


def continuation_ready(workers,pid,current,baseline,free,reserve):
    own=workers.get(str(pid))
    if not own or max(own['bytes'],own['rss_bytes'])>LIMIT:return False
    padded=dict(workers);padded[str(pid)]=dict(own,bytes=LIMIT,rss_bytes=LIMIT)
    return rbm.ready(padded,current,baseline,free,0,reserve)


class Guard:
    def __init__(self,phase):
        if phase not in PHASES:raise ValueError('Original scientific phase required')
        self.phase=phase;self.plan=rbm.read(REPORT/'plan.json');check_sources(self.plan)
        self.baseline=rbm.read(REPORT/'first-scientific-baseline.json');self.baseline_sha=inputs.sha(REPORT/'first-scientific-baseline.json')
        self.owner=rbm.read(REPORT/'monitor.json');self.gate=rbm.read(REPORT/(phase+'-gate.json'))
        self.captured=rbm.read(REPORT/(phase+'-ownership.json'));self.last=0.;self.counter=0
        if self.captured['pid']!=os.getpid() or not rbm.same_child(self.captured,rbm.identity.process(os.getpid())):
            raise ValueError('Actual dedicated original calibration worker required')

    def __call__(self,extra_disk_bytes=0):
        reserve=max(rbm.OUTPUT_RESERVE,extra_disk_bytes)
        while True:
            now=time.monotonic();free=0;error=None
            try:free=shutil.disk_usage(ROOT).free
            except Exception as exc:error=repr(exc)
            if not error and now-self.last<3 and free>=rbm.FLOOR+reserve:return
            check_sources(self.plan)
            if inputs.sha(REPORT/'first-scientific-baseline.json')!=self.baseline_sha:
                raise ValueError('Original first-scientific swap baseline changed')
            for name,digest in self.gate['predecessor_sha256'].items():
                if inputs.sha(ROOT/name)!=digest:raise ValueError('Original preceding scientific result changed')
            for name,digest in rbm.read(REPORT/'prerequisites.json')['source_sha256'].items():
                if inputs.sha(ROOT/name)!=digest:raise ValueError('Original full producer completion changed')
            try:
                workers,current=rbm.resources.footprints(os.getpid()),rbm.memory.memory();own=workers.get(str(os.getpid()))
                admitted=bool(not error and own and max(own['bytes'],own['rss_bytes'])<=LIMIT
                    and rbm.safe(workers,current,self.baseline,free,disk_reserve=reserve)
                    and rbm.identity.same_owner(self.owner,rbm.identity.process(self.owner['pid']))
                    and rbm.identity.same_owner(self.gate['watchdog'],rbm.identity.process(self.gate['watchdog']['pid'])))
                sample=dict(workers=workers,memory=current,free_bytes=free,disk_accounting_error=error)
            except Exception as exc:admitted=False;sample=dict(accounting_error=repr(exc),free_bytes=free)
            if admitted:self.last=now;return
            self.counter+=1
            record=dict(phase=self.phase,child=self.captured,sample=sample,extra_disk_reserve=reserve,
                original_baseline_sha256=self.baseline_sha,observed=rbm.stamp(),no_automatic_retry=True)
            try:rbm.status(REPORT/(self.phase+'-worker-resource-request.json'),record)
            except OSError:pass
            if not rbm.stop_child(self.captured):raise RuntimeError('Original calibration worker identity disappeared')
            # Same call stack continues here only under the original live owner.


def authorize(phase):
    if phase not in PHASES:raise ValueError('Fixed scientific phase required')
    plan=rbm.read(REPORT/'plan.json');check_sources(plan)
    owner=rbm.read(REPORT/'monitor.json');captured=rbm.read(REPORT/(phase+'-ownership.json'));gate=rbm.read(REPORT/(phase+'-gate.json'))
    actual=rbm.identity.process(os.getpid())
    if (not rbm.same_child(captured,actual) or captured['pid']!=os.getpid() or actual['command']!=' '.join(command(phase))
            or gate['pid']!=os.getpid() or gate['phase']!=phase or gate['command']!=command(phase)
            or gate['execution_plan_sha256']!=inputs.sha(REPORT/'plan.json')
            or gate['owner_sha256']!=inputs.sha(REPORT/'monitor.json')
            or gate['baseline_sha256']!=inputs.sha(REPORT/'first-scientific-baseline.json')
            or not rbm.identity.same_owner(owner,rbm.identity.process(owner['pid']))
            or not rbm.identity.same_owner(gate['watchdog'],rbm.identity.process(gate['watchdog']['pid']))
            or any(os.environ.get(key)!=value for key,value in ENV.items())):
        raise ValueError('Exact live source-bound owner/watchdog/worker/runtime dispatch gate required')
    for name,digest in gate['predecessor_sha256'].items():
        if inputs.sha(ROOT/name)!=digest:raise ValueError('Original frozen predecessor changed')
    return plan


def prerequisites():
    bindings={}
    for receipt,state,key,proof,pstate in (
        (rbm.REPORT/'completion.json','complete_full_own_gpu_rbm400_original15_15_and_independent_queries','verification_sha256',
         rbm.STUDY/'paired-verification.json',rbm.RESULTS['assemble'][1]),
        (ROOT/'experiments/upstream-own15-complete-trajectory-queue/completion.json','complete_own15_known_only_calibration_freeze_and_independent_recovered_audit','verification_sha256',
         study.preceding.REPORT/'independent-canonical-verification.json','complete_independent_own15_full_source_models_coefficients_and_frozen_group_scores')):
        record=rbm.read(receipt);result=rbm.read(proof)
        if record['state']!=state or record[key]!=inputs.sha(proof) or result['state']!=pstate:
            raise ValueError('Full independently verified original producer required')
        study.bind(bindings,receipt);study.bind(bindings,proof)
    return bindings


def frozen_plan():
    required=[Path(__file__),Path(study.__file__),Path(verification.__file__),study.DECL,
              ROOT/'scripts/test_upstream_local_rbm_blend.py',ROOT/'scripts/verify_upstream_own15_complete_trajectory_calibration.py',
              rbm.REPORT/'plan.json',ROOT/'experiments/upstream-local-rbm-blend-protocol/verification.json']
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.parent==ROOT/'scripts' and path.suffix=='.py':required.append(path)
    return dict(state='declared_own16_local_rbm_known_only_full_calibration_and_fixed_audits',created=rbm.stamp(),phases=list(PHASES),
        environment=ENV,family_ids=study.BASE_IDS+[study.FAMILY_ID],candidates=list(study.CANDIDATES),primary=study.CANDIDATES[0],
        ridge_lambda=10.,worker_cap_bytes=LIMIT,source_sha256={str(p.relative_to(ROOT)):inputs.sha(p) for p in sorted(set(required))},
        libraries={name:importlib.metadata.version(name) for name in ('numpy','scipy')},
        policy=dict(total_physical_and_rss_bytes=rbm.TOTAL_LIMIT,compressor_bytes=5*2**30,admission_compressor_bytes=4.5*2**30,
            swap_growth_bytes=128*2**20,disk_floor_bytes=rbm.FLOOR,stable_samples=3,remaining_output_reservation=True,
            same_process_pause_and_continuation=True,one_original_swap_baseline=True,primary_lane_only=True),
        labels='Known labels only until all65535 supports,17 ridge equations and six frozen vectors independently verify; all three fixed recovered audits and separate canonical scores; never select primary using recovered outcomes.')


def bootstrap(phase,owner_pid):
    plan=rbm.read(REPORT/'plan.json');check_sources(plan);owner=rbm.read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Original admitted supervisor required')
    captured=rbm.identity.process(os.getpid());captured['worker_command']=' '.join(command(phase))
    if captured['pgid']!=captured['pid']:raise ValueError('Original dedicated worker process group required')
    rbm.exclusive_json(REPORT/(phase+'-ownership.json'),captured);rbm.status(REPORT/'child-ownership.json',captured)
    while True:
        if not rbm.identity.same_owner(owner,rbm.identity.process(owner_pid)):
            os.killpg(captured['pid'],signal.SIGSTOP);continue
        path=REPORT/(phase+'-gate.json')
        if path.exists():
            gate=rbm.read(path)
            if (gate['pid']!=captured['pid'] or gate['command']!=command(phase)
                    or gate['execution_plan_sha256']!=inputs.sha(REPORT/'plan.json')
                    or not rbm.identity.same_owner(gate['watchdog'],rbm.identity.process(gate['watchdog']['pid']))):
                raise ValueError('Original live source-bound dispatch required')
            env=os.environ.copy();env.update(ENV);os.execvpe(sys.executable,command(phase),env)
        time.sleep(.1)


def watchdog(owner_pid,lane_fd):
    owner=rbm.read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact original owner required')
    try:
        while not (REPORT/'owner-release.json').exists():
            if not rbm.identity.same_owner(owner,rbm.identity.process(owner_pid)):
                path=REPORT/'child-ownership.json';captured=rbm.read(path) if path.exists() else None
                stopped=bool(captured and rbm.stop_child(captured))
                rbm.best_effort_json(REPORT/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,
                    original_primary_lane_retained=stopped,observed=rbm.stamp(),no_automatic_retry=True))
                while captured and rbm.same_child(captured,rbm.identity.process(captured['pid'])):time.sleep(2)
                return
            time.sleep(2)
    finally:os.close(lane_fd)


def main():
    plan=frozen_plan();check_sources(plan)
    if REPORT.exists() or study.REPORT.exists() or study.ART.exists():raise ValueError('Original scientific/control namespace retained; no retry')
    study.terminal.durable_directory(REPORT,ROOT,rbm.metadata_guard)
    rbm.exclusive_json(REPORT/'plan.json',plan);rbm.exclusive_json(REPORT/'monitor.json',rbm.identity.process(os.getpid()))
    lane=(ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a')
    child=captured=watch=None;baseline=None;completed={};pauses=0
    def check():
        check_sources(plan)
        if watch is not None and watch.poll() is not None:raise RuntimeError('Original watchdog unavailable')
    def interrupted(signum,frame):raise KeyboardInterrupt('Original calibration owner interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as output:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid()),'--lane-fd',str(lane.fileno())],
                    cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lane.fileno(),))
            while True:
                check()
                try:bindings=prerequisites();break
                except (FileNotFoundError,json.JSONDecodeError):
                    rbm.status(REPORT/'status.json',dict(state='waiting_full_actual_RBM_and_original_own15_verification',observed=rbm.stamp()));time.sleep(10)
            rbm.exclusive_json(REPORT/'prerequisites.json',dict(source_sha256=bindings,score_values_used=False,observed=rbm.stamp()))
            while True:
                check()
                try:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            for phase in PHASES:
                good=0
                while good<3:
                    check()
                    try:
                        workers,current=rbm.resources.footprints(os.getpid()),rbm.memory.memory()
                        ok=rbm.ready(workers,current,baseline,shutil.disk_usage(ROOT).free,LIMIT,output_remaining())
                    except Exception:ok=False
                    good=good+1 if ok else 0
                    rbm.status(REPORT/'status.json',dict(state='waiting_original_CPU_headroom',phase=phase,stable_samples=good,observed=rbm.stamp()))
                    if good<3:time.sleep(10)
                if baseline is None:
                    baseline=rbm.memory.memory();rbm.exclusive_json(REPORT/'first-scientific-baseline.json',baseline)
                with (REPORT/(phase+'.log')).open('x') as output:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--bootstrap',str(os.getpid()),'--phase',phase],
                        cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    ownership=REPORT/(phase+'-ownership.json')
                    while not ownership.exists():
                        check()
                        if child.poll() is not None:raise RuntimeError('Original scientific bootstrap failed')
                        time.sleep(.1)
                    captured=rbm.read(ownership)
                    if captured['pid']!=child.pid or not rbm.same_child(captured,rbm.identity.process(child.pid)):raise ValueError('Original bootstrap identity required')
                    rbm.exclusive_json(REPORT/(phase+'-gate.json'),dict(pid=child.pid,phase=phase,command=command(phase),
                        execution_plan_sha256=inputs.sha(REPORT/'plan.json'),owner_sha256=inputs.sha(REPORT/'monitor.json'),
                        baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json'),watchdog=rbm.identity.process(watch.pid),predecessor_sha256=completed))
                    while child.poll() is None:
                        check();actual=rbm.identity.process(child.pid)
                        if actual is None:
                            if child.poll() is not None:break
                            time.sleep(.2);continue
                        try:
                            workers,current=rbm.resources.footprints(child.pid),rbm.memory.memory();own=workers.get(str(child.pid))
                            allowed=bool(own and max(own['bytes'],own['rss_bytes'])<=LIMIT and rbm.safe(workers,current,baseline,shutil.disk_usage(ROOT).free))
                        except Exception:allowed=False
                        if not allowed or 'T' in actual['state']:
                            if not rbm.stop_child(captured):continue
                            pauses+=1;good=0
                            rbm.best_effort_json(REPORT/('pause-%06d.json'%pauses),dict(phase=phase,child=captured,observed=rbm.stamp(),baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json')))
                            while good<3:
                                check()
                                if child.poll() is not None:raise RuntimeError('Preserved scientific worker became terminal')
                                try:
                                    workers,current=rbm.resources.footprints(child.pid),rbm.memory.memory()
                                    request=REPORT/(phase+'-worker-resource-request.json');needed=output_remaining()
                                    if request.exists():needed=max(needed,rbm.read(request)['extra_disk_reserve'])
                                    ok=continuation_ready(workers,child.pid,current,baseline,shutil.disk_usage(ROOT).free,needed)
                                except Exception:ok=False
                                good=good+1 if ok else 0
                                try:rbm.status(REPORT/'status.json',dict(state='same_CPU_worker_RAM_preserved_waiting_original_readmission',phase=phase,stable_samples=good,observed=rbm.stamp()))
                                except OSError:pass
                                if good<3:time.sleep(10)
                            if not rbm.same_child(captured,rbm.identity.process(child.pid)):raise ValueError('Same original scientific worker required')
                            rbm.exclusive_json(REPORT/('continuation-%06d.json'%pauses),dict(phase=phase,child=captured,baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json'),observed=rbm.stamp()))
                            os.killpg(child.pid,signal.SIGCONT)
                        rbm.status(REPORT/'status.json',dict(state=phase,pid=child.pid,observed=rbm.stamp()));time.sleep(3)
                    if child.returncode:raise RuntimeError('Original scientific phase failed; no retry: '+phase)
                path=study.REPORT/RESULTS[phase][0];result=rbm.read(path)
                if result['state']!=RESULTS[phase][1]:raise ValueError('Complete original phase proof required')
                completed[str(path.relative_to(ROOT))]=inputs.sha(path)
                rbm.exclusive_json(REPORT/(phase+'-completion.json'),dict(phase=phase,result_sha256=inputs.sha(path),
                    execution_plan_sha256=inputs.sha(REPORT/'plan.json'),baseline_sha256=inputs.sha(REPORT/'first-scientific-baseline.json'),worker=captured,observed=rbm.stamp()))
                child=captured=None
            proof=study.REPORT/RESULTS['canonical'][0]
            rbm.exclusive_json(REPORT/'completion.json',dict(state='complete_local_rbm16_known_only_calibration_freeze_full_replay_and_independent_recovered_audit',
                verification_sha256=inputs.sha(proof),qualifying_labels_used_for_fitting_or_selection=False,observed=rbm.stamp()))
            rbm.exclusive_json(REPORT/'owner-release.json',dict(state='all_original_scientific_children_terminal_after_canonical_verification'))
    except BaseException as exc:
        stopped=bool(captured and child is not None and child.poll() is None and rbm.stop_child(captured))
        rbm.status(REPORT/'failure.json',dict(error=repr(exc),child=captured,actual_child_RAM_preserved=stopped,no_automatic_retry=True,observed=rbm.stamp()))
        if child is None or child.poll() is not None:rbm.exclusive_json(REPORT/'owner-release.json',dict(state='no_live_scientific_child_at_terminal_owner_failure'))
        raise
    finally:lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true')
    parser.add_argument('--watch',type=int);parser.add_argument('--lane-fd',type=int)
    parser.add_argument('--bootstrap',type=int);parser.add_argument('--phase',choices=PHASES);args=parser.parse_args()
    if args.watch:watchdog(args.watch,args.lane_fd)
    elif args.bootstrap:
        if not args.phase:parser.error('Fixed phase required')
        bootstrap(args.phase,args.bootstrap)
    elif args.start:main()
    else:parser.error('Explicit --start required; importing never starts numerical work')
