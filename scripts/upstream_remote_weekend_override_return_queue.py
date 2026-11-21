#!/usr/bin/env python3
"""Separate guarded full-return supervisor for the pinned remote policy variant."""
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
import upstream_remote_weekend_override_runtime as r

PREREQUISITE=(
 'upstream-own15-complete-trajectory-queue/completion.json',
 'complete_own15_known_only_calibration_freeze_and_independent_recovered_audit',
 'upstream-own15-complete-trajectory-calibration/independent-canonical-verification.json',
 'complete_independent_own15_full_source_models_coefficients_and_frozen_group_scores')


def prepare():
    from upstream_remote_weekend_override_context import Context
    context=Context()
    if r.PROTOCOL.parent.exists():raise ValueError('Existing production declaration retained')
    required=[Path(__file__),Path(r.__file__),r.ROOT/'scripts/upstream_remote_weekend_override_worker.py',
        r.ROOT/'scripts/upstream_remote_weekend_override_handoff.py',
        r.ROOT/'scripts/test_remote_weekend_override_runtime.py',
        r.ROOT/'experiments/upstream-remote-weekend-override-runtime-cpu/verification.json',
        r.ROOT/'scripts/upstream_remote_weekend_return_queue.py',
        r.ROOT/'scripts/upstream_split3_pilot.py',
        r.ROOT/'scripts/upstream_remote15h_native_stream.py',
        r.ROOT/'REMOTE_WEEKEND_OWNER_OVERRIDE_RETURN_PLAN.md',
        *[Path(m.__file__) for m in (r.u,r.ops,r.identity,r.memory,r.resources)]]
    bindings=dict(context.bindings)
    r.ops.merge_bindings(bindings,{str(p.relative_to(r.ROOT)):r.ops.sha(p) for p in required})
    r.durable_directory(r.PROTOCOL.parent)
    parent=r.ROOT/'experiments/upstream-remote-weekend-override-return-queue'
    if not parent.exists():r.durable_directory(parent)
    r.write(r.PROTOCOL,dict(state='declared_full_weekend_owner_override_return_adapter',created=r.stamp(),
        jobs=list(r.JOBS),phases=list(r.PHASES),environment=r.ENV,worker_cap_bytes=r.LIMIT,
        total_cap_bytes=r.TOTAL,disk_floor_bytes=r.FLOOR,numpy_version=importlib.metadata.version('numpy'),
        source_sha256=bindings,prerequisite=list(PREREQUISITE),
        policy=dict(compressor_bytes=5*2**30,admission_compressor_bytes=4.5*2**30,
                    swap_growth_bytes=128*2**20,stable_samples=3,one_first_scientific_baseline=True,
                    primary_lane_only=True,watchdog_and_worker_inherit_lane=True,
                    no_native_training=True,no_numerical_subset=True,no_automatic_retry=True,
                    original_remote_resource_guarantees=False,coordinator_local_guards_unchanged=True)),exclusive=True)
    r.check_protocol();print('Full return protocol declared; no worker launched',flush=True)


def prerequisite():
    receipt,state,proof,pstate=PREREQUISITE
    receipt,proof=r.ROOT/'experiments'/receipt,r.ROOT/'experiments'/proof
    a,b=r.read(receipt),r.read(proof)
    if a['state']!=state or a['verification_sha256']!=r.ops.sha(proof) or b['state']!=pstate:
        raise ValueError('Full original own15 independent completion required, without score selection')
    return {str(p.relative_to(r.ROOT)):r.ops.sha(p) for p in (receipt,proof)}


def bootstrap(job,phase,owner_pid,lane_fd):
    q,_,_=r.paths(job);r.check_protocol();plan=r.read(q/'plan.json');r.check_bindings(plan['source_sha256'])
    owner=r.read(q/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact declared return owner required')
    os.set_inheritable(lane_fd,True);r.validate_lane(lane_fd)
    captured=r.identity.process(os.getpid());captured['worker_command']=' '.join(r.worker_command(job,phase,lane_fd))
    if captured['pgid']!=captured['pid']:raise ValueError('Dedicated inherited-lane bootstrap required')
    r.write(q/(phase+'-ownership.json'),captured,exclusive=True);r.write(q/'child-ownership.json',captured)
    gate_path=q/(phase+'-gate.json')
    while True:
        if not r.identity.same_owner(owner,r.identity.process(owner_pid)):
            os.killpg(os.getpid(),signal.SIGSTOP);continue
        if gate_path.exists():
            gate=r.read(gate_path)
            if (gate['pid']!=os.getpid() or gate['command']!=r.worker_command(job,phase,lane_fd)
                or gate['owner_sha256']!=r.ops.sha(q/'monitor.json')
                or not r.identity.same_owner(gate['watchdog'],r.identity.process(gate['watchdog']['pid']))):
                raise ValueError('Exact live owner/watchdog gate required')
            r.write(q/(phase+'-dispatch.json'),dict(command=gate['command'],pid=os.getpid(),observed=r.stamp()),exclusive=True)
            env=os.environ.copy();env.update(r.ENV);os.execvpe(sys.executable,gate['command'],env)
        time.sleep(.1)


def watchdog(job,owner_pid,lane_fd):
    q,_,_=r.paths(job);owner=r.read(q/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact return owner required')
    try:
        r.write(q/'watch-ready.json',dict(watchdog=r.identity.process(os.getpid()),owner_sha256=r.ops.sha(q/'monitor.json'),
            inherited_lane_fd=lane_fd,observed=r.stamp()),exclusive=True)
        while True:
            if (q/'owner-release.json').exists():return
            if not r.identity.same_owner(owner,r.identity.process(owner_pid)):
                try:captured=r.read(q/'child-ownership.json')
                except (OSError,ValueError):captured=None
                stopped=r.stop(captured)
                r.best_effort(q/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,
                    inherited_primary_lane_retained=True,worker_also_inherits_lane=True,observed=r.stamp()))
                while captured and r.same_child(captured,r.identity.process(captured['pid'])):time.sleep(2)
                # A not-yet-journaled bootstrap independently holds the same FD and self-stops before exec.
                return
            time.sleep(2)
    finally:os.close(lane_fd)


def main(job):
    from upstream_remote_weekend_override_context import Context
    from upstream_remote_weekend_override_handoff import verify_handoff
    protocol=r.check_protocol();handoff=verify_handoff(job)
    q,report,art=r.paths(job)
    if q.exists() or report.exists() or art.exists():raise ValueError('Existing control/full/partial namespace retained')
    r.durable_directory(q)
    bindings={**protocol['source_sha256'],str(r.PROTOCOL.relative_to(r.ROOT)):r.ops.sha(r.PROTOCOL),
              str(handoff.relative_to(r.ROOT)):r.ops.sha(handoff)}
    r.write(q/'plan.json',dict(job=job,protocol_sha256=r.ops.sha(r.PROTOCOL),source_sha256=bindings,
                             original_idle_handoff_sha256=r.ops.sha(handoff),created=r.stamp()),exclusive=True)
    r.write(q/'monitor.json',r.identity.process(os.getpid()),exclusive=True)
    lane=(r.ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a')
    child=captured=watch=None;baseline=None;completed={};pauses=0
    def check():
        r.check_protocol();r.check_bindings(bindings)
        if watch is not None and watch.poll() is not None:raise RuntimeError('Return watchdog unavailable')
    def interrupted(signum,frame):raise KeyboardInterrupt('Separate full-return owner interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (q/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (q/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--job',job,
                    '--watch',str(os.getpid()),'--lane-fd',str(lane.fileno())],cwd=r.ROOT,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(lane.fileno(),))
            while not (q/'watch-ready.json').exists():
                check();time.sleep(.1)
            while True:
                check()
                try:prereqs=prerequisite();break
                except (FileNotFoundError,json.JSONDecodeError):
                    r.write(q/'status.json',dict(state='waiting_full_original_own15_independent_completion',observed=r.stamp()));time.sleep(10)
            r.write(q/'prerequisites.json',dict(source_sha256=prereqs,score_values_used=False),exclusive=True)
            context=Context()
            while True:
                check()
                try:
                    fetched=subprocess.run(['git','fetch','--quiet','origin','research/remote-weekend-execution'],cwd=r.ROOT,
                        capture_output=True,text=True,timeout=40)
                    if fetched.returncode:raise RuntimeError(fetched.stderr[-1000:])
                    revision=subprocess.check_output(['git','rev-parse','origin/research/remote-weekend-execution'],cwd=r.ROOT,text=True).strip()
                except (subprocess.TimeoutExpired,RuntimeError) as exc:
                    r.write(q/'status.json',dict(state='transient_remote_poll_failure_no_worker',error=repr(exc),observed=r.stamp()));time.sleep(30);continue
                target=revision+':'+context.transport.PREFIX+job+'/transfer.json'
                if subprocess.run(['git','cat-file','-e',target],cwd=r.ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
                    context.transport.validate_metadata(context.transport.job_spec(job),context.metadata(revision,job));break
                r.write(q/'status.json',dict(state='waiting_complete_pinned_variant_transfer',last_remote_commit=revision,observed=r.stamp()));time.sleep(30)
            r.write(q/'pinned-remote.json',dict(commit=revision,job=job,observed=r.stamp()),exclusive=True)
            while True:
                check()
                try:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(10)
            for phase in r.PHASES:
                good=0
                while good<3:
                    check()
                    try:ok=r.ready(r.resources.footprints(os.getpid()),r.memory.memory(),baseline,
                                  shutil.disk_usage(r.ROOT).free,r.LIMIT,r.output_remaining(job))
                    except Exception:ok=False
                    good=good+1 if ok else 0
                    r.write(q/'status.json',dict(state='waiting_original_local_headroom',phase=phase,stable_samples=good,observed=r.stamp()))
                    if good<3:time.sleep(10)
                if baseline is None:
                    baseline=r.memory.memory();r.write(q/'first-scientific-baseline.json',baseline,exclusive=True)
                with (q/(phase+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--job',job,'--bootstrap',
                        str(os.getpid()),'--phase',phase,'--lane-fd',str(lane.fileno())],cwd=r.ROOT,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lane.fileno(),))
                    ownership=q/(phase+'-ownership.json')
                    while not ownership.exists():
                        check()
                        if child.poll() is not None:raise RuntimeError('Scientific bootstrap failed before gate')
                        time.sleep(.1)
                    captured=r.read(ownership)
                    if captured['pid']!=child.pid or not r.same_child(captured,r.identity.process(child.pid)):
                        raise ValueError('Actual dedicated inherited-lane worker required')
                    r.write(q/(phase+'-gate.json'),dict(pid=child.pid,phase=phase,command=r.worker_command(job,phase,lane.fileno()),
                        protocol_sha256=r.ops.sha(r.PROTOCOL),queue_plan_sha256=r.ops.sha(q/'plan.json'),owner_sha256=r.ops.sha(q/'monitor.json'),
                        baseline_sha256=r.ops.sha(q/'first-scientific-baseline.json'),watchdog=r.identity.process(watch.pid),
                        predecessor_sha256=completed),exclusive=True)
                    while child.poll() is None:
                        check();actual=r.identity.process(child.pid)
                        if actual is None:
                            if child.poll() is not None:break
                            time.sleep(.2);continue
                        try:
                            workers,current=r.resources.footprints(child.pid),r.memory.memory();own=workers.get(str(child.pid))
                            valid=bool(own and max(own['bytes'],own['rss_bytes'])<=r.LIMIT and
                                r.safe(workers,current,baseline,shutil.disk_usage(r.ROOT).free,disk_reserve=r.output_remaining(job)))
                        except Exception:valid=False
                        if not valid or 'T' in actual['state']:
                            r.stop(captured);pauses+=1;good=0
                            r.best_effort(q/('pause-%06d.json'%pauses),dict(child=captured,phase=phase,observed=r.stamp()))
                            while good<3:
                                check()
                                if child.poll() is not None:raise RuntimeError('Preserved worker became terminal')
                                try:ok=r.continuation_ready(r.resources.footprints(child.pid),child.pid,r.memory.memory(),
                                    baseline,shutil.disk_usage(r.ROOT).free,r.output_remaining(job))
                                except Exception:ok=False
                                good=good+1 if ok else 0
                                r.best_effort(q/'status.json',dict(state='same_worker_RAM_preserved_waiting_original_readmission',
                                    phase=phase,stable_samples=good,observed=r.stamp()))
                                if good<3:time.sleep(10)
                            if not r.same_child(captured,r.identity.process(child.pid)):raise ValueError('Same preserved worker required')
                            r.write(q/('continuation-%06d.json'%pauses),dict(child=captured,phase=phase,
                                baseline_sha256=r.ops.sha(q/'first-scientific-baseline.json'),observed=r.stamp()),exclusive=True)
                            os.killpg(child.pid,signal.SIGCONT)
                        r.write(q/'status.json',dict(state=phase,pid=child.pid,observed=r.stamp()));time.sleep(3)
                    if child.returncode:raise RuntimeError('Scientific return phase failed; no retry: '+phase)
                result=report/('variant-transport-context.json' if phase=='transport' else 'independent-verification.json')
                completed[str(result.relative_to(r.ROOT))]=r.ops.sha(result)
                r.write(q/(phase+'-completion.json'),dict(result_sha256=r.ops.sha(result),worker=captured,
                    baseline_sha256=r.ops.sha(q/'first-scientific-baseline.json'),observed=r.stamp()),exclusive=True)
                child=captured=None
            proof=report/'independent-verification.json';result=r.read(proof)
            if (result['state']!='complete_independent_weekend_full_source_queries_projection_disjointness_and_all_compact_bits'
                or result['protocol_variant']!='pinned_remote_owner_resource_override_original_numerical_blueprint'):
                raise ValueError('Full independently verified variant producer required')
            completion=dict(state='complete_full_weekend_return_transport_and_independent_admission',job=job,
                remote_commit=revision,verification_sha256=r.ops.sha(proof),qualifying_labels_read=False,
                qualifying_labels_used_for_fitting_or_selection=False,original_remote_resource_guarantees=False,
                coordinator_local_guards_unchanged=True,observed=r.stamp())
            r.write(q/'completion.json',completion,exclusive=True)
            legacy=r.ROOT/'experiments/upstream-remote-weekend-return-queue'/job/'completion.json'
            completion=dict(completion,successor_completion=str((q/'completion.json').relative_to(r.ROOT)),
                            successor_completion_sha256=r.ops.sha(q/'completion.json'))
            r.write(legacy,completion,exclusive=True)
            r.write(q/'status.json',dict(state='complete',observed=r.stamp()))
            r.write(q/'owner-release.json',dict(state='all_owned_workers_terminal_after_full_independent_variant_proof'),exclusive=True)
    except BaseException as exc:
        stopped=False
        if child is not None and child.poll() is None:
            if captured is None:os.killpg(child.pid,signal.SIGSTOP);stopped=True
            else:stopped=r.stop(captured)
        r.best_effort(q/'failure.json',dict(error=repr(exc),child=captured,actual_child_RAM_preserved=stopped,
            inherited_lane_also_held_by_worker=True,no_automatic_retry=True,observed=r.stamp()))
        if child is None or child.poll() is not None:
            r.best_effort(q/'owner-release.json',dict(state='no_live_scientific_child_at_owner_failure'))
        raise
    finally:lane.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--start',action='store_true')
    p.add_argument('--job',choices=r.JOBS);p.add_argument('--watch',type=int);p.add_argument('--bootstrap',type=int)
    p.add_argument('--lane-fd',type=int);p.add_argument('--phase',choices=r.PHASES);a=p.parse_args()
    if a.prepare:prepare()
    elif a.watch and a.job and a.lane_fd is not None:watchdog(a.job,a.watch,a.lane_fd)
    elif a.bootstrap and a.job and a.phase and a.lane_fd is not None:bootstrap(a.job,a.phase,a.bootstrap,a.lane_fd)
    elif a.start and a.job:main(a.job)
    else:p.error('Explicit declared prepare/start/watch/bootstrap mode required')
