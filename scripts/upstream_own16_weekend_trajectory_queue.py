#!/usr/bin/env python3
"""Known-only15-family fit, freeze, full preaudit and separate recovered audit."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_weekend_trajectory_ops as ops
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory
import upstream_cf96_primary_lane as identity

REPORT = u.ROOT/'experiments/upstream-own16-weekend-trajectory-queue'
PHASES = {
    'known': ('upstream_own16_weekend_trajectory_calibration.py','known'),
    'freeze': ('upstream_own16_weekend_trajectory_calibration.py','freeze'),
    'independent_preaudit': ('verify_upstream_own16_weekend_trajectory_calibration.py','preaudit'),
    'audit': ('verify_upstream_own16_weekend_trajectory_calibration.py','audit'),
    'separate_canonical': ('verify_upstream_own16_weekend_trajectory_calibration.py','canonical'),
}
PREREQUISITES = [
    ('upstream-own15-complete-trajectory-calibration/independent-preaudit-verification.json',
     'complete_independent_all32767_supports_ridge16_and_six_vectors_before_audit','freeze_sha256',
     'upstream-own15-complete-trajectory-calibration/frozen.json',
     'complete_all_three_own15_candidates_frozen_before_any_new_audit'),
    ('upstream-weekend-trajectory-fold-bag-queue/completion.json',
     'complete_full_trajectory_qualifying_bag_and_independent_replay','verification_sha256',
     'upstream-weekend-trajectory-fold-bag/independent-verification.json',
     'complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag'),
]



def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2)+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe(workers, current, baseline=None, reserve=0, output_reserve=0):
    return (sum(w['bytes'] for w in workers.values())+reserve < 24*2**30 and
            sum(w['rss_bytes'] for w in workers.values())+reserve < 24*2**30 and
            current['compressor_bytes'] < 5*2**30 and
            (baseline is None or current['swap_used_bytes']-baseline['swap_used_bytes'] <= 128*2**20) and
            shutil.disk_usage(u.ROOT).free >= 18.75*2**30+output_reserve)


def command(phase):
    script, action = PHASES[phase]
    return [sys.executable, '-u', str(u.ROOT/'scripts'/script), action]


def same_child(captured, actual):
    return bool(actual and actual['pid'] == captured['pid'] and actual['started'] == captured['started'] and
                actual['pgid'] == captured['pid'] and actual['command'] in [captured['command'], captured.get('worker_command')])


def bootstrap(phase, owner_pid):
    owner = u.read(REPORT/'monitor.json')
    if owner['pid'] != owner_pid: raise ValueError('Exact admitted trajectory owner required')
    plan = u.read(REPORT/'plan.json')
    for name, digest in plan['source_sha256'].items():
        if ops.sha(u.ROOT/name) != digest: raise ValueError('Frozen source changed before dispatch')
    captured = identity.process(os.getpid()); captured['worker_command'] = ' '.join(command(phase))
    if captured['pgid'] != captured['pid']: raise ValueError('Dedicated scientific bootstrap required')
    write(REPORT/(phase+'-ownership.json'), captured)
    write(REPORT/'child-ownership.json', captured)
    gate = REPORT/(phase+'-gate.json')
    while True:
        if not identity.same_owner(owner, identity.process(owner_pid)):
            os.killpg(os.getpid(), signal.SIGSTOP); continue
        if gate.exists():
            permission = u.read(gate)
            if (permission['pid'] != os.getpid() or permission['command'] != command(phase) or
                permission['owner_sha256'] != ops.sha(REPORT/'monitor.json') or
                not identity.same_owner(permission['watchdog'], identity.process(permission['watchdog']['pid']))):
                raise ValueError('Exact live owner/watchdog/scientific command gate required')
            write(REPORT/(phase+'-dispatch.json'), dict(pid=os.getpid(), command=command(phase), observed=stamp()))
            env = os.environ.copy()
            for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']: env[key] = '2'
            os.execvpe(sys.executable, command(phase), env)
        time.sleep(.1)


def watchdog(owner_pid):
    owner = u.read(REPORT/'monitor.json')
    if owner['pid'] != owner_pid: raise ValueError('Exact trajectory monitor required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner, identity.process(owner_pid)):
            if (REPORT/'owner-release.json').exists(): return
            path = REPORT/'child-ownership.json'; captured = u.read(path) if path.exists() else None
            stopped = False
            if captured and same_child(captured, identity.process(captured['pid'])):
                os.killpg(captured['pid'], signal.SIGSTOP); stopped = True
            write(REPORT/'watchdog-recovery.json', dict(child=captured, actual_child_RAM_preserved=stopped, observed=stamp()))
            return
        time.sleep(2)


def main():
    if REPORT.exists(): raise ValueError('Existing full trajectory queue/evidence preserved')
    REPORT.mkdir(); child = None; captured = None; watch = None; lane = None; baseline = None
    files = [Path(__file__), Path(ops.__file__), Path(resources.__file__), Path(memory.__file__), Path(identity.__file__),
             u.ROOT/'WEEKEND_COMPLETE_TRAJECTORY_PLAN.md', u.ROOT/'WEEKEND_COMPLETE_TRAJECTORY_PLAN.md',
             u.ROOT/'scripts/upstream_own16_weekend_trajectory_calibration.py',u.ROOT/'scripts/verify_upstream_own16_weekend_trajectory_calibration.py',
             u.ROOT/'scripts/upstream_own15_complete_trajectory_calibration.py',u.ROOT/'scripts/verify_upstream_own15_complete_trajectory_calibration.py',
             u.ROOT/'experiments/upstream-own391-paired-data/independent-verification.json']
    bindings = {str(path.relative_to(u.ROOT)): ops.sha(path) for path in files}
    write(REPORT/'monitor.json', identity.process(os.getpid()))
    write(REPORT/'plan.json', dict(source_sha256=bindings, created=stamp(), phases=PHASES, prerequisites=PREREQUISITES,
        rule='Wait full preceding15-family label-free preaudit and new complete-trajectory bag/model proof. Primarylane only,noGPU,2GiB worker/24physicalRSS/5compress(admit4.5)/128MiB firstscientific swapgrowth/18.75GiBdisk+64MiBoutput admission. Retain all16 known-only families, uniformSLSQP simplex65535supports/fullactualKKT andobjective evenifnonunique, defaultlambda10ridge17equations/unpenalizedbias andfixedhalfcontrol. Freezeall6 vectors andfullsourceclosure beforefixedrecoveredlabels, all3frozenaudits andseparatecanonical. No Qweightfit/selection/familydrop. Same owned RAM preserved onguard/ownerloss,noretry/refit/reducedrecipe.'))
    def check():
        for name, digest in bindings.items():
            if ops.sha(u.ROOT/name) != digest: raise ValueError('Frozen trajectory source expectation changed')
    def interrupted(signum, frame): raise KeyboardInterrupt('Trajectory queue interrupted')
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as stream:
                watch = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--watch', str(os.getpid())],
                    cwd=u.ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            for rel, state, key, proof_rel, proof_state in PREREQUISITES:
                receipt, proof = u.ROOT/'experiments'/rel, u.ROOT/'experiments'/proof_rel
                while True:
                    check()
                    if watch.poll() is not None: raise RuntimeError('Trajectory watchdog unavailable while waiting')
                    try: record = u.read(receipt)
                    except (FileNotFoundError, json.JSONDecodeError):
                        write(REPORT/'status.json', dict(state='waiting_full_producer_and_independent_proof', prerequisite=rel, observed=stamp()))
                        time.sleep(10); continue
                    break
                if record['state'] != state or record[key] != ops.sha(proof) or u.read(proof)['state'] != proof_state:
                    raise ValueError('Full independent original producer completion required')
                for path in [receipt, proof]: bindings[str(path.relative_to(u.ROOT))] = ops.sha(path)
            lane = (u.ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a')
            while True:
                check()
                try: fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    write(REPORT/'status.json', dict(state='waiting_serialized_primary_lane', observed=stamp())); time.sleep(10)
            for phase in PHASES:
                good = 0
                while good < 3:
                    check(); workers, current = resources.footprints(os.getpid()), memory.memory()
                    if watch.poll() is not None: raise RuntimeError('Trajectory watchdog unavailable before phase')
                    ready = safe(workers, current, baseline, 2*2**30, 64*2**20) and current['compressor_bytes'] < 4.5*2**30
                    good = good+1 if ready else 0
                    write(REPORT/'status.json', dict(state='waiting_original_trajectory_headroom', phase=phase, good_samples=good,
                                                   workers=workers, memory=current, observed=stamp()))
                    if good < 3: time.sleep(10)
                if baseline is None:
                    baseline = memory.memory(); write(REPORT/'first-scientific-baseline.json', baseline)
                # Remove the old terminal ownership only; every old dispatch is retained.
                ownership = REPORT/'child-ownership.json'
                if ownership.exists():
                    old = u.read(ownership)
                    if same_child(old, identity.process(old['pid'])): raise ValueError('Prior scientific worker must be terminal')
                    ownership.unlink()
                with (REPORT/(phase+'.log')).open('x') as stream:
                    captured = None
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--bootstrap', str(os.getpid()), '--phase', phase],
                        cwd=u.ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    while not ownership.exists():
                        if child.poll() is not None or watch.poll() is not None: raise RuntimeError('Scientific launch handshake unavailable')
                        time.sleep(.1)
                    captured = u.read(ownership)
                    if captured['pid'] != child.pid or not same_child(captured, identity.process(child.pid)):
                        raise ValueError('Exact dedicated scientific bootstrap required')
                    watcher = identity.process(watch.pid)
                    if not watcher or watch.poll() is not None: raise RuntimeError('Live watchdog required before dispatch')
                    write(REPORT/(phase+'-gate.json'), dict(pid=child.pid, command=command(phase),
                        owner_sha256=ops.sha(REPORT/'monitor.json'), watchdog=watcher))
                    while child.poll() is None:
                        actual = identity.process(child.pid)
                        if actual and 'T' in actual['state']:
                            raise RuntimeError('Scientific worker self-suspended; preserve same process without retry')
                        check(); workers, current = resources.footprints(child.pid), memory.memory()
                        if child.poll() is not None: break
                        own = workers.get(str(child.pid))
                        if not own or max(own['bytes'], own['rss_bytes']) > 2*2**30 or not safe(workers, current, baseline) or watch.poll() is not None:
                            write(REPORT/'guard-sample.json', dict(phase=phase, workers=workers, memory=current, observed=stamp()))
                            raise RuntimeError('Full trajectory resource/watchdog guard; actual child RAM preserved')
                        write(REPORT/'status.json', dict(state=phase, pid=child.pid, workers=workers, memory=current, observed=stamp())); time.sleep(3)
                    if child.returncode: raise RuntimeError('Full trajectory phase failed; no retry: '+phase)
                    child = None; captured = None
            proof = u.ROOT/'experiments/upstream-own16-weekend-trajectory-calibration/independent-canonical-verification.json'
            if u.read(proof)['state'] != 'complete_independent_own16_full_source_models_coefficients_and_frozen_group_scores':
                raise ValueError('Full separate original data/ridge proof required')
            write(REPORT/'completion.json', dict(state='complete_own16_known_only_calibration_freeze_and_independent_recovered_audit',
                verification_sha256=ops.sha(proof), qualifying_labels_read=True,
                qualifying_labels_used_for_fitting_or_selection=False, observed=stamp()))
            write(REPORT/'owner-release.json', dict(state='all_owned_scientific_children_terminal_after_independent_proof'))
            write(REPORT/'status.json', dict(state='complete', observed=stamp()))
            write(REPORT/'owner-release.json',dict(state='all_owned_workers_terminal_after_full_own16_canonical_proof'))
    except BaseException as error:
        stopped = False
        if child is not None and child.poll() is None:
            if captured is None or same_child(captured, identity.process(child.pid)):
                os.killpg(child.pid, signal.SIGSTOP); stopped = True
        write(REPORT/'failure.json', dict(error=repr(error), child=captured, actual_child_RAM_preserved=stopped,
                                        no_automatic_retry=True, observed=stamp()))
        raise
    finally:
        if lane is not None: lane.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--watch', type=int); parser.add_argument('--bootstrap', type=int)
    parser.add_argument('--phase', choices=list(PHASES)); args=parser.parse_args()
    if args.bootstrap:
        if not args.phase: parser.error('bootstrap requires declared phase')
        bootstrap(args.phase, args.bootstrap)
    elif args.watch: watchdog(args.watch)
    else: main()
