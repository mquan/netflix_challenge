#!/usr/bin/env python3
"""Recover original completed own352 models using bounded independent verification."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory
import upstream_cf96_primary_lane as identities

REPORT = u.ROOT / 'experiments/upstream-own352-bounded-replay'
WORKER = u.ROOT / 'scripts/verify_upstream_own352_bounded_scalers.py'
VERIFIER = u.ROOT / 'scripts/verify_upstream_own352_temporal_canonical.py'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers, current, baseline, reserve=0):
    return (sum(v['bytes'] for v in workers.values())+reserve < 24*2**30
        and sum(v['rss_bytes'] for v in workers.values())+reserve < 24*2**30
        and current['swap_used_bytes']-baseline['swap_used_bytes'] <= 128*2**20
        and current['compressor_bytes'] < 5*2**30
        and shutil.disk_usage(u.ROOT).free >= 18.75*2**30)


def watchdog(supervisor):
    owner = u.read(REPORT / 'monitor.json')
    if owner['pid'] != supervisor:
        raise ValueError('Actual captured monitor required')
    while True:
        if (REPORT / 'owner-release.json').exists():
            return
        if not identities.same_owner(owner, identities.process(supervisor)):
            if (REPORT / 'owner-release.json').exists():
                return
            path = REPORT / 'child-ownership.json'
            captured = u.read(path) if path.exists() else None
            stopped = False
            if captured and identities.same_owner(captured, identities.process(captured['pid'])):
                os.killpg(captured['pid'], signal.SIGSTOP); stopped = True
            u.write(REPORT / 'watchdog-recovery.json', dict(
                state='monitor_disappeared_original_child_RAM_preserved', child=captured, stopped=stopped))
            return
        time.sleep(2)


def completed_fits():
    original=u.ROOT/'experiments/upstream-own352-temporal-mps'
    oldqueue=u.ROOT/'experiments/upstream-own352-temporal-queue'
    failure=u.read(oldqueue/'failure.json');expected=failure['child']
    actual=identities.process(expected['pid'])
    if (not identities.same_owner(expected,actual) or 'T' not in actual['state'] or
        actual['pgid']!=actual['pid'] or identities.children(actual['pid']) or
        'scripts/verify_upstream_own352_temporal_mps.py' not in actual['command']):
        raise ValueError('Only exact already-suspended childless read-only original verifier may retire')
    if u.read(original/'fit-completion.json')['state']!='complete_all_five_original_budget_own352_fits':
        raise ValueError('All original completed training fits required; never retire a fitting worker')
    manifest=u.read(original/'manifest.json');bindings=dict(manifest['code_sha256'])
    def bind(path,digest=None):
        key=str(path.relative_to(u.ROOT));value=u.sha(path) if digest is None else digest
        if key in bindings and bindings[key]!=value:raise ValueError('Original fitted source/model expectation conflicts')
        if u.sha(path)!=value:raise ValueError('Actual original fitted source/model changed')
        bindings[key]=value
    for key,value in list(bindings.items()):bind(u.ROOT/key,value)
    for path in [original/'manifest.json',original/'fit-completion.json',oldqueue/'plan.json',oldqueue/'failure.json',oldqueue/'watchdog-recovery.json']:
        bind(path)
    models=0
    for fold in range(5):
        path=original/'nn'/('%s.json'%fold);receipt=u.read(path)
        if (receipt['manifest_sha256']!=u.sha(original/'manifest.json') or receipt['fold']!=fold or
            len(receipt['chosen_iterations'])!=3 or any(type(v) is not int or not 0<v<=40 for v in receipt['chosen_iterations'])):
            raise ValueError('All original three-seed selected budgets and fold receipts required')
        expected_files={'ridge.npz','scalers.npz','prediction.f32'}|{'nn-%s.pt'%seed for seed in manifest['config']['seeds']}
        if set(receipt['outputs'])!=expected_files:raise ValueError('All actual original fitted pipelines required')
        bind(path)
        for name,digest in receipt['outputs'].items():
            bind(u.ROOT/'artifacts/upstream-own352-temporal-mps/nn'/str(fold)/name,digest)
            if name.endswith('.pt'):models+=1
    if models!=15:raise ValueError('All15 actual trained models required before read-only scratch retirement')
    return actual,bindings


def retire_readonly(expected):
    actual=identities.process(expected['pid'])
    if (not identities.same_owner(expected,actual) or 'T' not in actual['state'] or identities.children(expected['pid'])):
        raise ValueError('Original read-only verifier is no longer safely suspended/childless')
    os.killpg(expected['pid'],signal.SIGTERM);os.killpg(expected['pid'],signal.SIGCONT)
    for _ in range(100):
        if not identities.same_owner(expected,identities.process(expected['pid'])):
            u.write(REPORT/'readonly-retirement.json',dict(state='only_suspended_independent_verifier_retired_after_all15_fits_rehashed',
                original=expected,no_model_fitting_restarted=True,all_original_fitted_files_preserved=True,observed=stamp()))
            return
        time.sleep(.1)
    raise RuntimeError('Original read-only verifier did not retire; do not launch another')


def main():
    REPORT.mkdir(exist_ok=False)
    child = None
    captured = None
    owner = identities.process(os.getpid())
    u.write(REPORT / 'monitor.json', owner)
    inputs = u.ROOT / 'experiments/upstream-own352-temporal-data'
    retired,fit_bindings=completed_fits()
    sample=Path('/tmp/netflix-own352-paused-sample.txt')
    sample_copy=REPORT/'paused-verifier-sample.txt'
    if sample.exists():sample_copy.write_bytes(sample.read_bytes())
    paths = [Path(__file__), WORKER, VERIFIER, u.ROOT/'scripts/upstream_exact_streamed_scaler.py',
             u.ROOT/'OWN352_BOUNDED_REPLAY_PLAN.md', inputs / 'manifest.json',
             inputs / 'independent-verification.json', u.ROOT / 'scripts/upstream_own352_temporal_data.py',
             u.ROOT / 'scripts/verify_upstream_own352_temporal_data.py', u.ROOT / 'OWN352_TEMPORAL_PLAN.md']
    if sample_copy.exists():paths.append(sample_copy)
    frozen=dict(fit_bindings)
    for path in paths:
        key,value=str(path.relative_to(u.ROOT)),u.sha(path)
        if key in frozen and frozen[key]!=value:raise ValueError('Original fitted source expectation changed')
        frozen[key]=value
    u.write(REPORT / 'plan.json', dict(source_sha256=frozen, created=stamp(),
        original_readonly_verifier=retired,all15_original_fits_rehashed=True,
        original_model_fitting_restarted=False,original_guard_compressor_limit_bytes=5*2**30,
        observed_memory_before_retirement=memory.memory(),
        rule='All original579 inputs/eligible rows and428ridge statistics/scalers retained; full15 completed models unchanged. Only retire original already-suspended read-only verifier after complete fitted-file rehash. Full-column percentiles and original ordered float64 mean/std reproduced with bounded row scratch; require exact actual scaler equality and every saved-model prediction bit. Reserve16GiB before nonblocking GPU lock; 16GiB worker,24GiB total physical/RSS,128MiB swap growth,18.75GiB disk floor. Independent full preprocessing/model replay before release. Preserve actual child RAM on guard or monitor loss; no automatic retries.'))
    def check():
        for key, digest in frozen.items():
            if u.sha(u.ROOT / key) != digest:
                raise ValueError('Frozen production controller/input/code changed: '+key)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Neural study monitor signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    lane = (u.ROOT / 'experiments/upstream-source-extension/run.lock').open('a')
    try:
        retire_readonly(retired)
        baseline = memory.memory(); good = 0
        while True:
            check(); workers = resources.footprints(); current = memory.memory()
            admitted = safe(workers, current, baseline, 16*2**30)
            good = good+1 if admitted else 0
            u.write(REPORT / 'status.json', dict(state='waiting_full16GiB_reservation_and_GPU_lane',
                workers=workers, memory=current, good_samples=good, observed=stamp()))
            if good >= 2:
                try:
                    fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if safe(resources.footprints(), memory.memory(), baseline, 16*2**30):
                        break
                    fcntl.flock(lane, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
            time.sleep(10)
        with (REPORT / 'watchdog.log').open('x') as log:
            watch = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--watch', str(os.getpid())],
                cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for phase, script, args in [('independent_bounded_original_scalers', WORKER, []),
                                    ('separate_canonical_arithmetic_score', VERIFIER, [])]:
            check()
            with (REPORT / (phase+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(script), *args], cwd=u.ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                captured = identities.process(child.pid)
                if not captured or captured['pgid'] != child.pid or str(script) not in captured['command']:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGSTOP)
                    raise ValueError('Dedicated exact neural study child identity required')
                u.write(REPORT / 'child-ownership.json', captured)
                while child.poll() is None:
                    check(); workers = resources.footprints(child.pid); current = memory.memory()
                    own = workers.get(str(child.pid), dict(bytes=0, rss_bytes=0))
                    if (not safe(workers, current, baseline) or
                            max(own['bytes'], own['rss_bytes']) > 16*2**30 or watch.poll() is not None):
                        raise RuntimeError('Neural study resource/watchdog guard; preserve exact worker RAM')
                    u.write(REPORT / 'status.json', dict(state=phase, pid=child.pid,
                        workers=workers, memory=current, observed=stamp()))
                    time.sleep(3)
                if child.returncode:
                    raise RuntimeError('Original complete neural study phase failed; no retry')
                child = None; captured = None
        proof = u.ROOT / 'experiments/upstream-own352-bounded-replay/independent-canonical-verification.json'
        if u.read(proof)['state'] != 'complete_separate_bounded_original352_tree_half_arithmetic_and_full_canonical_score':
            raise ValueError('Full independent production proof required')
        u.write(REPORT / 'completion.json', dict(state='complete_original352_bounded_replay_and_separate_canonical_score',
            verification_sha256=u.sha(proof), qualifying_labels_read=False, production_started=False, observed=stamp()))
        u.write(REPORT / 'owner-release.json', dict(state='all_owned_phases_terminal_after_independent_verification'))
    except BaseException as error:
        if child is not None and child.poll() is None:
            # Popen still owns this unreaped direct child, including the small
            # interval before identity capture. start_new_session gives it a
            # dedicated process group; do not leave startup RAM unguarded.
            if captured is None or identities.same_owner(captured, identities.process(child.pid)):
                os.killpg(child.pid, signal.SIGSTOP)
        u.write(REPORT / 'failure.json', dict(error=repr(error), original_child_RAM_preserved=True,
            child=captured, observed=stamp(), no_automatic_restart=True))
        raise
    finally:
        lane.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--watch', type=int); args = parser.parse_args()
    watchdog(args.watch) if args.watch else main()
