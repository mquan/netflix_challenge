#!/usr/bin/env python3
"""Guard complete own352579-input neural fitting and independent pipeline replay."""
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

REPORT = u.ROOT / 'experiments/upstream-own352-temporal-queue'
WORKER = u.ROOT / 'scripts/upstream_own352_temporal_mps.py'
VERIFIER = u.ROOT / 'scripts/verify_upstream_own352_temporal_mps.py'


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


def main():
    REPORT.mkdir(exist_ok=False)
    child = None
    captured = None
    owner = identities.process(os.getpid())
    u.write(REPORT / 'monitor.json', owner)
    inputs = u.ROOT / 'experiments/upstream-own352-temporal-data'
    paths = [Path(__file__), WORKER, VERIFIER, inputs / 'manifest.json',
             inputs / 'independent-verification.json', u.ROOT / 'scripts/upstream_own352_temporal_data.py',
             u.ROOT / 'scripts/verify_upstream_own352_temporal_data.py', u.ROOT / 'OWN352_TEMPORAL_PLAN.md']
    frozen = {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
    u.write(REPORT / 'plan.json', dict(source_sha256=frozen, created=stamp(),
        rule='Full original579 inputs, H128/rank32, all5userfolds/3seeds, max40patience4/batch8192/AdamW and original inner/full eligible statistics. Reserve16GiB before nonblocking GPU lock; 16GiB worker,24GiB total physical/RSS,128MiB swap growth,18.75GiB disk floor. Independent full preprocessing/model replay before release. Preserve actual child RAM on guard or monitor loss; no automatic retries.'))
    def check():
        for key, digest in frozen.items():
            if u.sha(u.ROOT / key) != digest:
                raise ValueError('Frozen production controller/input/code changed: '+key)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Neural study monitor signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    lane = (u.ROOT / 'experiments/upstream-source-extension/run.lock').open('a')
    try:
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
        for phase, script, args in [('prepare_complete_original_study', WORKER, []),
                                    ('all_five_original_three_seed_fits', WORKER, ['--worker']),
                                    ('independent_all_original_pipelines', VERIFIER, [])]:
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
        proof = u.ROOT / 'experiments/upstream-own352-temporal-mps/independent-verification.json'
        if u.read(proof)['state'] != 'complete_actual_own352_all_fits_original_statistics_scalers_and_saved_model_replays':
            raise ValueError('Full independent production proof required')
        u.write(REPORT / 'completion.json', dict(state='complete_all_original352_fits_and_full_actual_saved_pipeline_replays',
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
