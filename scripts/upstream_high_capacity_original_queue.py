#!/usr/bin/env python3
"""Queue the original1600/2 preset after current studies, with native RAM guards."""
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

REPORT = u.ROOT / 'experiments/upstream-high-capacity-original'
NAME = 'tsvdx4-1600lm'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ready():
    complete = u.ROOT / 'experiments/upstream-own176-recent10-queue/completion.json'
    return complete.exists() and u.read(complete).get('state') == 'complete_all_own176_original_fits_and_independent_full_pipeline_replays'


def safe(workers, current, baseline, free, reserve=0):
    return (sum(v['bytes'] for v in workers.values()) + reserve <= 24 * 2**30 and
            sum(v['rss_bytes'] for v in workers.values()) + reserve <= 24 * 2**30 and
            current['swap_used_bytes'] - baseline['swap_used_bytes'] <= 128 * 2**20 and
            free >= 18.75 * 2**30)


def run():
    child = None
    native = None
    with (REPORT / 'queue.lock').open('a') as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (REPORT / 'queue-plan.json').exists():
            raise ValueError('Existing high-capacity owner/evidence must be preserved')
        files = [Path(__file__), u.ROOT / 'scripts/upstream_high_capacity_watchdog.py',
                 u.ROOT / 'scripts/verify_upstream_high_capacity_original.py',
                 u.ROOT / 'scripts/build_upstream_high_capacity_original.py',
                 Path(resources.__file__), REPORT / 'execution-plan.json']
        frozen = {str(p.relative_to(u.ROOT)): u.sha(p) for p in files}
        def check():
            for rel, digest in frozen.items():
                if u.sha(u.ROOT / rel) != digest:
                    raise ValueError('Frozen original execution binding changed: ' + rel)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Original high-capacity owner signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        u.write(REPORT / 'queue-plan.json', dict(created=stamp(), source_sha256=frozen,
                rule='Exact original1600-factor2/2 preset. Wait complete own176 and prior matched/temporal gates before shared GPU/native lane acquisition. Reserve22GiB,24GiB total RSS/physical footprint,22GiB native limit,128MiB swap-growth,18.75GiB disk floor. Original4 Rayon/1 BLAS threads. Independent original source/build/input admission before fitting; independent complete output proof afterward. Guard/interruption suspends the original native process group in RAM; no automatic restarts or recipe changes.'))
        try:
            while not ready():
                check()
                u.write(REPORT / 'queue-status.json', dict(state='waiting_for_complete_own176_and_preceding_original_studies', observed=stamp()))
                time.sleep(15)
            # Never hold this lane while waiting on an earlier GPU study.
            lane = (u.ROOT / 'experiments/upstream-source-extension/run.lock').open('a')
            fcntl.flock(lane, fcntl.LOCK_EX)
            primary_lane = (u.ROOT / 'experiments/full-upstream-reproduction/runner.lock').open('a')
            u.write(REPORT / 'queue-status.json', dict(state='waiting_for_original_primary_DAG_lane', observed=stamp()))
            fcntl.flock(primary_lane, fcntl.LOCK_EX)
            previous = memory.memory();good = 0
            while good < 2:
                check();workers = resources.footprints();current = memory.memory()
                admitted = (safe(workers, current, previous, shutil.disk_usage(u.ROOT).free, 22 * 2**30)
                            and current['compressor_bytes'] < 5 * 2**30)
                good = good + 1 if admitted else 0;previous = current
                u.write(REPORT / 'queue-status.json', dict(state='waiting_for_full_original1600_reservation',
                        workers=workers, memory=current, good_samples=good, observed=stamp()))
                if good < 2:time.sleep(15)
            # The old shared native binary did not match its initial build
            # receipt. Compile isolated original source after admission instead
            # of treating the mere presence of a binary as provenance.
            baseline = current
            with (REPORT / 'isolated-build.log').open('x') as log:
                child = subprocess.Popen([sys.executable, str(u.ROOT / 'scripts/build_upstream_high_capacity_original.py')],
                        cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    check();workers = resources.footprints(child.pid);current = memory.memory()
                    if not safe(workers, current, baseline, shutil.disk_usage(u.ROOT).free):
                        raise RuntimeError('Original isolated build resource guard; preserve compiler group')
                    u.write(REPORT / 'queue-status.json', dict(state='building_exact_original1600_in_isolated_target',
                            pid=child.pid, workers=workers, memory=current, observed=stamp()))
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError('Isolated source build failed; no model launch or automatic retry')
                child = None
            # Full immutable data/source hashes and compiled build proof again,
            # after all prerequisite studies, immediately before native launch.
            with (REPORT / 'launch-admission.log').open('x') as log:
                subprocess.run([sys.executable, str(u.ROOT / 'scripts/verify_upstream_high_capacity_original.py')],
                               cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            check();workers = resources.footprints();baseline = memory.memory()
            if not safe(workers, baseline, baseline, shutil.disk_usage(u.ROOT).free, 22 * 2**30):
                raise RuntimeError('Actual admission changed before launch')
            job = u.read(REPORT / 'independent-admission.json')
            frozen[str((REPORT / 'independent-admission.json').relative_to(u.ROOT))] = u.sha(REPORT / 'independent-admission.json')
            frozen[str((REPORT / 'original-build-receipt.json').relative_to(u.ROOT))] = u.sha(REPORT / 'original-build-receipt.json')
            env = u.environment()
            env.update(RAYON_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            started = stamp()
            with (REPORT / (NAME + '.log')).open('x') as log:
                child = subprocess.Popen(job['native_command'], cwd=job['cwd'], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                native = identities.process(child.pid)
                if native is None or native['pgid'] != child.pid or str(job['native_command'][0]) not in native['command']:
                    # Preserve the newly launched dedicated group even if it
                    # exited before identity capture; never signal reused PIDs.
                    if child.poll() is None:os.killpg(child.pid, signal.SIGSTOP)
                    raise RuntimeError('Original dedicated native identity unavailable')
                u.write(REPORT / 'native-ownership.json', dict(monitor=identities.process(os.getpid()), native=native,
                        original_command=job['native_command'], environment={k:env[k] for k in ['RAYON_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}))
                with (REPORT / 'watchdog.log').open('x') as watchlog:
                    watch = subprocess.Popen([sys.executable, str(u.ROOT / 'scripts/upstream_high_capacity_watchdog.py')],
                            cwd=u.ROOT, stdout=watchlog, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    check();workers = resources.footprints(child.pid);current = memory.memory()
                    own = workers.get(str(child.pid), dict(bytes=0, rss_bytes=0))
                    if not safe(workers, current, baseline, shutil.disk_usage(u.ROOT).free) or max(own['bytes'], own['rss_bytes']) > 22 * 2**30:
                        raise RuntimeError('Original full-model resource guard; native RAM must remain preserved')
                    if watch.poll() is not None and child.poll() is None:
                        raise RuntimeError('Independent native monitor-loss watchdog exited unexpectedly')
                    u.write(REPORT / 'queue-status.json', dict(state='original1600_full_two_phase_native_running',
                            pid=child.pid, workers=workers, memory=current, observed=stamp()))
                    time.sleep(2)
            result = dict(command=job['native_command'], cwd=job['cwd'], returncode=child.returncode,
                          started=started, finished=stamp(), epochs_per_phase=2, factors=1600,
                          qualifying_labels_read=False)
            u.write(REPORT / (NAME + '-run.json'), result)
            if child.returncode:raise RuntimeError('Original native failed; preserve outputs without automatic restart')
            child = None
            with (REPORT / 'full-output-verification.log').open('x') as log:
                subprocess.run([sys.executable, str(u.ROOT / 'scripts/verify_upstream_high_capacity_original.py'), '--complete'],
                               cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            proof = REPORT / (NAME + '-verification.json')
            if u.read(proof)['state'] != 'complete_original1600_both_full_phases_and_all_native_query_vectors':
                raise ValueError('Independent full original model proof required')
            u.write(REPORT / 'completion.json', dict(state='complete_original1600_and_independent_full_output_verification',
                    verification_sha256=u.sha(proof), observed=stamp(), no_score_promotion=True))
        except BaseException as error:
            if child is not None and child.poll() is None:
                if native is None or identities.same_owner(native, identities.process(child.pid)):
                    os.killpg(child.pid, signal.SIGSTOP)
            u.write(REPORT / 'failure.json', dict(error=repr(error), observed=stamp(),
                    native_RAM_preserved_if_still_live=True, no_restart_attempted=True))
            raise


if __name__ == '__main__':
    run()
