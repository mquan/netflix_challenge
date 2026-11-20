#!/usr/bin/env python3
"""Keep the GPU slot available to raw epochs while waiting for full NN headroom."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_knn3pair2_virtual_queue as original

u = original.u
DECLARED = u.ROOT / 'experiments/upstream-knn3pair2-virtual-queue'
REPORT = u.ROOT / 'experiments/upstream-knn3pair2-opportunistic-queue'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def admitted(previous, current, other_rss, disk, busy):
    return (not busy and other_rss < 8 * 2**30 and disk > 12 * 2**30
            and original.model.context_admitted(previous, current, other_rss))


def try_slot(slot, ready):
    try:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    try:
        if ready():
            return True
    except BaseException:
        fcntl.flock(slot, fcntl.LOCK_UN)
        raise
    fcntl.flock(slot, fcntl.LOCK_UN)
    return False


def verify_original_files(plan):
    for name, digest in plan['code_sha256'].items():
        if u.sha(u.ROOT / name) != digest:
            raise ValueError('Original declared numerical/source implementation changed: ' + name)


def checked_prepare(script, logname):
    # Identical complete CPU preparation, under the existing 24 GiB /8 GiB
    # resource guard. No GPU slot or source-worker interruption is needed.
    with (REPORT / logname).open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(u.ROOT / 'scripts' / script)],
                                 cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        high = 0
        try:
            while child.poll() is None:
                own = original.model.base.preceding.group_rss(child.pid)
                other = original.model.base.preceding.resources.other_rss({child.pid})
                free = shutil.disk_usage(u.ROOT).free
                high = high + 1 if own + other > 24 * 2**30 else 0
                u.write(REPORT / 'preparation-status.json', dict(pid=child.pid,
                        script=script, own_rss_bytes=own, other_rss_bytes=other,
                        disk_free_bytes=free, observed=stamp(), gpu_slot_owned=False))
                if high >= 3 or free < 8 * 2**30:
                    raise RuntimeError('Original full preparation resource guard; outputs preserved')
                time.sleep(5)
        except BaseException:
            original.model.base.preceding.stop(child)
            raise
        if child.returncode:
            raise RuntimeError('Full original preparation failed; partial evidence preserved')


def current_state(previous):
    current = original.model.base.memory.memory()
    other = original.model.base.preceding.resources.other_rss({os.getpgrp()})
    free = shutil.disk_usage(u.ROOT).free
    busy = original.predecessors()
    return current, other, free, busy, admitted(previous, current, other, free, busy)


def run():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (REPORT / 'plan.json').exists():
            raise ValueError('Existing replacement coordinator evidence preserved')
        plan = u.read(DECLARED / 'plan.json')
        verify_original_files(plan)
        recovery = u.read(REPORT / 'recovery.json')
        if recovery['original_declaration_sha256'] != u.sha(DECLARED / 'plan.json'):
            raise ValueError('Original pre-fit declaration differs')
        u.write(REPORT / 'plan.json', dict(created=stamp(), script_sha256=u.sha(Path(__file__)),
                original_declaration_sha256=u.sha(DECLARED / 'plan.json'),
                recovery_sha256=u.sha(REPORT / 'recovery.json'),
                original_code_sha256=plan['code_sha256'],
                scheduling='CPU preparation without GPU lock. Stable original full-neural headroom before attempting GPU lock; immediate recheck after nonblocking acquisition. Release failed acquisition/recheck. Original 16GiB reserve and 24GiB RSS/8GiB disk guard retained.',
                numerical_recipe_or_data_or_budget_changed=False, qualifying_labels_read=False))
        original.REPORT = REPORT  # Logging only, within this new interpreter.
        proofs = [u.ROOT / f'experiments/upstream-split3-knn3-chains/{name}-verification.json'
                  for name in original.model.data.NEW]
        while not all(p.exists() for p in proofs) or original.predecessors():
            u.write(REPORT / 'status.json', dict(state='waiting_for_original_complete_source_proofs_and_predecessors',
                    observed=stamp(), gpu_slot_owned=False))
            time.sleep(20)
        # Preparation reserves 8GiB and does not own the GPU slot; the existing
        # raw epoch process may continue. Every byte is still the full data.
        previous = original.model.base.memory.memory()
        good = 0
        while good < 2:
            current = original.model.base.memory.memory()
            other = original.model.base.preceding.resources.other_rss({os.getpgrp()})
            free = shutil.disk_usage(u.ROOT).free
            good = good + 1 if (other < 16 * 2**30 and free > 12 * 2**30
                               and current['compressor_bytes'] < 5 * 2**30
                               and current['swap_used_bytes']-previous['swap_used_bytes'] < 32 * 2**20) else 0
            u.write(REPORT / 'status.json', dict(state='waiting_for_full_cpu_preparation_headroom',
                    observed=stamp(), other_rss_bytes=other, memory=current,
                    good_samples=good, gpu_slot_owned=False))
            if good < 2:
                previous = current
                time.sleep(20)
        verify_original_files(plan)
        checked_prepare('upstream_knn3pair2_screen.py', 'ridge.log')
        checked_prepare('verify_upstream_knn3pair2_screen.py', 'independent-ridge.log')
        checked_prepare('upstream_knn3pair2_virtual_features.py', 'features.log')
        original.model.configure()
        original.model.prepare()
        u.write(REPORT / 'preparation-complete.json', dict(state='complete_original_full_cpu_preparation',
                observed=stamp(), virtual_manifest_sha256=u.sha(original.model.data.REPORT / 'manifest.json'),
                full_model_manifest_sha256=u.sha(original.model.REPORT / 'manifest.json'),
                numerical_recipe_or_data_or_budget_changed=False, gpu_slot_owned=False))
        previous = original.model.base.memory.memory()
        good = 0
        with (u.ROOT / 'experiments/upstream-source-extension/run.lock').open('a') as slot:
            while True:
                current, other, free, busy, ok = current_state(previous)
                good = good + 1 if ok else 0
                u.write(REPORT / 'status.json', dict(state='waiting_for_original_full_neural_headroom_without_gpu_lock',
                        observed=stamp(), other_rss_bytes=other, memory=current,
                        good_samples=good, gpu_slot_owned=False))
                previous = current
                if good >= 2:
                    def recheck():
                        return current_state(previous)[-1]
                    if try_slot(slot, recheck):
                        break
                    good = 0
                time.sleep(20)
            try:
                verify_original_files(plan)
                u.write(REPORT / 'admitted.json', dict(state='original_full_neural_headroom_and_gpu_slot_admitted',
                        observed=stamp(), other_rss_bytes=other, memory=current,
                        full_original_peak_reserve_bytes=16 * 2**30))
                original.monitored(['--worker'], 'nn.log')
                original.monitored(['--replay'], 'saved-nn-replay.log')
                original.model.configure()
                original.model.base.preceding.summarize()
            finally:
                fcntl.flock(slot, fcntl.LOCK_UN)
        results = []
        for tree, script, name in original.MEANS:
            checked_prepare(script, name + '.log')
            with (REPORT / (name + '-independent.log')).open('x') as log:
                subprocess.run([sys.executable, '-u', str(u.ROOT / 'scripts/verify_upstream_development_mean.py'), name],
                               cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            p = u.ROOT / 'experiments' / name / 'independent-mean-check.json'
            check = u.read(p)
            results.append(dict(tree=tree, rmse=check['rmse'], independent_mean_sha256=u.sha(p)))
        verify_original_files(plan)
        current = u.read(u.ROOT / 'experiments/fresh-reproduction-status.json')['best_development_rmse']
        u.write(REPORT / 'completion.json', dict(state='complete_original_knn3_pair_neural_study_and_both_declared_tree_means',
                observed=stamp(), results=results, reference_at_declaration=plan['reference_development_rmse'],
                current_verified_reference_at_summary=current,
                improves_current_reference=min(r['rmse'] for r in results) < current,
                numerical_recipe_or_data_or_budget_changed=False, qualifying_labels_read=False,
                automatic_production_started=False, goal_complete=False))
        u.write(REPORT / 'status.json', dict(state='complete_actual_replayed_knn3_pair_study', results=results))


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        u.write(REPORT / 'failure.json', dict(state='incomplete_preserved', error=repr(error), observed=stamp()))
        raise
