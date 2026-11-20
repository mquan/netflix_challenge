#!/usr/bin/env python3
"""Run the unchanged full KNN3 study using proven exact memory operations."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_knn3pair2_compact_mps_context as model

u = model.u
REPORT = u.ROOT / 'experiments/upstream-knn3pair2-compact-queue'
MEANS = [('upstream-rbm50-tree', 'upstream_rbm50_tree_knn3pair2_compact_mps.py', 'upstream-rbm50-tree-knn3pair2-compact-mps'),
         ('upstream-item154-tree', 'upstream_item154_tree_knn3pair2_compact_mps.py', 'upstream-item154-tree-knn3pair2-compact-mps')]


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def admitted(previous, current, other_rss, disk_free):
    return (other_rss < 14 * 2**30 and disk_free > 12 * 2**30
            and current['compressor_bytes'] < 5 * 2**30
            and current['swap_used_bytes']-previous['swap_used_bytes'] < 32 * 2**20)


def observe(previous):
    current = model.recipe.base.memory.memory()
    other = model.recipe.base.preceding.resources.other_rss({os.getpgrp()})
    disk = shutil.disk_usage(u.ROOT).free
    return current, other, disk, admitted(previous, current, other, disk)


def monitored(args, logname):
    with (REPORT / logname).open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(model.__file__)), *args],
                cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        high = 0
        try:
            while child.poll() is None:
                own = model.recipe.base.preceding.group_rss(child.pid)
                other = model.recipe.base.preceding.resources.other_rss({child.pid})
                free = shutil.disk_usage(u.ROOT).free
                high = high + 1 if own > 10 * 2**30 or own + other > 24 * 2**30 else 0
                u.write(REPORT / 'worker-status.json', dict(observed=stamp(), pid=child.pid,
                        own_rss_bytes=own, other_rss_bytes=other, disk_free_bytes=free,
                        own_limit_bytes=10 * 2**30, total_limit_bytes=24 * 2**30))
                if high >= 3 or free < 8 * 2**30:
                    raise RuntimeError('Compact execution resource guard; full partial artifacts preserved')
                time.sleep(5)
        except BaseException:
            model.recipe.base.preceding.stop(child)
            raise
        if child.returncode:
            raise RuntimeError('Complete compact worker failed; no smaller retry or automatic restart')


def run():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'run.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (REPORT / 'plan.json').exists():
            raise ValueError('Existing compact study declaration preserved')
        source = u.ROOT / 'experiments/upstream-knn3pair2-virtual-queue/plan.json'
        original = u.read(source)
        for name, digest in original['code_sha256'].items():
            if u.sha(u.ROOT / name) != digest:
                raise ValueError('First declared numerical/source implementation changed')
        model.verified_cache();model.prepare()
        files = [Path(__file__), Path(model.__file__), Path(model.scaling.__file__),
                 *[u.ROOT / 'scripts' / row[1] for row in MEANS],
                 u.ROOT / 'scripts/verify_upstream_development_mean.py']
        hashes = {str(p.relative_to(u.ROOT)): u.sha(p) for p in files}
        u.write(REPORT / 'plan.json', dict(created=stamp(), code_sha256=hashes,
                first_declaration_sha256=u.sha(source), recovery_sha256=u.sha(REPORT / 'recovery.json'),
                full_model_manifest_sha256=u.sha(model.REPORT / 'manifest.json'),
                exact_actual_input_scaler_proof_sha256=u.sha(model.parity.REPORT / 'verification.json'),
                fixed_addition=original['fixed_addition'], inputs=400, hidden=128, context_rank=32,
                weights=[.5,.5], predeclared_tree_partners=[m[0] for m in MEANS],
                reference_development_rmse=original['reference_development_rmse'],
                policy='Same full neural recipe/data/folds/seeds/optimizer/stopping. Exact cached matrices, float32 input assembly, unchanged eligible statistics and exact block normalization. TenGiB own-RSS guard under unchanged total24GiB/disk8GiB guard; two stable samples below14GiB other RSS while exclusively reserving the GPU slot, release failed reservations immediately. No model/data/budget reduction, coefficient search or qualifying selection.',
                models_fitted_before_declaration=False, numerical_recipe_or_budget_changed=False,
                qualifying_labels_read=False, automatic_production=False))
        with (u.ROOT / 'experiments/upstream-source-extension/run.lock').open('a') as slot:
            while True:
                u.write(REPORT / 'status.json', dict(state='waiting_for_complete_current_gpu_epoch_boundary',
                        observed=stamp(), gpu_slot_owned=False))
                fcntl.flock(slot, fcntl.LOCK_EX)
                previous = model.recipe.base.memory.memory();good = 0
                for _ in range(2):
                    current, other, free, ready = observe(previous)
                    good = good + 1 if ready else 0
                    u.write(REPORT / 'status.json', dict(state='checking_complete_compact_neural_headroom',
                            observed=stamp(), other_rss_bytes=other, memory=current,
                            good_samples=good, gpu_slot_owned=True))
                    if not ready:
                        break
                    previous = current
                    if good < 2:
                        time.sleep(20)
                if good >= 2:
                    break
                fcntl.flock(slot, fcntl.LOCK_UN)
                time.sleep(20)
            try:
                for name, digest in hashes.items():
                    if u.sha(u.ROOT / name) != digest:
                        raise ValueError('Frozen exact execution changed')
                u.write(REPORT / 'admitted.json', dict(observed=stamp(), other_rss_bytes=other,
                        original_full_recipe_preserved=True, own_peak_budget_bytes=10 * 2**30,
                        total_budget_bytes=24 * 2**30, gpu_slot_owned=True))
                monitored(['--worker'], 'nn.log')
                monitored(['--replay'], 'saved-nn-replay.log')
                model.configure();model.recipe.base.preceding.summarize()
            finally:
                fcntl.flock(slot, fcntl.LOCK_UN)
        results = []
        for tree, script, name in MEANS:
            with (REPORT / (name + '.log')).open('x') as log:
                subprocess.run([sys.executable, '-u', str(u.ROOT / 'scripts' / script)], cwd=u.ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with (REPORT / (name + '-independent.log')).open('x') as log:
                subprocess.run([sys.executable, '-u', str(u.ROOT / 'scripts/verify_upstream_development_mean.py'), name],
                               cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            path = u.ROOT / 'experiments' / name / 'independent-mean-check.json';check = u.read(path)
            results.append(dict(tree=tree, rmse=check['rmse'], independent_mean_sha256=u.sha(path)))
        current = u.read(u.ROOT / 'experiments/fresh-reproduction-status.json')['best_development_rmse']
        u.write(REPORT / 'completion.json', dict(state='complete_exact_compact_knn3_study_and_both_fixed_means',
                observed=stamp(), results=results, current_verified_reference=current,
                improves_current_reference=min(r['rmse'] for r in results) < current,
                original_full_model_and_budget_preserved=True, qualifying_labels_read=False,
                automatic_production_started=False, goal_complete=False))
        u.write(REPORT / 'status.json', dict(state='complete_exact_compact_knn3_study', observed=stamp(), results=results))


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        u.write(REPORT / 'failure.json', dict(state='incomplete_preserved', observed=stamp(), error=repr(error)))
        raise
