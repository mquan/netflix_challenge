#!/usr/bin/env python3
"""Admission checks and replay of completed fits; no refitting after guard race."""
import os
import datetime as dt
import fcntl
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_cf96_memory_resume as memory
import upstream_research_footprints as resource

REPORT=u.ROOT/'experiments/upstream-own167-context218-replay-recovery'
ORIGIN=u.ROOT/'experiments/upstream-own167-context218-mps'
RUN=u.ROOT/'artifacts/upstream-own167-context218-mps'
VERIFIER=u.ROOT/'scripts/verify_upstream_own167_context218_mps.py'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def checked_fit():
    mp=ORIGIN/'manifest.json';m=u.read(mp);digest=u.sha(mp)
    completion=u.read(ORIGIN/'fit-completion.json')
    if completion['state']!='complete_all_five_original_budget_nested_folds':raise ValueError('Completed original fitting required')
    qp=u.read(u.ROOT/'experiments/upstream-own167-context218-queue/plan.json')
    for key,h in qp['source_sha256'].items():
        if u.sha(u.ROOT/key)!=h:raise ValueError('Original numerical/guard code changed')
    global_path=u.ROOT/'experiments/upstream-native-nn-pool6-tree31/independent-verification.json'
    global_key=str(global_path.relative_to(u.ROOT))
    if u.sha(global_path)!=m['source_sha256'][global_key] or m['preceding_global_development_rmse']!=u.read(global_path)['rmse']:
        raise ValueError('Comparison reference must derive from bound original source')
    if m['fixed_tree_weight']!=.5 or m['config']['seeds']!=[2026,7,13] or m['config']['nn_epochs']!=40:raise ValueError('Original declared comparison/seeds/budget differ')
    bindings={str(mp.relative_to(u.ROOT)):digest};counts=[]
    for fold in range(5):
        p=ORIGIN/'nn'/f'{fold}.json';r=u.read(p);folder=RUN/'nn'/str(fold)
        if r['manifest_sha256']!=digest or r['fold']!=fold or r['family']!='nn':raise ValueError('All original fold manifests must match')
        selected=r['chosen_iterations']
        if not isinstance(selected,list) or len(selected)!=3 or any(type(n) is not int or not 0<=n<=40 for n in selected):raise ValueError('Exactly three original valid seed budgets required')
        expected={f'nn-{seed}.pt' for seed,n in zip([2026,7,13],selected) if n}
        if {p.name for p in folder.glob('nn-*.pt')}!=expected:raise ValueError('Actual checkpoint inventory differs from all three budgets')
        if set(r['outputs'])!=expected|{'ridge.npz','scalers.npz','prediction.f32'}:raise ValueError('Complete actual fold artifact inventory differs')
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
        for name,h in r['outputs'].items():
            path=folder/name
            if u.sha(path)!=h:raise ValueError('Actual completed artifact changed')
            bindings[str(path.relative_to(u.ROOT))]=h
        counts.append(selected)
    return digest,bindings,counts

def main():
    REPORT.mkdir(exist_ok=False)
    digest,bindings,counts=checked_fit()
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(resource.__file__),VERIFIER]}
    u.write(REPORT/'plan.json',dict(created=stamp(),original_manifest_sha256=digest,source_sha256=frozen,
        completed_fit_bindings=bindings,actual_three_seed_budgets=counts,
        rule='All five fits complete before original guard process-exit scan failure. Preserve failure and original fitting. Revalidate source-derived comparison and exact three-seed inventory; run unchanged independent verifier only. Corrected scanner includes relative/lowRSS peers and ignores exited/zombie handles. No numerical training change.'))
    child=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Replay recovery signal')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        workers=resource.footprints();baseline=memory.memory()
        if sum(v['bytes'] for v in workers.values())+8*2**30>24*2**30:raise RuntimeError('Insufficient complete verifier headroom; no fitting restarted')
        with (REPORT/'replay.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(VERIFIER)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            high=0
            while child.poll() is None:
                for key,h in frozen.items():
                    if u.sha(u.ROOT/key)!=h:raise ValueError('Frozen recovery/verifier changed')
                workers=resource.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                high=high+1 if (sum(v['bytes'] for v in workers.values())>24*2**30 or sum(v['rss_bytes'] for v in workers.values())>24*2**30 or
                    own['bytes']>14*2**30 or own['rss_bytes']>14*2**30 or current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20) else 0
                u.write(REPORT/'status.json',dict(state='unchanged_actual_saved_models_and_original_statistics_replay',pid=child.pid,workers=workers,memory=current,observed=stamp()))
                if high>=3 or shutil.disk_usage(u.ROOT).free<16*2**30:raise RuntimeError('Replay guard exceeded; fits preserved')
                time.sleep(5)
            if child.returncode:raise RuntimeError('Independent replay failed; all original fits preserved')
        after,again,_=checked_fit()
        if after!=digest or again!=bindings:raise ValueError('Original complete fits changed during replay')
        result=u.read(ORIGIN/'independent-verification.json')
        actual_nn=next(r['rmse'] for r in result['records'] if r['arm']=='nn')
        actual_half=next(r['rmse'] for r in result['records'] if r['arm']=='fixed-half-tree-nn')
        if (result['fixed_half_improves_global']!=(actual_half<result['preceding_global_development_rmse']) or
            result['nn_improves_nn154']!=(actual_nn<result['control_scores']['nn154']) or
            result['nn_improves_nn166']!=(actual_nn<result['control_scores']['nn166'])):raise ValueError('Derived known-only promotion flags differ')
        u.write(REPORT/'verification.json',dict(state='complete_original_full_fits_and_strengthened_replay_admission',observed=stamp(),
            original_manifest_sha256=digest,independent_verification_sha256=u.sha(ORIGIN/'independent-verification.json'),
            all_five_folds_exact_three_seed_budget_and_checkpoint_inventory_verified=True,source_derived_global_reference_verified=True,
            records=result['records'],control_scores=result['control_scores'],qualifying_labels_read=False,model_fitting_repeated=False))
        print(result['records'],flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait()
        u.write(REPORT/'failure.json',dict(error=repr(error),observed=stamp(),fitting_outputs_preserved=True));raise

if __name__=='__main__':main()
