#!/usr/bin/env python3
"""Prepare verified fixed raw additions, fit/replay NN, then evaluate fixed mean."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import datetime as dt
import fcntl
from pathlib import Path
import subprocess
import sys
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-autorec109-queue'
SCRIPTS=['upstream_autorec109.py','upstream_autorec109_mps_context.py','upstream_rbm50_tree_autorec109_mps.py']

def execute(name):
    with (REPORT/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/name)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing declaration preserved')
        hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),*[u.ROOT/'scripts'/name for name in SCRIPTS]]}
        u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),code_sha256=hashes,
            additions=['mps-autorec_ep05','mps-autorec_ep10'],raw_prediction_columns=109,learned_support_columns=90,
            neural_inputs=336,hidden=128,context_rank=32,weights=[.5,.5],
            mean_origins=['upstream-rbm50-tree','upstream-autorec109-mps-context'],
            rule='Keep original five nested user folds/seeds/optimizer/stopping. All15 actual NN models must replay. Compare separately declared fixed half mean to verified current best .8592720612630157; no weight search or qualifying-based selection.',
            raw_full50_completion_claimed=False,qualifying_labels_read=False,automatic_production=False))
        try:
            u.write(REPORT/'status.json',dict(state='waiting_for_epoch_boundary_to_prepare_cache'))
            with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
                fcntl.flock(shared,fcntl.LOCK_EX)
                execute(SCRIPTS[0])
            u.write(REPORT/'status.json',dict(state='neural_controller_running'))
            execute(SCRIPTS[1])
            for name,digest in hashes.items():
                if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared code changed')
            execute(SCRIPTS[2])
            result=u.read(u.ROOT/'experiments/upstream-rbm50-tree-autorec109-mps/verification.json')
            u.write(REPORT/'status.json',dict(state='complete_actual_replayed_development_and_fixed_mean',rmse=result['rmse'],improves_previous=result['improves_previous'],goal_complete=False))
        except BaseException as error:
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise

if __name__=='__main__':run()
