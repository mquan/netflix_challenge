#!/usr/bin/env python3
"""Fixed original attention64 pair study with complete budgets and both tree partners."""
import datetime as dt
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import upstream_attention2_virtual_mps_context as model

u=model.u
REPORT=u.ROOT/'experiments/upstream-attention2-virtual-queue'
PREDECESSORS={'upstream_item154_tree.py','upstream_item154_tree_supervisor.py',
 'upstream_itemae_virtual_queue.py','upstream_itemae_virtual_mps_context.py',
 'upstream_itemae_virtual_features.py','upstream_itemae_pair2_screen.py','verify_upstream_itemae_pair2_screen.py'}
MEANS=[('upstream-rbm50-tree','upstream_rbm50_tree_attention2_virtual_mps.py','upstream-rbm50-tree-attention2-virtual-mps'),
       ('upstream-item154-tree','upstream_item154_tree_attention2_virtual_mps.py','upstream-item154-tree-attention2-virtual-mps')]

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def predecessors():
    found={}
    for row in subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines():
        parts=row.strip().split(None,2)
        if len(parts)!=3 or parts[1].startswith('Z'):continue
        if PREDECESSORS.intersection(Path(arg).name for arg in parts[2].split()):found[parts[0]]=parts[2]
    return found

def execute(script,logname):
    with (REPORT/logname).open('x') as log:
        subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def monitored(extra,logname):
    with (REPORT/logname).open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(model.__file__)),*extra],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=model.base.preceding.group_rss(child.pid);other=model.base.preceding.resources.other_rss({child.pid});free=shutil.disk_usage(u.ROOT).free
                high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,observed=stamp()))
                if high>=3 or free<8*2**30:raise RuntimeError('Original neural resource guard; partial outputs preserved')
                time.sleep(5)
        except BaseException:model.base.preceding.stop(child);raise
        if child.returncode:raise RuntimeError('Attention pair neural child failed; outputs preserved')

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing fixed attention pair declaration preserved')
        files=[Path(__file__),Path(model.__file__),Path(model.data.__file__),Path(model.data.screen.__file__),
               u.ROOT/'scripts/verify_upstream_attention2_screen.py',u.ROOT/'scripts/verify_upstream_development_mean.py',*[u.ROOT/'scripts'/m[1] for m in MEANS]]
        hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
        u.write(REPORT/'plan.json',dict(created=stamp(),code_sha256=hashes,fixed_addition=model.data.NEW,
            original_source_epochs_per_phase=12,raw_prediction_columns=109,learned_support_columns=154,
            inputs=400,hidden=128,context_rank=32,weights=[.5,.5],predeclared_tree_partners=[m[0] for m in MEANS],
            reference_development_rmse=.8592309819864167,qualifying_labels_read=False,
            rule='Both original attention64 terminal/epoch5 pairs only after verified12/12 native completion. Preserve own107/item154 and all original five nested user folds/seeds/optimizer/stopping. Require separate ridge replay, full virtual tensor hashes, all15 actual saved NN replays and separate fixed-half mean checks with each of the two declared actual tree families. No raw subset, weight search, qualifying outcome or automatic production.',
            resource_policy=dict(total_rss_limit_bytes=24*2**30,disk_floor_bytes=8*2**30,peak_neural_reserve_bytes=16*2**30,
                wait_for_actual_cpu_tree_and_prior_full_neural_study=True)))
        source_proof=u.ROOT/'experiments/upstream-split3-next3/s3-attn64-verification.json'
        while not source_proof.exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_original_complete12_12_attention64_and_independent_proof',observed=stamp()));time.sleep(20)
        while True:
            busy=predecessors()
            if not busy:break
            u.write(REPORT/'status.json',dict(state='waiting_for_actual_cpu_tree_and_preceding_fixed_neural_study',predecessors=busy,observed=stamp()));time.sleep(20)
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX);previous=model.base.memory.memory();good=0
            while good<2:
                current=model.base.memory.memory();other=model.base.preceding.resources.other_rss({__import__('os').getpgrp()})
                free=shutil.disk_usage(u.ROOT).free
                good=good+1 if (not predecessors() and other<8*2**30 and model.context_admitted(previous,current,other) and free>12*2**30) else 0
                u.write(REPORT/'status.json',dict(state='waiting_for_complete_full_size_neural_headroom',other_jobs_rss_bytes=other,memory=current,good_samples=good,observed=stamp()))
                if good<2:previous=current;time.sleep(20)
            for name,digest in hashes.items():
                if u.sha(u.ROOT/name)!=digest:raise ValueError('Predeclared attention study code changed')
            execute('upstream_attention2_screen.py','ridge.log');execute('verify_upstream_attention2_screen.py','independent-ridge.log')
            execute('upstream_attention2_virtual_features.py','features.log');model.configure();model.prepare()
            monitored(['--worker'],'nn.log');monitored(['--replay'],'saved-nn-replay.log')
            model.configure();model.base.preceding.summarize()
        results=[]
        for tree,script,name in MEANS:
            while not (u.ROOT/'experiments'/tree/'independent-verification.json').exists():
                u.write(REPORT/'status.json',dict(state='waiting_for_complete_declared_tree_family_proof',tree=tree,observed=stamp()));time.sleep(20)
            execute(script,name+'.log')
            with (REPORT/(name+'-independent.log')).open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_development_mean.py'),name],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            p=u.ROOT/'experiments'/name/'independent-mean-check.json';check=u.read(p)
            results.append(dict(tree=tree,rmse=check['rmse'],independent_mean_sha256=u.sha(p)))
        for name,digest in hashes.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen fixed study changed')
        current=u.read(u.ROOT/'experiments/fresh-reproduction-status.json')['best_development_rmse']
        u.write(REPORT/'completion.json',dict(state='complete_original_attention_pair_neural_study_and_both_declared_tree_means',observed=stamp(),results=results,
            reference_at_declaration=.8592309819864167,current_verified_reference_at_summary=current,
            improves_current_reference=min(r['rmse'] for r in results)<current,automatic_production_started=False,qualifying_labels_read=False,goal_complete=False))
        u.write(REPORT/'status.json',dict(state='complete_actual_replayed_attention_pair_study',results=results,goal_complete=False))

if __name__=='__main__':
    try:run()
    except BaseException as error:
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',error=repr(error),observed=stamp()));raise
