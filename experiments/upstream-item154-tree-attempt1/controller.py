#!/usr/bin/env python3
"""Fit the unchanged nested tree recipe on own107 plus154 original/own item context."""
import argparse
import fcntl
import os
import subprocess
import datetime as dt
from pathlib import Path
import sys
import time
import numpy as np
import upstream_cf12_tree as template
import upstream_item154_mps_context as neural
import upstream_cf48_103_context as old_neural
import upstream_item154_features as data
import upstream_latent90_local as resources

u,n=template.u,template.n
RUN=u.ROOT/'artifacts/upstream-item154-tree'
REPORT=u.ROOT/'experiments/upstream-item154-tree'
CONFIG=dict(neural.CONFIG)

def code_hashes():
    return {**neural.code_hashes(),**old_neural.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in
        [Path(__file__),Path(template.__file__),Path(resources.__file__),u.ROOT/'scripts/upstream_rbm50_tree.py']}}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,CONFIG,code_hashes
    template.RUN,template.REPORT,template.CONFIG=RUN,REPORT,CONFIG

def checked_nn(mod):
    ep=mod.REPORT/'evaluation.json';pp=mod.REPORT/'independent-verification.json';rp=mod.REPORT/'nn-replay.json'
    e,p,r=[u.read(path) for path in [ep,pp,rp]]
    row=next(v for v in p['records'] if v['arm']=='nn')
    if (p['state']!='complete' or p['evaluation_sha256']!=u.sha(ep) or
        [v['fold'] for v in r['records']]!=list(range(5)) or not all(v['bitwise_equal'] for v in r['records']) or
        u.sha(mod.RUN/'nn.f32')!=row['prediction_sha256']):raise ValueError('Actual independently verified NN required')
    return row, [ep,pp,rp,mod.REPORT/'manifest.json',mod.RUN/'nn.f32']

def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial tree cache preserved')
    data.configure();dm=n.verify_prepared();proof=u.read(data.REPORT/'cache-column-verification.json')
    if proof['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or not proof['original_support_floats_preserved_bitwise']:
        raise ValueError('Verified own154 context cache required')
    candidates=[(mod,*checked_nn(mod)) for mod in [neural]]
    chosen,row,paths=min(candidates,key=lambda item:item[1]['independent_rmse'])
    sources={}
    for mod,checked,inputs in candidates:
        for p in inputs:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for p in [data.REPORT/'manifest.json',data.REPORT/'cache-column-verification.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    best=u.ROOT/'experiments/upstream-rbm50-tree-item154-mps/verification.json'
    sources[str(best.relative_to(u.ROOT))]=u.sha(best)
    cm=u.read(chosen.REPORT/'manifest.json')
    for key in ['labels.npy','users.npy','groups.npy']:
        if dm['prepared_sha256'][key]!=cm['prepared_sha256'][key]:raise ValueError('Reused NN row/fold identity differs')
    for fold in range(5):
        p=chosen.REPORT/'nn'/f'{fold}.json';receipt=u.read(p)
        if receipt['manifest_sha256']!=u.sha(chosen.REPORT/'manifest.json'):raise ValueError('Originating NN receipt differs')
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
        for name,digest in receipt['outputs'].items():
            a=chosen.RUN/'nn'/str(fold)/name
            if u.sha(a)!=digest:raise ValueError('Originating saved NN changed')
            sources[str(a.relative_to(u.ROOT))]=digest
    configure();RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    for name,digest in dm['prepared_sha256'].items():
        u.link_or_copy(data.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Retained context cache changed')
    u.link_or_copy(chosen.RUN/'nn.f32',RUN/'nn.f32')
    m=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=dm['rows'],prediction_columns=107,design_columns=183,nonlinear_features=397,
        baseline_rmse=dm['baseline_rmse'],previous_best_rmse=u.read(u.ROOT/'experiments/upstream-rbm50-tree-item154-mps/verification.json')['rmse'],
        reused_nn_prediction_sha256=row['prediction_sha256'],reused_nn_report=str(chosen.REPORT.relative_to(u.ROOT)),
        arms=['control','tree','nn','fixed-half-tree-nn'],qualifying_labels_read=False,isolation=dm['isolation'],
        scope='Fixed original tree hyperparameters and five nested user folds, own107 plus all154 original/own learned contexts. Reuse the fixed verified own107/item154 MPS neural partner with actual originating model/replay bindings. No synthetic NN receipts or qualifying-based selection. NN and tree use their own exact source schemas.')
    u.write(REPORT/'manifest.json',m);return m

def wait_memory():
    previous=resources.memory.memory();good=0
    while good<2:
        current=resources.memory.memory();workers=resources.footprints()
        good=good+1 if (sum(r['bytes'] for r in workers.values())<21*2**30 and
            current['compressor_bytes']<6*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_tree_memory',observed=resources.stamp(),workers=workers,memory=current,good_samples=good))
        if good<2:previous=current;time.sleep(30)

def guarded(command,log_path):
    with log_path.open('x') as log:
        child=subprocess.Popen(['nice','-n','10']+command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=template.group_rss(child.pid);other=resources.other_rss({child.pid});free=__import__('shutil').disk_usage(u.ROOT).free
                high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<8*2**30:raise RuntimeError('Item154 tree resource guard; outputs preserved')
                time.sleep(5)
        except BaseException:template.stop(child);raise
        if child.returncode:raise RuntimeError('Item154 tree child failed; outputs preserved')


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing tree declaration preserved')
        u.write(REPORT/'plan.json',dict(created=resources.stamp(),script_sha256=u.sha(Path(__file__)),reference_rmse=.8592309819864167,raw_prediction_columns=107,learned_context_columns=154,features=398,weights=[.5,.5],rule='Unchanged original tree hyperparameters/five nested user folds/seeds/stopping; all154 contexts match the fixed current neural partner. Hardlink original matrices. Actual15 saved-tree replays and independent scores required. No qualifying outcomes, subset/weight search or automatic production.',memory_total_bytes=24*2**30,disk_floor_bytes=8*2**30,qualifying_labels_read=False))
        while not (neural.REPORT/'nn-replay.json').exists() or not (neural.REPORT/'independent-verification.json').exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_complete_rbm50_nn_comparison',observed=resources.stamp()));time.sleep(30)
        with (neural.base.source.REPORT/'run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            wait_memory();prepare();configure()
            u.write(REPORT/'status.json',dict(state='nested_tree_fitting'))
            guarded([sys.executable,str(Path(__file__)),'--worker'],REPORT/'tree.log')
            u.write(REPORT/'status.json',dict(state='saved_tree_replay'))
            guarded([sys.executable,str(Path(__file__)),'--replay'],REPORT/'tree-replay.log')
            result=template.summarize()
            u.write(REPORT/'status.json',dict(state='development_complete',records=result['records'],goal_complete=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    if a.worker:n.worker('tree')
    elif a.replay:template.parent.verifier.replay('tree')
    else:run()
