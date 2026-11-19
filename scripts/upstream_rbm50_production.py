#!/usr/bin/env python3
"""Predeclared standalone own107/RBM50 NN full fit, replay, freeze and audit."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_production as core
import upstream_rbm50_context as development
import upstream_rbm50_bank as bank
import upstream_latent90_local as resources

u,n=core.u,core.n
RUN=u.ROOT/'artifacts/upstream-rbm50-production'
REPORT=u.ROOT/'experiments/upstream-rbm50-production'
CORE_FILE=Path(core.__file__)

def code_hashes():
    return {**development.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in
        [Path(__file__),CORE_FILE,Path(bank.__file__),Path(resources.__file__),
         u.ROOT/'scripts/verify_upstream_rbm50_bank.py',u.ROOT/'scripts/verify_upstream_rbm50_production.py',
         u.ROOT/'scripts/audit_qualifying.py']}}

def configure():
    development.configure()
    core.RUN,core.REPORT,core.bank,core.code_hashes=RUN,REPORT,bank,code_hashes

def prepare():
    configure()
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial standalone production preserved')
    dm=n.verify_prepared();ep=development.REPORT/'evaluation.json';pp=development.REPORT/'independent-verification.json'
    rp=development.REPORT/'nn-replay.json';e,p,r=[u.read(path) for path in [ep,pp,rp]]
    nn=next(v for v in p['records'] if v['arm']=='nn')
    if (p['state']!='complete' or p['evaluation_sha256']!=u.sha(ep) or
        [v['fold'] for v in r['records']]!=list(range(5)) or not all(v['bitwise_equal'] for v in r['records']) or
        nn['independent_rmse']>=e['previous_best_rmse'] or
        any(v['independent_rmse']<=nn['independent_rmse'] for v in p['records'] if v['arm']!='nn') or
        u.sha(development.RUN/'nn.f32')!=nn['prediction_sha256']):
        raise ValueError('Complete independently verified stronger standalone NN required')
    bm=u.read(bank.REPORT/'manifest.json');bp=bank.REPORT/'column-verification.json';checked=u.read(bp)
    if (checked['state']!='complete' or checked['manifest_sha256']!=u.sha(bank.REPORT/'manifest.json') or
        checked['verifier_sha256']!=u.sha(u.ROOT/'scripts/verify_upstream_rbm50_bank.py') or
        not all(checked[k] for k in ['all_retained_columns_checked','all_four_raw_transforms_checked',
            'all_fifty_support_columns_bitwise_checked','added_known_columns_bitwise_match_development'])):
        raise ValueError('Separate complete paired-bank verification required')
    sources={**bm['source_sha256'],**bm['code_sha256']}
    for path in [ep,pp,rp,development.REPORT/'manifest.json',development.RUN/'nn.f32',bank.REPORT/'manifest.json',bp]:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in bm['files'].items():
        path=bank.RUN/name
        if u.sha(path)!=digest:raise ValueError('Paired bank changed')
        sources[str(path.relative_to(u.ROOT))]=digest
    records=[]
    for fold in range(5):
        path=development.REPORT/'nn'/f'{fold}.json';receipt=u.read(path)
        if receipt['manifest_sha256']!=u.sha(development.REPORT/'manifest.json'):raise ValueError('Actual originating NN budget required')
        for name,digest in receipt['outputs'].items():
            artifact=development.RUN/'nn'/str(fold)/name
            if u.sha(artifact)!=digest:raise ValueError('Originating saved model changed')
            sources[str(artifact.relative_to(u.ROOT))]=digest
        sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(receipt['chosen_iterations'])
    iterations=core.budgets(records,n.CONFIG['nn_epochs'])
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    if z.shape!=(1128164,183):raise ValueError('Full ridge schema differs')
    if not np.array_equal(y,np.load(development.RUN/'labels.npy')):raise ValueError('Known labels differ')
    gram=np.zeros((183,183));rhs=np.zeros(183)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    w=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True);np.save(RUN/'ridge.npy',w)
    plan=dict(state='standalone_nn_selected_before_full_fit_and_new_qualifying_outcomes',created=resources.stamp(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        seeds=n.CONFIG['seeds'],iterations={'nn':iterations},weights={'nn':1.0},
        prediction_columns=107,design_columns=183,nonlinear_features=293,features_plus_ridge=294,
        qualifying_rows=2817131,qualifying_labels_read=False,ridge_sha256=u.sha(RUN/'ridge.npy'),
        prediction_chunk_rows=core.CHUNK,development_rmse=nn['independent_rmse'],previous_best_rmse=e['previous_best_rmse'],
        iteration_rule='Per-seed medians of the five actual nested user-fold NN stopping budgets; no qualifying labels.',
        final_rule='Fit the selected standalone NN on all known labels, actual saved-model reload/inference replay, then exact family vector freeze and canonical recovered-label audit. No tree mixing or qualifying-fitted weights.',
        bank_scope_note='Previously disclosed nine paired-parent realization differences retained. All newly added prediction/context columns match originating development exactly.')
    u.write(REPORT/'plan.json',plan);return plan

def wait_memory():
    previous=resources.memory.memory();good=0
    while good<2:
        current=resources.memory.memory();workers=resources.footprints()
        good=good+1 if (sum(r['bytes'] for r in workers.values())<21*2**30 and
            current['compressor_bytes']<6*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_full_nn_memory',observed=resources.stamp(),workers=workers,memory=current,good_samples=good))
        if good<2:previous=current;time.sleep(30)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        with (development.base.source.REPORT/'run.lock').open('a') as shared:
            u.write(REPORT/'status.json',dict(state='predeclared_waiting_for_active_meta_fit'))
            fcntl.flock(shared,fcntl.LOCK_EX);wait_memory();core.verify()
            for stage,args in [('full_known_nn_fit',['--worker']),('actual_saved_model_replay',['--replay'])]:
                u.write(REPORT/'status.json',dict(state=stage))
                with (REPORT/'nn.log').open('a') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),*args],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt=u.read(REPORT/'nn.json');replay=u.read(REPORT/'nn-replay.json')
            if not replay['bitwise_equal'] or replay['family_receipt_sha256']!=u.sha(REPORT/'nn.json'):
                raise ValueError('Actual saved full-model replay required')
            source=RUN/'nn/qualifying.f32';frozen=REPORT/'qualifying-final.f32';u.link_or_copy(source,frozen)
            if u.sha(frozen)!=receipt['outputs']['qualifying.f32']:raise ValueError('Frozen vector differs')
            u.write(REPORT/'report.json',dict(state='complete',qualifying_rows=2817131,qualifying_labels_read=False,
                frozen_prediction_sha256=u.sha(frozen),plan_sha256=u.sha(REPORT/'plan.json'),
                nn_receipt_sha256=u.sha(REPORT/'nn.json'),nn_replay_sha256=u.sha(REPORT/'nn-replay.json'),
                scope='Predeclared known-CV-only standalone own107/RBM50 NN, fit on known labels and frozen after actual saved-model replay. No qualifying-based candidate or weight selection.'))
            subprocess.run([sys.executable,str(u.ROOT/'scripts/audit_qualifying.py'),'--model-dir',str(REPORT),
                '--output-dir',str(REPORT/'recovered-audit')],cwd=u.ROOT,check=True)
            subprocess.run([sys.executable,str(u.ROOT/'scripts/verify_upstream_rbm50_production.py')],cwd=u.ROOT,check=True)
            u.write(REPORT/'status.json',dict(state='qualifying_audit_and_separate_score_verification_complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    if a.prepare:prepare()
    elif a.worker:core.worker('nn')
    elif a.replay:core.worker('nn',True)
    else:run()
