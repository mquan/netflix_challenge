#!/usr/bin/env python3
"""Conditional own103 NN full refit with exact, separately bound tree91 reuse."""
import argparse
import datetime as dt
import fcntl
from pathlib import Path
import sys
import time
import numpy as np
import upstream_context91_production as previous
import upstream_cf48_103_context as development
import upstream_cf48_103_bank as bank

u,n,core,template=previous.u,previous.n,previous.core,previous.template
RUN=u.ROOT/'artifacts/upstream-cf48-103-production'
REPORT=u.ROOT/'experiments/upstream-cf48-103-production'
SOURCE=development.base.source

def code_hashes():
    paths=[Path(__file__),Path(previous.__file__),Path(template.__file__),previous.CORE_FILE,
        u.ROOT/'scripts/verify_upstream_cf48_103_bank.py']
    return {**development.code_hashes(),**bank.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}

def development_gate():
    development.configure();n.verify_prepared()
    ep=development.REPORT/'evaluation.json';pp=development.REPORT/'independent-verification.json'
    rp=development.REPORT/'nn-replay.json';e,p,r=[u.read(path) for path in [ep,pp,rp]]
    if (e['state']!='complete' or p['state']!='complete' or p['evaluation_sha256']!=u.sha(ep) or
            not p['fixed_average_bitwise_equal'] or [v['fold'] for v in r['records']]!=list(range(5)) or
            not all(v['bitwise_equal'] for v in r['records'])):raise ValueError('Completed independent development/replay required')
    candidate=next(v for v in p['records'] if v['arm']=='fixed-half-tree-nn')
    vector=development.RUN/'fixed-half-tree-nn.f32'
    if u.sha(vector)!=candidate['prediction_sha256']:raise ValueError('Development vector changed')
    score=next(v['rmse'] for v in e['records'] if v['arm']=='fixed-half-tree-nn')
    if abs(score-candidate['independent_rmse'])>1e-12:raise ValueError('Independent development score differs')
    sources={str(path.relative_to(u.ROOT)):u.sha(path) for path in [ep,pp,rp,vector,development.REPORT/'manifest.json']}
    return dict(rmse=score,improved=score<e['previous_best_rmse'],previous=e['previous_best_rmse'],source_sha256=sources)

def budget_sources():
    choices={};sources={}
    for family,report,cache in [('tree',SOURCE.REPORT,SOURCE.RUN),('nn',development.REPORT,development.RUN)]:
        manifest=report/'manifest.json';sources[str(manifest.relative_to(u.ROOT))]=u.sha(manifest);records=[]
        for fold in range(5):
            path=report/family/f'{fold}.json';receipt=u.read(path)
            if receipt['manifest_sha256']!=u.sha(manifest) or receipt['fold']!=fold or receipt['family']!=family:
                raise ValueError('Actual originating budget receipt required')
            for name,digest in receipt['outputs'].items():
                artifact=cache/family/str(fold)/name
                if u.sha(artifact)!=digest:raise ValueError('Originating model changed')
                sources[str(artifact.relative_to(u.ROOT))]=digest
            sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(receipt['chosen_iterations'])
        choices[family]=core.budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    return choices,sources

def tree_origin(iterations):
    # The original tree is bound to its original paired91 inputs and ridge.
    # The new NN103 ridge must never be used to infer these saved trees.
    previous.bank=bank.parent
    return previous.tree_origin(u.read(bank.parent.REPORT/'manifest.json'),iterations)

def prepare():
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial own103 production preserved')
    measured=development_gate()
    if not measured['improved']:raise ValueError('Own103 does not improve the verified previous development best')
    data=bank.build();proofpath=bank.REPORT/'column-verification.json';proof=u.read(proofpath)
    if (proof['state']!='complete' or proof['manifest_sha256']!=u.sha(bank.REPORT/'manifest.json') or
            proof['verifier_sha256']!=u.sha(u.ROOT/'scripts/verify_upstream_cf48_103_bank.py') or
            not proof['all_retained_columns_checked'] or not proof['all_twelve_raw_transforms_checked'] or
            not proof['added_known_columns_bitwise_match_development']):
        raise ValueError('Independent paired103 column verification required')
    configure();iterations,sources=budget_sources();reused,origin_sources=tree_origin(iterations['tree'])
    for collection in [data['source_sha256'],measured['source_sha256'],origin_sources]:
        for name,digest in collection.items():
            if name in sources and sources[name]!=digest:raise ValueError('Conflicting source lineage')
            sources[name]=digest
    for path in [bank.REPORT/'manifest.json',proofpath]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    if z.shape!=(len(y),179):raise ValueError('Own103 ridge schema mismatch')
    gram=np.zeros((179,179));rhs=np.zeros(179)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    weights=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True);np.save(RUN/'ridge.npy',weights)
    plan=dict(state='frozen_before_nn103_refit',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        seeds=n.CONFIG['seeds'],iterations=iterations,qualifying_rows=2817131,qualifying_labels_read=False,
        ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,development_rmse=measured['rmse'],tree_reuse=reused,
        family_schemas=dict(tree=dict(prediction_columns=91,ridge_columns=167,features=227,source_report=str(previous.ORIGIN_REPORT)),
            nn=dict(prediction_columns=103,ridge_columns=179,features=239)),
        iteration_rule='Per-seed medians of actual original tree91 and own103 NN five-fold nested training receipts.',
        tree_replay_scope='Prior actual saved-tree inference replay is revalidated with exact original bank, ridge, implementation, model and vector hashes. Unchanged tree inference is not rerun.',
        final_rule='Full-known NN103 fit and saved-model inference replay; exact own tree91 reuse; fixed float32 half mean frozen before canonical recovered-label audit.',
        bank_scope_note='Paired parent native realization retains the previously disclosed differences from cached development. No qualifying outcomes enter fitting or selection.')
    u.write(REPORT/'plan.json',plan);return plan

def replay_origin_tree():
    # Revalidate the already completed actual inference replay and every bound
    # original source/artifact; publication equality is checked by template.worker.
    plan=core.verify();reused,_=tree_origin(plan['iterations']['tree'])
    if reused!=plan['tree_reuse']:raise ValueError('Original tree replay lineage changed')

def configure():
    development.configure();n.RUN,n.REPORT=RUN,REPORT
    template.RUN,template.REPORT=RUN,REPORT
    template.FIRST_RUN,template.FIRST_REPORT=previous.ORIGIN_RUN,previous.ORIGIN_REPORT
    template.replay_origin_tree=replay_origin_tree
    core.RUN,core.REPORT,core.bank,core.voting=RUN,REPORT,bank,development
    core.code_hashes,core.prepare,core.worker=code_hashes,prepare,template.worker
    core.__file__=str(Path(__file__).resolve())

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (SOURCE.REPORT/'run.lock').open('a') as fitting:
            while True:
                try:fcntl.flock(fitting,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_active_nested_fit',observed=dt.datetime.now(dt.timezone.utc).isoformat()));time.sleep(30)
            result=development_gate()
            if not result['improved']:
                u.write(REPORT/'queue-status.json',dict(state='not_promoted_development_did_not_improve',**result));return
            configure();core.run()
            u.write(REPORT/'queue-status.json',dict(state='audit_complete_pending_separate_frozen_mean_verification'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['tree','nn']);p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    template.worker(a.family,a.replay) if a.family else run()
