#!/usr/bin/env python3
"""Full-fit contextNN91 with exact verified full-fit tree91 reuse."""
import argparse
import datetime as dt
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_cf12_production as template
import upstream_extension_context as development
import upstream_extension_bank as bank
import upstream_tree91_context91 as candidate

u,n,core=template.u,template.n,template.core
CORE_FILE=template.CORE_FILE
RUN=u.ROOT/'artifacts/upstream-context91-production'
REPORT=u.ROOT/'experiments/upstream-context91-production'
ORIGIN_RUN=u.ROOT/'artifacts/upstream-tree91-context86-production'
ORIGIN_REPORT=u.ROOT/'experiments/upstream-tree91-context86-production'
ORIGIN_SCRIPT=u.ROOT/'scripts/upstream_tree91_context86_production.py'


def code_hashes():
    paths=[Path(__file__),Path(template.__file__),CORE_FILE,Path(candidate.__file__),ORIGIN_SCRIPT]
    return {**development.code_hashes(),**bank.code_hashes(),
        **{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def development_gate():
    development.configure();n.verify_prepared()
    proof=u.read(candidate.REPORT/'verification.json')
    if (proof['state']!='complete_verified_fixed_combination' or not proof['improves_previous'] or
            not proof['fixed_average_bitwise_equal'] or proof['qualifying_labels_read'] is not False or
            not proof['all_ten_original_fold_receipts_and_models_rehashed']):
        raise ValueError('Verified improved tree91/context91 candidate required')
    for name,digest in proof['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Development binding changed: '+name)
    return proof


def budget_sources():
    choices={};sources={}
    for family,report,cache in [('tree',development.source.REPORT,development.source.RUN),('nn',development.REPORT,development.RUN)]:
        manifest=report/'manifest.json';sources[str(manifest.relative_to(u.ROOT))]=u.sha(manifest);records=[]
        for fold in range(5):
            path=report/family/f'{fold}.json';receipt=u.read(path)
            if receipt['manifest_sha256']!=u.sha(manifest) or receipt['fold']!=fold or receipt['family']!=family:
                raise ValueError('Actual original development budget receipt required')
            for name,digest in receipt['outputs'].items():
                artifact=cache/family/str(fold)/name
                if u.sha(artifact)!=digest:raise ValueError('Budget-source model changed')
                sources[str(artifact.relative_to(u.ROOT))]=digest
            sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(receipt['chosen_iterations'])
        choices[family]=core.budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    return choices,sources


def tree_origin(data,iterations):
    paths=[ORIGIN_REPORT/'plan.json',ORIGIN_REPORT/'tree.json',ORIGIN_REPORT/'tree-replay.json']
    plan,receipt,replay=[u.read(p) for p in paths];schema=bank.REPORT/'manifest.json'
    if (plan['source_sha256'].get(str(schema.relative_to(u.ROOT)))!=u.sha(schema) or
            (data['prediction_columns'],data['design_columns'],data['nonlinear_features'])!=(91,167,227) or
            plan['seeds']!=n.CONFIG['seeds'] or plan['iterations']['tree']!=iterations or
            plan['versions']!=n.versions() or plan['qualifying_rows']!=2817131 or plan['qualifying_labels_read'] is not False):
        raise ValueError('Original full tree91 exact bank/budgets/environment required')
    for key in ['tree_lr','tree_leaves','tree_min_leaf','threads','ridge_lambda']:
        if plan['config'][key]!=n.CONFIG[key]:raise ValueError('Reused tree recipe differs')
    if (receipt['plan_sha256']!=u.sha(paths[0]) or receipt['qualifying_labels_read'] is not False or
            not replay['bitwise_equal'] or replay['plan_sha256']!=u.sha(paths[0]) or
            replay['family_receipt_sha256']!=u.sha(paths[1]) or replay['prediction_sha256']!=receipt['outputs']['qualifying.f32']):
        raise ValueError('Reused tree saved-model replay lineage differs')
    expected={'qualifying.f32'}|{f'tree-{seed}.txt' for seed,count in zip(plan['seeds'],iterations) if count}
    if set(receipt['outputs'])!=expected:raise ValueError('Reused tree model set incomplete')
    sources={**plan['source_sha256'],**plan['code_sha256']}
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Reused tree source changed: '+name)
    for name,digest in receipt['outputs'].items():
        path=ORIGIN_RUN/'tree'/name
        if u.sha(path)!=digest:raise ValueError('Reused tree artifact changed')
        sources[str(path.relative_to(u.ROOT))]=digest
    for p in paths+[ORIGIN_RUN/'ridge.npy']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    if u.sha(ORIGIN_RUN/'ridge.npy')!=plan['ridge_sha256']:raise ValueError('Original tree ridge differs')
    return dict(source_directory=str(ORIGIN_RUN/'tree'),source_report=str(ORIGIN_REPORT),
        plan_sha256=u.sha(paths[0]),family_receipt_sha256=u.sha(paths[1]),prediction_sha256=receipt['outputs']['qualifying.f32'],
        prediction_columns=91,fit_performed_in_this_run=False),sources


def prepare():
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial context91 production preserved')
    measured=development_gate();data=bank.build();configure();iterations,sources=budget_sources()
    reused,reused_sources=tree_origin(data,iterations['tree'])
    for collection in [data['source_sha256'],measured['source_sha256'],reused_sources]:
        for name,digest in collection.items():
            if name in sources and sources[name]!=digest:raise ValueError('Conflicting source lineage: '+name)
            sources[name]=digest
    for path in [candidate.REPORT/'verification.json',bank.REPORT/'manifest.json']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    gram=np.zeros((167,167));rhs=np.zeros(167)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    weights=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True);np.save(RUN/'ridge.npy',weights)
    if u.sha(RUN/'ridge.npy')!=u.sha(ORIGIN_RUN/'ridge.npy'):raise ValueError('Full same-bank tree/NN ridge must match exactly')
    plan=dict(state='frozen_before_context_nn91_refit',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        seeds=n.CONFIG['seeds'],iterations=iterations,qualifying_rows=2817131,qualifying_labels_read=False,
        ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,development_rmse=measured['rmse'],tree_reuse=reused,
        iteration_rule='Per-seed median of five actual originating tree91 and new contextNN91 fold receipts.',
        final_rule='Exact same-bank full-fit tree91 reuse with fresh original-context NN91 full-known refit; three-seed family means, clipping and fixed equal float32 mean. Both228-input families use the same paired91 bank/ridge. Replay, freeze, then recovered-group audit.',
        bank_scope_note='Paired own91 native realization; some retrained predictors differ from cached development as quantified by existing paired-source verification. No claim of identical cached predictions.')
    u.write(REPORT/'plan.json',plan);return plan


def replay_origin_tree():
    with (REPORT/'tree-origin-replay.log').open('a') as log:
        subprocess.run([sys.executable,str(ORIGIN_SCRIPT),'--family','tree','--replay'],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def configure():
    development.configure();n.RUN,n.REPORT=RUN,REPORT
    template.RUN,template.REPORT,template.FIRST_RUN,template.FIRST_REPORT=RUN,REPORT,ORIGIN_RUN,ORIGIN_REPORT
    template.replay_origin_tree=replay_origin_tree
    core.RUN,core.REPORT,core.bank,core.voting=RUN,REPORT,bank,development
    core.code_hashes,core.prepare,core.worker=code_hashes,prepare,template.worker
    core.__file__=str(Path(__file__).resolve())


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);development_gate()
        shared=development.source.REPORT/'run.lock'
        with shared.open('a') as fitting:
            while True:
                try:fcntl.flock(fitting,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_other_context_fit'));time.sleep(30)
            newer=u.ROOT/'experiments/upstream-cf48-102-context/independent-verification.json'
            if newer.exists():
                proof=u.read(newer);evaluation=newer.parent/'evaluation.json'
                if proof['evaluation_sha256']!=u.sha(evaluation):raise ValueError('Newer context proof binding differs')
                score=next(r['rmse'] for r in u.read(evaluation)['records'] if r['arm']=='fixed-half-tree-nn')
                if score<development_gate()['rmse']:
                    u.write(REPORT/'queue-status.json',dict(state='stronger_known_candidate_available',rmse=score));return
            u.write(REPORT/'queue-status.json',dict(state='full_context91_production'))
            configure();core.run()
            u.write(REPORT/'queue-status.json',dict(state='audit_complete_pending_independent_review'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['tree','nn']);p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    template.worker(a.family,a.replay) if a.family else run()
