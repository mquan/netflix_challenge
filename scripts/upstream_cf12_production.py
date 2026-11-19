#!/usr/bin/env python3
"""Qualifying path for the measured85-tree/86-context-NN development blend."""
import argparse
import datetime as dt
import fcntl
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import upstream_production as core
import upstream_cf12_context as development
import upstream_cf12_bank as bank
import upstream_extension_run as extension
import upstream_context_production as previous_queue

u,n=core.u,core.n
CORE_FILE=Path(core.__file__)
ORIGINAL_WORKER=core.worker
RUN=u.ROOT/'artifacts/upstream-cf12-production'
REPORT=u.ROOT/'experiments/upstream-cf12-production'
FIRST_REPORT=u.ROOT/'experiments/upstream-phase-production'
FIRST_RUN=u.ROOT/'artifacts/upstream-phase-production'
OPERATIONAL=u.ROOT/'experiments/upstream-production/status.json'


def code_hashes():
    paths=[CORE_FILE,Path(__file__),Path(bank.__file__),Path(extension.__file__),
        Path(previous_queue.__file__),u.ROOT/'scripts/upstream_phase_production.py',
        u.ROOT/'scripts/upstream_source_extension_watch.py']
    return {**development.code_hashes(),**bank.code_hashes(),
        **{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def development_gate():
    development.configure();n.verify_prepared()
    evaluation=development.REPORT/'evaluation.json';proof=u.read(development.REPORT/'independent-verification.json')
    replay=u.read(development.REPORT/'nn-replay.json')
    if proof['evaluation_sha256']!=u.sha(evaluation) or not proof['fixed_average_bitwise_equal']:
        raise ValueError('CF12 full development score is not independently verified')
    if [r['fold'] for r in replay['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in replay['records']):
        raise ValueError('CF12 complete neural replay required')
    result=u.read(evaluation);candidate=next(r['rmse'] for r in result['records'] if r['arm']=='fixed-half-tree-nn')
    if not candidate<result['previous_best_rmse']:raise ValueError('CF12 candidate did not improve complete development')
    return candidate


def tree_origin(data,expected_iterations):
    plan_path=FIRST_REPORT/'plan.json';receipt_path=FIRST_REPORT/'tree.json';replay_path=FIRST_REPORT/'tree-replay.json'
    plan,receipt,replay=[u.read(p) for p in [plan_path,receipt_path,replay_path]]
    parent_manifest=bank.parent.REPORT/'manifest.json';parent_key=str(parent_manifest.relative_to(u.ROOT))
    if (plan['qualifying_labels_read'] is not False or plan['qualifying_rows']!=data['qualifying_rows'] or
        plan['source_sha256'].get(parent_key)!=data['parent_bank_manifest_sha256'] or
        data['parent_bank_manifest_sha256']!=u.sha(parent_manifest)):
        raise ValueError('Reused tree was not fitted on this exact85-predictor parent bank')
    if plan['seeds']!=n.CONFIG['seeds'] or plan['iterations']['tree']!=expected_iterations:
        raise ValueError('Reused tree seeds or original nested budgets differ')
    for key in ['tree_lr','tree_leaves','tree_min_leaf','threads','ridge_lambda']:
        if plan['config'][key]!=n.CONFIG[key]:raise ValueError('Reused tree training configuration differs')
    if plan['versions']!=n.versions() or plan['code_sha256'].get('scripts/upstream_production.py')!=u.sha(CORE_FILE):
        raise ValueError('Reused tree training implementation/environment changed')
    if receipt['plan_sha256']!=u.sha(plan_path) or receipt['qualifying_labels_read'] is not False:
        raise ValueError('Reused tree receipt binding differs')
    if (not replay['bitwise_equal'] or replay['plan_sha256']!=u.sha(plan_path) or
        replay['family_receipt_sha256']!=u.sha(receipt_path) or
        replay['prediction_sha256']!=receipt['outputs']['qualifying.f32']):
        raise ValueError('Reused tree semantic replay is incomplete')
    expected={'qualifying.f32'}|{f'tree-{seed}.txt' for seed,count in zip(plan['seeds'],expected_iterations) if count}
    if set(receipt['outputs'])!=expected:raise ValueError('Reused tree model artifacts incomplete')
    sources=dict(plan['source_sha256'])
    for name,digest in plan['code_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Reused tree source code changed')
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Reused tree source inputs changed')
    for name,digest in receipt['outputs'].items():
        p=FIRST_RUN/'tree'/name
        if u.sha(p)!=digest:raise ValueError('Reused tree artifact changed')
        sources[str(p.relative_to(u.ROOT))]=digest
    ridge=FIRST_RUN/'ridge.npy'
    if u.sha(ridge)!=plan['ridge_sha256']:raise ValueError('Reused tree ridge changed')
    for p in [plan_path,receipt_path,replay_path,ridge]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    return dict(plan_sha256=u.sha(plan_path),family_receipt_sha256=u.sha(receipt_path),
        prediction_sha256=receipt['outputs']['qualifying.f32'],source_directory=str(FIRST_RUN/'tree'),
        source_report=str(FIRST_REPORT),prediction_columns=85,fit_performed_in_this_run=False),sources


def prepare():
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial CF12 production preserved')
    candidate=development_gate();data=bank.build();development.configure()
    sources=dict(data['source_sha256'])
    for p in [bank.REPORT/'manifest.json',development.REPORT/'manifest.json',development.REPORT/'evaluation.json',
              development.REPORT/'independent-verification.json',development.REPORT/'nn-replay.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    choices={}
    for family,origin,cache in [('tree',development.parent.REPORT,development.parent.RUN),('nn',development.REPORT,development.RUN)]:
        records=[]
        for fold in range(5):
            p=origin/family/(str(fold)+'.json');receipt=u.read(p)
            if receipt['manifest_sha256']!=u.sha(origin/'manifest.json'):raise ValueError('CF12 budget source changed')
            for name,digest in receipt['outputs'].items():
                if u.sha(cache/family/str(fold)/name)!=digest:raise ValueError('CF12 budget artifact changed')
            sources[str(p.relative_to(u.ROOT))]=u.sha(p);records.append(receipt['chosen_iterations'])
        choices[family]=core.budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    reused,reused_sources=tree_origin(data,choices['tree'])
    for name,digest in reused_sources.items():
        if name in sources and sources[name]!=digest:raise ValueError('Conflicting tree/NN source binding')
        sources[name]=digest
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    if z.shape!=(len(y),162):raise ValueError('CF12 neural ridge schema differs')
    gram=np.zeros((162,162));rhs=np.zeros(162)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    w=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda']);np.save(RUN/'ridge.npy',w)
    plan=dict(state='frozen_before_neural_refitting',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        iterations=choices,seeds=n.CONFIG['seeds'],qualifying_rows=data['qualifying_rows'],qualifying_labels_read=False,
        ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,tree_reuse=reused,development_rmse=candidate,
        iteration_rule='Original85 tree budgets are verified against its completed full-data fit; new86 NN budgets are the per-seed medians of its five actual nested receipts.',
        final_rule='Reuse first verified85-predictor full-ProbeX tree, with original ridge/inputs and explicit lineage. Fresh full-ProbeX86-predictor context NN refit. Three seed means, family clipping, fixed equal float32-family average. Independent semantic replay, freeze, then recovered audit.')
    u.write(REPORT/'plan.json',plan);return plan


def replay_origin_tree():
    # A clean interpreter retains the original85 plan/configuration and avoids
    # mixing the old tree's161-column ridge with the new NN's162-column ridge.
    with (REPORT/'tree-origin-replay.log').open('a') as log:
        subprocess.run([sys.executable,str(u.ROOT/'scripts/upstream_phase_production.py'),
            '--family','tree','--replay'],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def worker(family,replay_only=False):
    if family=='nn':return ORIGINAL_WORKER(family,replay_only)
    plan=core.verify();origin=FIRST_RUN/'tree/qualifying.f32';root=RUN/'tree';path=root/'qualifying.f32'
    receipt_path=REPORT/'tree.json';expected=plan['tree_reuse']['prediction_sha256']
    if u.sha(origin)!=expected:raise ValueError('Reused tree qualifying vector changed')
    values=np.fromfile(origin,dtype='<f4')
    if values.shape!=(plan['qualifying_rows'],) or not np.isfinite(values).all():raise ValueError('Reused tree vector invalid')
    if not receipt_path.exists():
        if replay_only:raise ValueError('Reused tree publication receipt missing')
        if root.exists():raise ValueError('Partial reused tree publication preserved')
        root.mkdir();u.link_or_copy(origin,path)
        u.write(receipt_path,dict(plan_sha256=u.sha(REPORT/'plan.json'),outputs={'qualifying.f32':u.sha(path)},
            fit_performed=False,qualifying_labels_read=False,reused_from=plan['tree_reuse']))
    receipt=u.read(receipt_path)
    if (receipt['plan_sha256']!=u.sha(REPORT/'plan.json') or receipt['fit_performed'] is not False or
        receipt['reused_from']!=plan['tree_reuse'] or receipt['outputs']!={'qualifying.f32':expected} or u.sha(path)!=expected):
        raise ValueError('Reused tree publication lineage changed')
    if replay_only:
        replay_origin_tree()
        if not np.array_equal(np.fromfile(path,dtype='<f4').view(np.uint32),np.fromfile(origin,dtype='<f4').view(np.uint32)):
            raise ValueError('Reused tree publication replay differs')
        u.write(REPORT/'tree-replay.json',dict(bitwise_equal=True,plan_sha256=u.sha(REPORT/'plan.json'),
            family_receipt_sha256=u.sha(receipt_path),prediction_sha256=u.sha(path),
            fit_performed=False,origin_replay_sha256=u.sha(FIRST_REPORT/'tree-replay.json')))


def configure():
    development.configure()
    core.RUN,core.REPORT,core.bank,core.voting=RUN,REPORT,bank,development
    core.code_hashes,core.prepare,core.worker=code_hashes,prepare,worker
    core.__file__=str(Path(__file__).resolve())


def relay(stopping):
    while not stopping.wait(5):
        p=REPORT/'status.json'
        if p.exists():
            state=u.read(p);state['scheduler']='cf12-context-production';u.write(OPERATIONAL,state)


def run_queued():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);candidate=development_gate()
        lock_path=extension.experiment.REPORT/'run.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
        while True:
            gate=extension.first_gate()
            if gate=='target_already_verified':
                u.write(REPORT/'queue-status.json',dict(state='not_needed_target_already_verified'));return
            if gate=='waiting':
                u.write(REPORT/'queue-status.json',dict(state='waiting_for_first_frozen_audit',first_gate=gate));time.sleep(30);continue
            with lock_path.open('a') as shared:
                try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_extension_development'));time.sleep(30);continue
                p=extension.REPORT/'status.json';state=u.read(p).get('state') if p.exists() else None
                if state not in previous_queue.SAFE_EXTENSION_STATES:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_extension_production',extension_state=state));time.sleep(30);continue
                final=u.ROOT/'experiments/upstream-extension-production'
                if (final/'independent-qualifying-verification.json').exists() and extension.first_gate(final)=='target_already_verified':
                    u.write(REPORT/'queue-status.json',dict(state='not_needed_extension_target_verified'));return
                prior=extension.experiment.REPORT/'evaluation.json'
                if prior.exists():
                    proof=u.read(extension.experiment.REPORT/'independent-verification.json')
                    if proof['evaluation_sha256']!=u.sha(prior):raise ValueError('Extension comparison not independently verified')
                    score=next(r['rmse'] for r in u.read(prior)['records'] if r['arm']=='fixed-half-tree-nn')
                    if score<=candidate:
                        u.write(REPORT/'queue-status.json',dict(state='stronger_known_development_available',candidate=candidate,extension=score));return
                while development.resources.other_rss(set())>17*2**30:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_memory'));time.sleep(30)
                # The generic final runner waits for its bank before calling
                # prepare, so assemble the completed paired schema here.
                bank.build();u.write(REPORT/'queue-status.json',dict(state='final_production_running'))
                stopping=threading.Event();thread=threading.Thread(target=relay,args=(stopping,),daemon=True);thread.start()
                try:core.run()
                finally:
                    stopping.set();thread.join()
                    if (REPORT/'status.json').exists():
                        last=u.read(REPORT/'status.json');last['scheduler']='cf12-context-production';u.write(OPERATIONAL,last)
                u.write(REPORT/'queue-status.json',dict(state='production_audit_complete',goal_requires_independent_review=True));return


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();configure()
    if args.family:worker(args.family,args.replay)
    else:run_queued()
