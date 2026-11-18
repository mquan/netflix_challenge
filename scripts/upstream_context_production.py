#!/usr/bin/env python3
"""Qualifying refit of the measured context improvement, serialized after audit."""
import argparse
import datetime as dt
import fcntl
from pathlib import Path
import threading
import time
import numpy as np
import upstream_context_blend as development
import upstream_production as core
import upstream_phase_bank as bank
import upstream_extension_run as extension

u,n=core.u,core.n
CORE_FILE=Path(core.__file__)
RUN=u.ROOT/'artifacts/upstream-context-production'
REPORT=u.ROOT/'experiments/upstream-context-production'
OPERATIONAL=u.ROOT/'experiments/upstream-production/status.json'
SAFE_EXTENSION_STATES={'waiting_for_first_audit_and_fresh_models','waiting_for_development_lock',
    'development_not_improved','production_audit_complete'}


def code_hashes():
    paths=[CORE_FILE,Path(__file__),Path(bank.__file__),Path(bank.core.__file__),
        Path(bank.phase.__file__),Path(extension.__file__)]
    return {**development.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def development_gate():
    development.configure();n.verify_prepared()
    evaluation=development.REPORT/'evaluation.json'
    proof=u.read(development.REPORT/'independent-verification.json')
    replay=u.read(development.REPORT/'nn-replay.json')
    if proof['evaluation_sha256']!=u.sha(evaluation) or not proof['fixed_average_bitwise_equal']:
        raise ValueError('Context development verification changed')
    if len(replay['records'])!=5 or not all(row['bitwise_equal'] for row in replay['records']):
        raise ValueError('Context model replay is incomplete')
    result=u.read(evaluation)
    candidate=next(row['rmse'] for row in result['records'] if row['arm']=='fixed-half-tree-nn')
    parent=next(row['rmse'] for row in u.read(development.parent.REPORT/'evaluation.json')['records'] if row['arm']=='fixed-half-tree-nn')
    if not candidate<parent:raise ValueError('Context candidate did not improve complete known development')
    return candidate


def prepare():
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial context production preserved')
    candidate=development_gate();data=bank.build();development.configure()
    sources={**data['source_sha256']}
    for p in [bank.REPORT/'manifest.json',development.REPORT/'manifest.json',
        development.REPORT/'evaluation.json',development.REPORT/'independent-verification.json',
        development.REPORT/'nn-replay.json',development.parent.REPORT/'manifest.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    choices={};budget_provenance={}
    for family,origin,cache in [('tree',development.parent.REPORT,development.parent.RUN),
                                 ('nn',development.REPORT,development.RUN)]:
        records=[]
        for fold in range(5):
            p=origin/family/(str(fold)+'.json');receipt=u.read(p)
            if receipt['manifest_sha256']!=u.sha(origin/'manifest.json'):
                raise ValueError('Context budget source changed')
            for name,digest in receipt['outputs'].items():
                if u.sha(cache/family/str(fold)/name)!=digest:raise ValueError('Context budget model changed')
            sources[str(p.relative_to(u.ROOT))]=u.sha(p);records.append(receipt['chosen_iterations'])
        choices[family]=core.budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
        budget_provenance[family]=str(origin.relative_to(u.ROOT))
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    gram=np.zeros((z.shape[1],z.shape[1]));rhs=np.zeros(z.shape[1])
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    w=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    np.save(RUN/'ridge.npy',w)
    plan=dict(state='frozen_before_final_fitting',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        iterations=choices,budget_provenance=budget_provenance,
        iteration_rule='Per-seed median of five original tree budgets and five context NN budgets, separately verified; no synthetic tree receipts.',
        qualifying_labels_read=False,qualifying_rows=data['qualifying_rows'],prediction_chunk_rows=core.CHUNK,
        ridge_sha256=u.sha(RUN/'ridge.npy'),seeds=n.CONFIG['seeds'],development_rmse=candidate,
        final_rule='Fresh full-known-ProbeX ridge/tree/context-NN refits. Three seed means per family, family clipping, fixed equal float32-family average. Qualifying outcomes first read after predictions freeze.')
    u.write(REPORT/'plan.json',plan);return plan


def configure():
    bank.configure();development.configure()
    core.RUN,core.REPORT,core.bank,core.voting=RUN,REPORT,bank,development
    core.code_hashes,core.prepare=code_hashes,prepare
    core.__file__=str(Path(__file__).resolve())


def relay(stopping):
    while not stopping.wait(5):
        p=REPORT/'status.json'
        if p.exists():
            state=u.read(p);state['scheduler']='context-production';u.write(OPERATIONAL,state)


def run_queued():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        candidate=development_gate()
        # Hold the extension's development lock during our complete final run.
        # If its development has already ended, its status prevents overlapping
        # its final production. If it is still idle, it waits for this lock.
        lock_path=extension.experiment.REPORT/'run.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
        while True:
            gate=extension.first_gate()
            if gate=='target_already_verified':
                u.write(REPORT/'queue-status.json',dict(state='not_needed_target_already_verified'));return
            if gate=='waiting' or not (bank.REPORT/'manifest.json').exists():
                u.write(REPORT/'queue-status.json',dict(state='waiting_for_first_frozen_audit',first_gate=gate));time.sleep(30);continue
            with lock_path.open('a') as extension_lock:
                try:fcntl.flock(extension_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_extension_development'));time.sleep(30);continue
                path=extension.REPORT/'status.json'
                state=u.read(path).get('state') if path.exists() else None
                if state not in SAFE_EXTENSION_STATES:
                    u.write(REPORT/'queue-status.json',dict(state='waiting_for_extension_production',extension_state=state));time.sleep(30);continue
                final=u.ROOT/'experiments/upstream-extension-production'
                if (final/'independent-qualifying-verification.json').exists() and extension.first_gate(final)=='target_already_verified':
                    u.write(REPORT/'queue-status.json',dict(state='not_needed_extension_target_verified'));return
                prior=extension.experiment.REPORT/'evaluation.json'
                if prior.exists():
                    proof_path=extension.experiment.REPORT/'independent-verification.json'
                    if not proof_path.exists() or u.read(proof_path)['evaluation_sha256']!=u.sha(prior):
                        raise ValueError('Extension comparison not independently verified')
                    score=next(row['rmse'] for row in u.read(prior)['records'] if row['arm']=='fixed-half-tree-nn')
                    if score<=candidate:
                        u.write(REPORT/'queue-status.json',dict(state='stronger_known_development_available',context_rmse=candidate,extension_rmse=score,goal_complete=False));return
                u.write(REPORT/'queue-status.json',dict(state='final_production_running'))
                stopping=threading.Event();thread=threading.Thread(target=relay,args=(stopping,),daemon=True);thread.start()
                try:core.run()
                finally:
                    stopping.set();thread.join()
                    if (REPORT/'status.json').exists():
                        last=u.read(REPORT/'status.json');last['scheduler']='context-production';u.write(OPERATIONAL,last)
                u.write(REPORT/'queue-status.json',dict(state='production_audit_complete',goal_requires_independent_review=True));return


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    configure()
    if args.family:core.worker(args.family,args.replay)
    else:run_queued()
