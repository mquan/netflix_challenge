#!/usr/bin/env python3
"""Fresh tree91 refit plus exactly matching own full-fit contextNN86 reuse."""
import argparse
import copy
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_cf12_matching_production as origin
import upstream_extension_bank as bank
import upstream_source_extension as development
import upstream_tree91_context86 as candidate
import upstream_cf96_memory_resume as memory

u,n,core=origin.u,origin.n,origin.core
CORE_FILE=origin.CORE_FILE
ORIGINAL_WORKER=origin.ORIGINAL_WORKER
ORIGINAL_WRITE=u.write
RUN=u.ROOT/'artifacts/upstream-tree91-context86-production'
REPORT=u.ROOT/'experiments/upstream-tree91-context86-production'


def code_hashes():
    paths=[Path(__file__),Path(candidate.__file__),Path(memory.__file__),CORE_FILE]
    return {**origin.code_hashes(),**bank.code_hashes(),**development.code_hashes(),
            **{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def candidate_gate():
    path=candidate.REPORT/'verification.json';record=u.read(path)
    if (record['state']!='complete_verified_fixed_combination' or not record['improves_previous'] or
            record['qualifying_labels_read'] is not False or record['weights']!=[.5,.5] or
            record['origins']!=[list(x) for x in candidate.ORIGINS] or record['rows']!=1128164 or
            not record['fixed_average_bitwise_equal'] or
            not record['all_ten_original_fold_receipts_and_models_rehashed'] or
            not record['original_saved_model_replays_verified']):
        raise ValueError('Complete verified fixed known-ProbeX combination required')
    for relative,digest in record['source_sha256'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Candidate source changed')
    prediction=candidate.RUN/'fixed-half-tree-nn.f32'
    if u.sha(prediction)!=record['prediction_sha256']:raise ValueError('Candidate prediction changed')
    return record


def budgets():
    choices={};sources={}
    for family,name in candidate.ORIGINS:
        report=u.ROOT/'experiments'/name;cache=u.ROOT/'artifacts'/name;records=[]
        manifest=report/'manifest.json';sources[str(manifest.relative_to(u.ROOT))]=u.sha(manifest)
        for fold in range(5):
            path=report/family/f'{fold}.json';receipt=u.read(path)
            if (receipt['manifest_sha256']!=u.sha(manifest) or receipt['fold']!=fold or receipt['family']!=family):
                raise ValueError('Actual originating budget receipt required')
            for name,digest in receipt['outputs'].items():
                model=cache/family/str(fold)/name
                if u.sha(model)!=digest:raise ValueError('Actual budget model changed')
                sources[str(model.relative_to(u.ROOT))]=digest
            sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(receipt['chosen_iterations'])
        choices[family]=core.budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    return choices,sources


def check_neural_origin(plan,receipt,replay,config,iterations,plan_digest,receipt_digest):
    if (plan['config']!=config or plan['seeds']!=config['seeds'] or
            plan['iterations']['nn']!=iterations or plan['qualifying_rows']!=2817131 or
            plan['qualifying_labels_read'] is not False):raise ValueError('Reused NN recipe differs')
    expected={'qualifying.f32','scaler.npz'}|{
        f'nn-{seed}.pt' for seed,count in zip(plan['seeds'],iterations) if count}
    if (set(receipt['outputs'])!=expected or receipt['plan_sha256']!=plan_digest or
            receipt['qualifying_labels_read'] is not False):raise ValueError('Reused NN model receipt differs')
    if (not replay['bitwise_equal'] or replay['plan_sha256']!=plan_digest or
            replay['family_receipt_sha256']!=receipt_digest or
            replay['prediction_sha256']!=receipt['outputs']['qualifying.f32']):
        raise ValueError('Reused NN saved-model replay differs')


def neural_origin(data,iterations):
    paths=[origin.REPORT/'plan.json',origin.REPORT/'nn.json',origin.REPORT/'nn-replay.json']
    plan,receipt,replay=[u.read(p) for p in paths]
    check_neural_origin(plan,receipt,replay,origin.development.parent.CONFIG,iterations,u.sha(paths[0]),u.sha(paths[1]))
    if plan['versions']!=n.versions():raise ValueError('Reused NN environment differs')
    schema_path=origin.bank.REPORT/'manifest.json';schema=u.read(schema_path)
    parent=bank.parent.REPORT/'manifest.json';parent_key=str(parent.relative_to(u.ROOT))
    if ((schema['prediction_columns'],schema['design_columns'],schema['nonlinear_features'])!=(86,162,222) or
            schema['rows']!=data['rows'] or schema['qualifying_rows']!=data['qualifying_rows'] or
            schema['qualifying_labels_read'] is not False or
            schema['parent_bank_manifest_sha256']!=data['source_sha256'][parent_key] or
            schema['parent_bank_manifest_sha256']!=u.sha(parent) or
            plan['source_sha256'].get(str(schema_path.relative_to(u.ROOT)))!=u.sha(schema_path)):
        raise ValueError('Reused NN exact bank/row provenance differs')
    sources=dict(plan['source_sha256'])
    for relative,digest in {**plan['source_sha256'],**plan['code_sha256']}.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Reused NN source changed')
        sources[relative]=digest
    for name,digest in receipt['outputs'].items():
        path=origin.RUN/'nn'/name
        if u.sha(path)!=digest:raise ValueError('Reused NN artifact changed')
        sources[str(path.relative_to(u.ROOT))]=digest
    ridge=origin.RUN/'ridge.npy'
    if u.sha(ridge)!=plan['ridge_sha256']:raise ValueError('Reused NN ridge changed')
    for path in paths+[schema_path,ridge]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    return dict(source_directory=str(origin.RUN/'nn'),source_report=str(origin.REPORT),
        plan_sha256=u.sha(paths[0]),family_receipt_sha256=u.sha(paths[1]),
        prediction_sha256=receipt['outputs']['qualifying.f32'],prediction_columns=86,
        design_columns=162,features_plus_ridge=223,fit_performed_in_this_run=False),sources


def prepare():
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial new production preparation preserved')
    measured=candidate_gate();data=bank.build();configure()
    if (data['prediction_columns'],data['design_columns'],data['nonlinear_features'])!=(91,167,227):
        raise ValueError('Fresh tree91 paired schema differs')
    choices,sources=budgets();reused,reused_sources=neural_origin(data,choices['nn'])
    for collection in [measured['source_sha256'],data['source_sha256'],reused_sources]:
        for name,digest in collection.items():
            if name in sources and sources[name]!=digest:raise ValueError('Conflicting frozen source binding')
            sources[name]=digest
    for path in [candidate.REPORT/'verification.json',bank.REPORT/'manifest.json']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    gram=np.zeros((167,167));rhs=np.zeros(167)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    weights=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    np.save(RUN/'ridge.npy',weights)
    plan=dict(state='frozen_before_tree91_refitting',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        seeds=n.CONFIG['seeds'],iterations=choices,qualifying_rows=2817131,qualifying_labels_read=False,
        ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,
        development_rmse=measured['rmse'],neural_reuse=reused,
        iteration_rule='Per-seed medians of the five actual tree91 and contextNN86 development receipts. Reused NN origin must have exactly these budgets, full config, versions, ridge and bank.',
        final_rule='Fresh full-known-ProbeX tree91 with its own167-column ridge and228 inputs. Reuse our fully replayed full-known-ProbeX contextNN86 with its distinct162-column ridge and223 inputs. Three-seed clipped family means and fixed equal float32 mean; freeze before recovered audit.')
    u.write(REPORT/'plan.json',plan);return plan


def worker(family,replay_only=False):
    if family=='tree':return ORIGINAL_WORKER(family,replay_only)
    plan=core.verify();root=RUN/'nn';path=root/'qualifying.f32';receipt_path=REPORT/'nn.json'
    original=origin.RUN/'nn/qualifying.f32';expected=plan['neural_reuse']['prediction_sha256']
    if u.sha(original)!=expected:raise ValueError('Original full NN prediction changed')
    values=np.fromfile(original,dtype='<f4')
    if values.shape!=(2817131,) or not np.isfinite(values).all():raise ValueError('Invalid full NN prediction')
    if not receipt_path.exists():
        if replay_only or root.exists():raise ValueError('Incomplete reused NN publication preserved')
        root.mkdir();u.link_or_copy(original,path)
        u.write(receipt_path,dict(plan_sha256=u.sha(REPORT/'plan.json'),outputs={'qualifying.f32':u.sha(path)},
            fit_performed=False,qualifying_labels_read=False,reused_from=plan['neural_reuse']))
    receipt=u.read(receipt_path)
    if (receipt['plan_sha256']!=u.sha(REPORT/'plan.json') or receipt['fit_performed'] is not False or
            receipt['reused_from']!=plan['neural_reuse'] or receipt['outputs']!={'qualifying.f32':expected} or
            u.sha(path)!=expected):raise ValueError('Reused NN publication lineage changed')
    if replay_only:
        with (REPORT/'nn-origin-replay.log').open('a') as log:
            subprocess.run([sys.executable,str(Path(origin.__file__)),'--family','nn','--replay'],
                cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not np.array_equal(values.view(np.uint32),np.fromfile(path,dtype='<f4').view(np.uint32)):
            raise ValueError('Reused NN publication replay differs')
        u.write(REPORT/'nn-replay.json',dict(bitwise_equal=True,plan_sha256=u.sha(REPORT/'plan.json'),
            family_receipt_sha256=u.sha(receipt_path),prediction_sha256=u.sha(path),fit_performed=False,
            origin_replay_sha256=u.sha(origin.REPORT/'nn-replay.json')))


def publish(path,value):
    if Path(path)==REPORT/'report.json' and value.get('state')=='complete':
        value=dict(value,scope='Own freshly trained tree91 plus our exact full-fit contextNN86 reused with verified native source, bank, ridge, budgets, models and independent semantic replay. Fixed equal float32 family mean frozen before the adaptive recovered-group audit.')
    return ORIGINAL_WRITE(path,value)


def configure():
    n.CONFIG=copy.deepcopy(u.read(development.REPORT/'manifest.json')['config'])
    n.RUN,n.REPORT=RUN,REPORT
    core.RUN,core.REPORT,core.bank,core.voting=RUN,REPORT,bank,development
    core.code_hashes,core.prepare,core.worker=code_hashes,prepare,worker
    core.__file__=str(Path(__file__).resolve());u.write=publish


def origin_processes():
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines();found={}
    for row in rows:
        pid,command=row.strip().split(None,1)
        if any(Path(arg).name=='upstream_cf12_matching_production.py' for arg in command.split()[1:]):
            found[pid]=command
    return found


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);candidate_gate()
        while True:
            active=origin_processes()
            u.write(REPORT/'queue-status.json',dict(state='waiting_for_matching86_completion',
                observed=dt.datetime.now(dt.timezone.utc).isoformat(),actual_origin_processes=active))
            if not active:break
            time.sleep(30)
        audit=origin.REPORT/'independent-qualifying-verification.json'
        if audit.exists() and origin.previous.extension.first_gate(origin.REPORT)=='target_already_verified':
            u.write(REPORT/'queue-status.json',dict(state='not_needed_existing_target_pass',goal_requires_independent_review=True));return
        # Validate the required origin before waiting for resources or creating
        # any new fitting directory. An incomplete origin is never substituted.
        data=bank.build();configure();choices,_=budgets();neural_origin(data,choices['nn'])
        with (development.REPORT/'run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            previous=memory.memory();good=0
            while good<2:
                current=memory.memory();owned=memory.footprints();total=sum(v['bytes'] for v in owned.values())
                disk=os.statvfs(u.ROOT);free=disk.f_bavail*disk.f_frsize
                good=good+1 if memory.admitted(previous,current,total) and free>26*2**30 else 0
                u.write(REPORT/'queue-status.json',dict(state='waiting_for_physical_memory',
                    observed=dt.datetime.now(dt.timezone.utc).isoformat(),memory=current,owned_workers=owned,
                    owned_footprint_bytes=total,disk_free_bytes=free,consecutive_good_samples=good))
                if good<2:previous=current;time.sleep(30)
            configure();u.write(REPORT/'queue-status.json',dict(state='production_running'));core.run()
            u.write(REPORT/'queue-status.json',dict(state='production_audit_complete',goal_requires_independent_review=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=['tree','nn']);p.add_argument('--replay',action='store_true');a=p.parse_args()
    configure()
    if a.family:worker(a.family,a.replay)
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'queue-status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
