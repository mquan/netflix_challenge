#!/usr/bin/env python3
"""Bounded next-stage training, mismatch diagnostics, and fresh-only evaluation."""
import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import upstream_repro as u
import upstream_exchange as x

ROOT = u.ROOT
TRACKED = ROOT/'experiments/upstream-expansion'
RUN = ROOT/'artifacts/upstream-expansion'
ASSIGNMENTS = {'m1':['dnn-64'], 'm3':['dnn-24__attn-32', 'tsvdx4-400lm',
               'tsvdx4-400lm__knnf', 'rbmx2-150', 'rbmx2-150__knns']}
BRANCHES = {'m1':'research/upstream-verification','m3':'research/upstream-reproduction'}
NEW = [*ASSIGNMENTS['m1'], *ASSIGNMENTS['m3']]
ARMS = {'original-eight':x.JOBS, 'plus-temporal':x.JOBS+['tsvdx4-400lm','tsvdx4-400lm__knnf'],
        'plus-rbm':x.JOBS+['rbmx2-150','rbmx2-150__knns'],
        'plus-neural':x.JOBS+['dnn-64','dnn-24__attn-32'], 'all-fourteen':x.JOBS+NEW}
PLAN = dict(version=2, source_commit=u.PIN, assignments=ASSIGNMENTS, arms=ARMS,
            folds=5, fold_seed=20261116, ridge_lambda=1000.0,
            input_clip=[0,6],output_clip=[1.05,4.95],unpenalized_intercept=True,
            tuning='none; fixed before expansion fits',
            limitation='ProbeX development comparisons; base recipes historically development-selected, not untouched Test',
            diagnostics='Separate from expansion features; same-recipe replay versus support including qualifying query identities, with qualifying outcomes zero')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def status(machine, state, active=None, error=None, pushing=False):
    dest=TRACKED/machine
    dest.mkdir(parents=True,exist_ok=True)
    done=sorted(p.stem for p in dest.glob('*.json') if p.name not in ['status.json','diagnostic.json'] and (dest/(p.stem+'.npz')).exists())
    value=dict(machine=machine,state=state,active=active,completed=done,error=error,updated=now())
    previous=u.read(dest/'status.json') if (dest/'status.json').exists() else None
    if previous and all(previous.get(k)==v for k,v in value.items() if k!='updated'):
        elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(previous['updated'])).total_seconds()
        if elapsed<1800:return previous
    u.write(dest/'status.json',value)
    page=ROOT/('EXPANSION_'+machine.upper()+'.md')
    page.write_text('# Upstream expansion: '+machine+'\n\n'+json.dumps(value,indent=2)+'\n')
    x.publish(BRANCHES[machine], ['experiments/upstream-expansion',page.name],
              'upstream expansion '+machine+': '+state+' '+(active or ''),pushing)
    return value


def step(machine, name, pushing):
    status(machine,'training',name,pushing=pushing)
    log_path=RUN/(name+'.driver.log')
    with log_path.open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u','scripts/upstream_repro.py','--run-job',name],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT)
        tick=time.monotonic()
        while True:
            try:
                code=child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic()-tick > 1800:
                    status(machine,'training',name,pushing=pushing)
                    tick=time.monotonic()
        if code:
            raise RuntimeError(f'{name} failed with status {code}; preserve native partial outputs/logs')


def export(machine,name):
    u.validate_fresh()
    destination=TRACKED/machine
    destination.mkdir(parents=True,exist_ok=True)
    native=u.FRESH/'preds_new'/(name+'.probex.npy')
    receipt=u.read(u.BASE/'fresh-receipts'/(name+'.json'))
    if u.sha(native) != receipt['outputs']['preds_new/'+name+'.probex.npy']:
        raise ValueError('Native output differs from its receipt')
    y=np.load(u.FRESH/'data/probex/ratings.npy',allow_pickle=False)
    users=np.load(u.FRESH/'data/probex/user_idxs.npy',allow_pickle=False)
    prediction=np.load(native,allow_pickle=False)
    if prediction.shape!=y.shape or prediction.dtype!=np.dtype('float32') or not np.isfinite(prediction).all():
        raise ValueError('Invalid fresh expansion output')
    np.savez_compressed(destination/(name+'.npz'),prediction=prediction)
    np.savez_compressed(destination/'development.npz',labels=y,users=users)
    reference=u.REFERENCE/'preds_new'/(name+'.probex.npy')
    report=dict(job=name,source_commit=u.PIN,receipt=receipt,
                artifact_sha256=u.sha(destination/(name+'.npz')),prediction_bytes_sha256=x.array_sha(prediction),
                native_npy_sha256=u.sha(native),
                development_sha256=u.sha(destination/'development.npz'),
                fresh_rmse=x.metric(prediction,y),reference_available=reference.exists())
    if reference.exists():
        fixture=np.load(reference,allow_pickle=False)
        report.update(released_rmse=x.metric(fixture,y),prediction_rms_difference=float(np.sqrt(np.mean((prediction.astype(float)-fixture)**2))))
    u.write(destination/(name+'.json'),report)
    shutil.copyfile(u.BASE/'fresh-receipts'/(name+'.log'),destination/(name+'.log'))


def diagnostic(machine,pushing):
    u.validate_fresh()
    status(machine,'diagnostic','tsvdx4-60 support provenance',pushing=pushing)
    if not (u.BASE/'fresh-receipts/tsvdx4-60.json').exists():
        launch=u.BASE/'investigation/local-launch.json'
        if launch.exists():
            pid=u.read(launch)['pid']
            while not (u.BASE/'fresh-receipts/tsvdx4-60.json').exists():
                os.kill(pid,0)
                time.sleep(15)
        else:
            step(machine,'tsvdx4-60',pushing)
    helper=ROOT/'scripts/upstream_neighborhood_probe.rs'
    shutil.copyfile(helper,u.FRESH/'src/bin/upstream_neighborhood_probe.rs')
    env=u.environment()
    cargo=shutil.which('cargo',path=env['PATH'])
    if not cargo:
        raise RuntimeError('Cargo unavailable for diagnostic helper')
    with (RUN/'diagnostic.log').open('w') as log:
        subprocess.run([cargo,'build','--release','--locked','--features','blas','--bin','upstream_neighborhood_probe'],cwd=u.FRESH,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(u.BASE/'target/release/upstream_neighborhood_probe')],cwd=u.FRESH,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    y=np.load(u.FRESH/'data/probex/ratings.npy',allow_pickle=False)
    fixture=np.load(u.REFERENCE/'preds_new/tsvdx4-60__knns.probex.npy',allow_pickle=False).astype(float)
    records=[]
    for mode in ['same-recipe','support-with-qual']:
        native=u.FRESH/'investigation'/(mode+'.probex.npy')
        p=np.load(native,allow_pickle=False).astype(float)
        records.append(dict(mode=mode,rmse=x.metric(p,y),prediction_rms_difference=float(np.sqrt(np.mean((p-fixture)**2))),native_sha256=u.sha(native)))
    result=dict(machine=machine,source_commit=u.PIN,helper_sha256=u.sha(helper),qualifying_outcomes_zero=True,
                support_matrix_sha256=u.sha(u.FRESH/'investigation/bin_supp.fulltrain-plus-qual.npy'),
                records=records,conclusion='Diagnostic measurement only; support-provenance hypothesis requires prediction-level agreement, not merely lower RMSE')
    u.write(TRACKED/machine/'diagnostic.json',result)
    shutil.copyfile(RUN/'diagnostic.log',TRACKED/machine/'diagnostic.log')
    status(machine,'diagnostic-complete',pushing=pushing)


def collect_expansion():
    reference_labels=np.load(u.REFERENCE/'data/probex/ratings.npy',allow_pickle=False)
    predictions={}
    for name in x.JOBS:
        saved=x.LOCAL/'incoming'/(name+'.npz')
        frozen=u.read(x.REPORT/'fresh-only-blends.json')['input_artifact_sha256'][name]
        if u.sha(saved)!=frozen:raise ValueError('Frozen original-eight artifact changed')
        with np.load(saved,allow_pickle=False) as a:predictions[name]=a['prediction']
    users=None
    for machine in ['m1','m3']:
        for name in ASSIGNMENTS[machine]:
            report_path=TRACKED/machine/(name+'.json')
            if not report_path.exists():return None
            report=u.read(report_path)
            artifact=report_path.with_suffix('.npz')
            if report['source_commit']!=u.PIN or u.sha(artifact)!=report['artifact_sha256']:
                raise ValueError('Expansion artifact integrity failure')
            development=report_path.parent/'development.npz'
            if u.sha(development)!=report['development_sha256']:raise ValueError('Development artifact changed')
            with np.load(development,allow_pickle=False) as a:
                labels,group=a['labels'],a['users']
            if not np.array_equal(labels,reference_labels):raise ValueError('Development row order changed')
            if users is not None and not np.array_equal(users,group):raise ValueError('User identities differ across machines')
            users=group
            with np.load(artifact,allow_pickle=False) as a:value=a['prediction']
            if value.shape!=reference_labels.shape or value.dtype!=np.dtype('float32') or not np.isfinite(value).all() or x.array_sha(value)!=report['prediction_bytes_sha256']:
                raise ValueError('Malformed or changed expansion vector')
            if abs(x.metric(value,reference_labels)-report['fresh_rmse'])>1e-8:raise ValueError('Independent model rescore differs')
            if (report['receipt']['source_commit']!=u.PIN or report['receipt']['job']!=name
                    or report['receipt']['outputs'].get('preds_new/'+name+'.probex.npy') != report['native_npy_sha256']):
                raise ValueError('Missing native output receipt')
            predictions[name]=value
    return reference_labels,users,predictions


def watch(pushing):
    remote='refs/remotes/origin/research/upstream-reproduction'
    while True:
        fetch=x.git('fetch','origin','refs/heads/research/upstream-reproduction:'+remote,check=False)
        if fetch.returncode:raise RuntimeError(fetch.stderr)
        paths=x.git('ls-tree','-r','--name-only',remote,'--','experiments/upstream-expansion/m3').stdout.splitlines()
        for path in paths:
            destination=ROOT/path
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(x.worker_blob(remote,path))
        remote_status=TRACKED/'m3/status.json'
        if remote_status.exists() and u.read(remote_status)['state']=='failed':
            raise RuntimeError('M3 reported failure: '+str(u.read(remote_status).get('error')))
        data=collect_expansion()
        if data is not None:
            y,users,predictions=data
            folds=x.user_folds(users)
            records=[]
            for arm,columns in ARMS.items():
                values=x.crossfit(np.column_stack([predictions[n] for n in columns]),y,folds)
                output=RUN/(arm+'.f32');values.astype('<f4').tofile(output)
                records.append(dict(arm=arm,rmse=x.metric(values,y),fold_rmse=[x.metric(values[folds==f],y[folds==f]) for f in range(5)],prediction_sha256=u.sha(output)))
            base=records[0]
            for r in records:
                r['delta_rmse']=r['rmse']-base['rmse']
                r['fold_delta']=[b-a for a,b in zip(base['fold_rmse'],r['fold_rmse'])]
            if abs(base['rmse']-u.read(x.REPORT/'fresh-only-blends.json')['records'][1]['rmse'])>1e-9:
                raise ValueError('Original-eight control does not reproduce')
            u.write(TRACKED/'evaluation.json',dict(plan=PLAN,records=records,remote_commit=x.git('rev-parse',remote).stdout.strip()))
            status('m1','complete',pushing=pushing)
            return
        status('m1','waiting-for-m3',pushing=pushing)
        time.sleep(300)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--machine',choices=['m1','m3'],required=True)
    ap.add_argument('--publish',action='store_true')
    ap.add_argument('--plan-only',action='store_true')
    args=ap.parse_args()
    if args.plan_only:
        print(json.dumps(PLAN,indent=2));return
    x.branch_guard(BRANCHES[args.machine])
    RUN.mkdir(parents=True,exist_ok=True)
    plan_path=TRACKED/'plan.json'
    if not plan_path.exists() or u.read(plan_path)!=PLAN:raise ValueError('Committed expansion plan differs')
    for key in ['folds','fold_seed','ridge_lambda','input_clip','output_clip']:
        if PLAN[key]!=x.PLAN[key]:raise ValueError('Shared cross-fit settings differ: '+key)
    lock=RUN/'worker.lock'
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError('Expansion worker already active')
    with lock.open('x') as f:f.write(str(os.getpid()))
    inhibitor=None
    try:
        os.nice(10)
        if Path('/usr/bin/caffeinate').exists():inhibitor=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        diagnostic(args.machine,args.publish)
        for name in ASSIGNMENTS[args.machine]:
            step(args.machine,name,args.publish)
            export(args.machine,name)
            status(args.machine,'job-complete',name,pushing=args.publish)
        if args.machine=='m1':watch(args.publish)
        else:status('m3','complete',pushing=args.publish)
    except BaseException as error:
        traceback.print_exc()
        with contextlib.suppress(Exception):status(args.machine,'failed',error=repr(error),pushing=args.publish)
        raise
    finally:
        lock.unlink(missing_ok=True)
        if inhibitor:inhibitor.terminate()


if __name__=='__main__':main()
