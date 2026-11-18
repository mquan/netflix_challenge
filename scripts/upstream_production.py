#!/usr/bin/env python3
"""Refit the fixed fresh blend, replay it, freeze qual, then run one label audit."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import upstream_repro as u
import upstream_nonlinear as n
import upstream_production_bank as bank
import upstream_voting_blend as voting
import verify_upstream_nonlinear as replay

RUN=u.ROOT/'artifacts/upstream-production'
REPORT=u.ROOT/'experiments/upstream-production'
CHUNK=65536


def budgets(records,maximum):
    a=np.asarray(records)
    if a.shape!=(5,3) or not np.issubdtype(a.dtype,np.integer) or np.any((a<0)|(a>maximum)):
        raise ValueError('Invalid nested iteration budgets')
    return np.median(a,axis=0).astype(int).tolist()


def code_hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),
        u.ROOT/'scripts/upstream_production_bank.py',u.ROOT/'scripts/upstream_nonlinear.py',
        u.ROOT/'scripts/upstream_snapshot_blend.py',u.ROOT/'scripts/verify_upstream_nonlinear.py']}


def verify():
    plan=u.read(REPORT/'plan.json')
    if plan['code_sha256']!=code_hashes() or plan['config']!=n.CONFIG or plan['versions']!=n.versions():
        raise ValueError('Frozen production recipe changed')
    for name,digest in plan['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen production source changed: '+name)
    if u.sha(RUN/'ridge.npy')!=plan['ridge_sha256']:raise ValueError('Final ridge changed')
    return plan


def prepare():
    if (REPORT/'plan.json').exists():return verify()
    if RUN.exists():raise ValueError('Partial production preparation preserved')
    data=bank.build();voting.configure();n.verify_prepared()
    evaluation=voting.REPORT/'evaluation.json'
    if u.read(voting.REPORT/'independent-verification.json')['evaluation_sha256']!=u.sha(evaluation):
        raise ValueError('Voting experiment not independently verified')
    sources={str((bank.REPORT/'manifest.json').relative_to(u.ROOT)):u.sha(bank.REPORT/'manifest.json'),
             str(evaluation.relative_to(u.ROOT)):u.sha(evaluation),**data['source_sha256']}
    for name,digest in data['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    choices={}
    for family in ['tree','nn']:
        records=[]
        for fold in range(5):
            path=voting.REPORT/family/(str(fold)+'.json');receipt=u.read(path)
            if receipt['manifest_sha256']!=u.sha(voting.REPORT/'manifest.json'):raise ValueError('Budget source changed')
            sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(receipt['chosen_iterations'])
        choices[family]=budgets(records,n.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    gram=np.zeros((z.shape[1],z.shape[1]));rhs=np.zeros(z.shape[1])
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    w=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    np.save(RUN/'ridge.npy',w)
    plan=dict(state='frozen_before_final_fitting',created=dt.datetime.now(dt.timezone.utc).isoformat(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        iteration_rule='Per-seed median across five nested user-fold budgets, frozen before qualifying labels',
        iterations=choices,qualifying_labels_read=False,qualifying_rows=data['qualifying_rows'],
        prediction_chunk_rows=CHUNK,ridge_sha256=u.sha(RUN/'ridge.npy'),seeds=n.CONFIG['seeds'],
        final_rule='Fit each family on all known ProbeX rows, average its three seed corrections, clip family predictions, then fixed equal mean of float32 family predictions.')
    u.write(REPORT/'plan.json',plan);return plan


def infer(family,plan,root):
    w=np.load(RUN/'ridge.npy')
    if u.sha(RUN/'ridge.npy')!=plan['ridge_sha256']:raise ValueError('Final ridge changed')
    models=[]
    for seed,count in zip(plan['seeds'],plan['iterations'][family]):
        if not count:models.append(None);continue
        if family=='tree':
            import lightgbm as lgb
            models.append(lgb.Booster(model_file=str(root/f'tree-{seed}.txt')))
        else:models.append(replay.neural_model(root/f'nn-{seed}.pt'))
    scaler=None
    if family=='nn':
        with np.load(root/'scaler.npz') as archive:scaler={key:archive[key] for key in archive.files}
    z=np.load(bank.RUN/'qual-design.npy',mmap_mode='r');f=np.load(bank.RUN/'qual-features.npy',mmap_mode='r')
    prediction=np.empty(len(z),dtype='<f4')
    for lo in range(0,len(z),CHUNK):
        hi=min(lo+CHUNK,len(z));base=np.clip(z[lo:hi]@w,1.05,4.95)
        features=np.column_stack([f[lo:hi],base]).astype(np.float32)
        if scaler is not None:features,_=n.scaled(features,None,scaler)
        correction=np.zeros(hi-lo,dtype=np.float64)
        for model in models:
            if model is not None:
                part=model.predict(features,num_threads=n.CONFIG['threads']) if family=='tree' else n.nn_predict(model,features)
                correction+=part/len(models)
        prediction[lo:hi]=np.clip(base+correction,1.05,4.95).astype('<f4')
    return prediction


def worker(family,replay_only=False):
    plan=verify();root=RUN/family;receipt=REPORT/(family+'.json')
    if replay_only:
        record=u.read(receipt)
        if record['plan_sha256']!=u.sha(REPORT/'plan.json'):raise ValueError('Family plan binding changed')
        for name,digest in record['outputs'].items():
            if u.sha(root/name)!=digest:raise ValueError('Final family artifact changed')
        pred=infer(family,plan,root);stored=np.fromfile(root/'qualifying.f32',dtype='<f4')
        if not np.array_equal(pred.view(np.uint32),stored.view(np.uint32)):raise ValueError('Final model replay differs')
        u.write(REPORT/(family+'-replay.json'),dict(bitwise_equal=True,plan_sha256=u.sha(REPORT/'plan.json'),
            family_receipt_sha256=u.sha(receipt),prediction_sha256=u.sha(root/'qualifying.f32')));return
    if receipt.exists():
        record=u.read(receipt)
        if record['plan_sha256']!=u.sha(REPORT/'plan.json'):raise ValueError('Family plan binding changed')
        for name,digest in record['outputs'].items():
            if u.sha(root/name)!=digest:raise ValueError('Final family artifact changed')
        return
    if root.exists():raise ValueError('Partial final family preserved')
    root.mkdir();w=np.load(RUN/'ridge.npy');z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r')
    y=np.load(bank.RUN/'labels.npy').astype(np.float64);f=np.load(bank.RUN/'probex-features.npy',mmap_mode='r')
    base=np.clip(z@w,1.05,4.95);features=np.column_stack([f,base]).astype(np.float32)
    if family=='nn':
        features,scaler=n.scaled(features,np.ones(len(y),dtype=bool));np.savez(root/'scaler.npz',**scaler)
    for seed,count in zip(plan['seeds'],plan['iterations'][family]):
        if not count:continue
        if family=='tree':
            import lightgbm as lgb
            params=dict(objective='regression',metric='l2',learning_rate=n.CONFIG['tree_lr'],num_leaves=n.CONFIG['tree_leaves'],
                min_data_in_leaf=n.CONFIG['tree_min_leaf'],feature_fraction=.9,bagging_fraction=.8,bagging_freq=5,
                seed=seed,num_threads=n.CONFIG['threads'],verbosity=-1,deterministic=True,force_col_wise=True)
            model=lgb.train(params,lgb.Dataset(features,label=y-base),num_boost_round=count)
            model.save_model(str(root/f'tree-{seed}.txt'))
        else:
            import torch
            model,_=n.train_nn(features,y-base,n.CONFIG,seed,count)
            torch.save(dict(state=model.state_dict(),config=n.CONFIG,feature_width=features.shape[1],seed=seed,epochs=count),root/f'nn-{seed}.pt')
        del model
        n.event(stage='full_probe_refit_seed',family=family,seed=seed,iterations=count)
    del features,f,z,y,base
    infer(family,plan,root).tofile(root/'qualifying.f32')
    u.write(receipt,dict(plan_sha256=u.sha(REPORT/'plan.json'),outputs={p.name:u.sha(p) for p in root.iterdir() if p.is_file()},
                         qualifying_labels_read=False))


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (bank.REPORT/'manifest.json').exists() or not (voting.REPORT/'independent-verification.json').exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_paired_bank_and_nested_budgets'));time.sleep(60)
        from upstream_qualifying_backfill import other_training_rss
        while other_training_rss()>18*2**30:
            u.write(REPORT/'status.json',dict(state='waiting_for_training_memory'));time.sleep(60)
        prepare()
        for family in ['tree','nn']:
            u.write(REPORT/'status.json',dict(state='full_probe_refitting',family=family))
            for extra in [[],['--replay']]:
                with (REPORT/(family+'.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__)),'--family',family]+extra,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        u.write(REPORT/'status.json',dict(state='replaying' if extra else 'full_probe_refitting',family=family,pid=child.pid))
                        child.wait()
                    except BaseException:
                        from upstream_full_reproduce import stop
                        stop(child);raise
                    if child.returncode:raise RuntimeError('Final worker failed; inspect preserved family output')
        a=np.fromfile(RUN/'tree/qualifying.f32',dtype='<f4').astype(np.float64)
        b=np.fromfile(RUN/'nn/qualifying.f32',dtype='<f4').astype(np.float64)
        if a.shape!=(2817131,) or b.shape!=a.shape:raise ValueError('Final prediction count differs')
        final=((a+b)/2).astype('<f4')
        if not np.isfinite(final).all() or np.any((final<1.05)|(final>4.95)):raise ValueError('Invalid final predictions')
        final.tofile(REPORT/'qualifying-final.f32')
        report=dict(state='complete',qualifying_labels_read=False,qualifying_rows=len(final),
            frozen_prediction_sha256=u.sha(REPORT/'qualifying-final.f32'),plan_sha256=u.sha(REPORT/'plan.json'),
            replay_receipts={family:u.sha(REPORT/(family+'-replay.json')) for family in ['tree','nn']},
            scope='Fresh locally trained bank and known-ProbeX-fitted weights; original-capacity mapped temporal model disclosed. Frozen before recovered qualifying audit.')
        u.write(REPORT/'report.json',report)
        audit=REPORT/'recovered-audit'
        if not (audit/'report.json').exists():
            subprocess.run([sys.executable,str(u.ROOT/'scripts/audit_qualifying.py'),'--model-dir',str(REPORT),
                            '--output-dir',str(audit)],check=True)
        result=u.read(audit/'report.json')
        if result['source']['prediction_sha256']!=report['frozen_prediction_sha256']:raise ValueError('Audit binds another prediction')
        # Independent compensated summation after the frozen, source-verified audit.
        import gzip,math
        if u.sha(audit/'qual_ratings.csv.gz')!=result['label_sha256']:raise ValueError('Audit labels changed')
        with gzip.open(audit/'qual_ratings.csv.gz','rt') as stream:
            labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
        checked={}
        for value,name in [(0,'quiz'),(1,'test')]:
            mask=labels[:,1]==value
            score=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(final[mask],labels[mask,0]))/int(mask.sum()))
            if abs(score-result[name]['rmse'])>1e-12:raise ValueError('Independent qualifying score differs')
            checked[name]=dict(rows=int(mask.sum()),rmse=score,below_target=score<.85725)
        u.write(REPORT/'independent-qualifying-verification.json',dict(groups=checked,
            prediction_sha256=report['frozen_prediction_sha256'],audit_report_sha256=u.sha(audit/'report.json'),
            both_groups_below_target=all(r['below_target'] for r in checked.values()),model_fitting_performed=False,
            limitation='Community-recovered outcomes, adaptive local audit; not official Netflix scoring.'))
        u.write(REPORT/'status.json',dict(state='audit_complete',both_groups_below_target=result['both_source_groups_below_target'],
            quiz_rmse=result['quiz']['rmse'],test_rmse=result['test']['rmse'],goal_requires_independent_review=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--family',choices=['tree','nn']);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.family:worker(args.family,args.replay)
    else:run()
