#!/usr/bin/env python3
"""Conditional full tree154 refit with the unchanged, replayed full NN154 partner."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_item154_tree as development
import upstream_item154_stream_production as preceding
import upstream_item154_stream_bank as bank
import upstream_production as core
import upstream_own5_calibration as binding
u,n=core.u,core.n
RUN=u.ROOT/'artifacts/upstream-item154-tree-production-recovered'
REPORT=u.ROOT/'experiments/upstream-item154-tree-production-recovered'
REFERENCE=u.ROOT/'experiments/upstream-rbm50-tree-item154-mps/verification.json'
SUPERVISOR=u.ROOT/'experiments/upstream-item154-tree-supervisor/completion.json'

def code_hashes():
    return {**development.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in
        [REPORT/'recovery.json',u.ROOT/'experiments/upstream-item154-tree-production/declaration.json',u.ROOT/'experiments/upstream-item154-tree-production/failure.json',u.ROOT/'scripts/upstream_item154_tree_production.py',u.ROOT/'scripts/verify_upstream_item154_tree_production.py',u.ROOT/'scripts/test_item154_tree_stream_fixture.py',Path(__file__),Path(core.__file__),Path(bank.__file__),Path(binding.__file__),
         u.ROOT/'scripts/verify_upstream_item154_tree_production_recovered.py',u.ROOT/'scripts/test_item154_tree_stream_fixture_recovered.py']}}

def configure():
    development.configure();n.RUN,n.REPORT=RUN,REPORT
    core.RUN,core.REPORT,core.bank,core.code_hashes,core.infer=RUN,REPORT,bank,code_hashes,infer

def infer(family,plan,root):
    if family!='tree':raise ValueError('Only the new full tree is fitted')
    import lightgbm as lgb
    w=np.load(RUN/'ridge.npy')
    if u.sha(RUN/'ridge.npy')!=plan['ridge_sha256']:raise ValueError('Original full ridge changed')
    models=[None if not count else lgb.Booster(model_file=str(root/f'tree-{seed}.txt'))
        for seed,count in zip(plan['seeds'],plan['iterations']['tree'])]
    out=np.empty(plan['qualifying_rows'],dtype='<f4')
    for lo in range(0,len(out),core.CHUNK):
        hi=min(lo+core.CHUNK,len(out));z,f=bank.qual_chunk(lo,hi);base=np.clip(z@w,1.05,4.95)
        x=np.column_stack([f,base]).astype(np.float32);correction=np.zeros(hi-lo)
        for model in models:
            if model is not None:correction+=model.predict(x,num_threads=n.CONFIG['threads'])/len(models)
        out[lo:hi]=np.clip(base+correction,1.05,4.95).astype('<f4')
    return out

def selection():
    development.configure();development.n.verify_prepared()
    finished=u.read(SUPERVISOR);evaluation=u.read(development.REPORT/'evaluation.json')
    proof=u.read(development.REPORT/'independent-verification.json');replay=u.read(development.REPORT/'tree-replay.json')
    if (finished['state']!='complete_original_tree_fit_actual_replay_and_independent_scoring' or
        finished['evaluation_sha256']!=u.sha(development.REPORT/'evaluation.json') or
        finished['verification_sha256']!=u.sha(development.REPORT/'independent-verification.json') or
        finished['replay_sha256']!=u.sha(development.REPORT/'tree-replay.json') or
        proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(development.REPORT/'evaluation.json') or
        not proof['fixed_average_bitwise_equal'] or [r['fold'] for r in replay['records']]!=list(range(5)) or
        not all(r['bitwise_equal'] for r in replay['records'])):raise ValueError('Complete independent development and all actual tree replays required')
    row=next(r for r in evaluation['records'] if r['arm']=='fixed-half-tree-nn')
    checked=next(r for r in proof['records'] if r['arm']=='fixed-half-tree-nn')
    if row['prediction_sha256']!=checked['prediction_sha256'] or abs(row['rmse']-checked['independent_rmse'])>1e-12:raise ValueError('Independent complete score differs')
    vectors={}
    for arm in ['tree','nn','fixed-half-tree-nn']:
        path=development.RUN/(arm+'.f32');record=next(r for r in evaluation['records'] if r['arm']==arm)
        if u.sha(path)!=record['prediction_sha256']:raise ValueError('Original complete development vector changed')
        vectors[arm]=np.fromfile(path,dtype='<f4')
        if vectors[arm].shape!=(1128164,) or not np.isfinite(vectors[arm]).all():raise ValueError('Complete finite known vectors required')
    expected=((vectors['tree'].astype(np.float64)+vectors['nn'].astype(np.float64))/2).astype('<f4')
    if not np.array_equal(expected.view(np.uint32),vectors['fixed-half-tree-nn'].view(np.uint32)):raise ValueError('Exact known fixed half mean differs')
    y=np.load(development.RUN/'labels.npy')
    if not np.array_equal(y,np.load(u.FRESH/'data/probex/ratings.npy')):raise ValueError('Canonical complete known labels differ')
    score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(expected,y))/len(y))
    if abs(score-row['rmse'])>1e-12:raise ValueError('Separate complete known score differs')
    return row

def prepare(row):
    sources={}
    def bind(p,digest=None):
        name=str(p.relative_to(u.ROOT));digest=u.sha(p) if digest is None else digest
        if name in sources and sources[name]!=digest:raise ValueError('Conflicting frozen source: '+name)
        sources[name]=digest
    m=u.read(development.REPORT/'manifest.json');p=u.read(preceding.REPORT/'plan.json')
    receipt=u.read(preceding.REPORT/'nn.json');replay=u.read(preceding.REPORT/'nn-replay.json')
    if (p['config']!=development.CONFIG or p['qualifying_labels_read'] is not False or
        receipt['plan_sha256']!=u.sha(preceding.REPORT/'plan.json') or not replay['bitwise_equal'] or
        replay['plan_sha256']!=receipt['plan_sha256'] or replay['family_receipt_sha256']!=u.sha(preceding.REPORT/'nn.json') or
        replay['prediction_sha256']!=receipt['outputs']['qualifying.f32']):raise ValueError('Original actual full NN154 recipe/replay required')
    bm=u.read(bank.REPORT/'manifest.json');bp=u.read(bank.REPORT/'column-verification.json')
    if (bp['state']!='complete' or bp['manifest_sha256']!=u.sha(bank.REPORT/'manifest.json') or
        bp['verifier_sha256']!=u.sha(Path(bank.__file__)) or not bp['all_streamed_qualifying_columns_compared_bitwise'] or
        not bp['added_known_columns_bitwise_match_development'] or bm['design_columns']!=183 or bm['nonlinear_features']!=397):raise ValueError('Original verified streaming item154 bank required')
    for collection in [m['code_sha256'],m['source_manifests_sha256'],p['source_sha256'],p['code_sha256'],bm['source_sha256'],bm['code_sha256']]:
        for name,digest in collection.items():
            if 'recovered-audit/' not in name and 'qual_ratings' not in name:bind(u.ROOT/name,digest)
    for name,digest in bm['files'].items():bind(bank.RUN/name,digest)
    counts=[]
    for fold in range(5):
        fp=development.REPORT/'tree'/f'{fold}.json';r=u.read(fp)
        if r['fold']!=fold or r['family']!='tree' or r['manifest_sha256']!=u.sha(development.REPORT/'manifest.json'):raise ValueError('Original actual nested tree budget required')
        counts.append(r['chosen_iterations']);bind(fp)
        for name,digest in r['outputs'].items():bind(development.RUN/'tree'/str(fold)/name,digest)
    for name,digest in receipt['outputs'].items():bind(preceding.RUN/'nn'/name,digest)
    for q in [SUPERVISOR,REFERENCE,REPORT/'declaration.json',REPORT/'selection.json',u.ROOT/'experiments/upstream-item154-tree-stream-fixture-recovered/verification.json',
        development.RUN/'tree.f32',development.RUN/'nn.f32',development.RUN/'fixed-half-tree-nn.f32',
        development.REPORT/'manifest.json',development.REPORT/'evaluation.json',development.REPORT/'independent-verification.json',development.REPORT/'tree-replay.json',
        preceding.REPORT/'plan.json',preceding.REPORT/'nn.json',preceding.REPORT/'nn-replay.json',preceding.RUN/'ridge.npy',bank.REPORT/'manifest.json',bank.REPORT/'column-verification.json']:
        bind(q)
    binding.verify_bindings(sources)
    if RUN.exists() or (REPORT/'plan.json').exists():raise ValueError('Existing full-tree preparation preserved')
    RUN.mkdir();u.link_or_copy(preceding.RUN/'ridge.npy',RUN/'ridge.npy')
    configure()
    plan=dict(state='full_tree154_known_selected_before_full_fit',created=binding.stamp(),config=n.CONFIG,versions=n.versions(),
        code_sha256=code_hashes(),source_sha256=sources,iterations={'tree':core.budgets(counts,n.CONFIG['tree_rounds'])},seeds=n.CONFIG['seeds'],
        qualifying_rows=2817131,qualifying_labels_read=False,ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,
        development_rmse=row['rmse'],weights=[.5,.5],new_family_schema=dict(prediction_columns=107,learned_support_columns=154,features_plus_ridge=398),
        reused_nn_origin=str(preceding.REPORT.relative_to(u.ROOT)),reused_ridge_origin=str((preceding.RUN/'ridge.npy').relative_to(u.ROOT)),
        rule='Only if complete independently verified five-fold fixed mean beats the declared current CV best. Median of five actual per-seed tree budgets; same full-known183-column ridge, original397 tree features plus ridge. Reuse unchanged actually replayed full NN154. Actual new tree saved-model full-query replay; fixed half mean frozen before separate recovered audit. Original paired-parent source differences remain disclosed.')
    u.write(REPORT/'plan.json',plan)

def guarded(command,name):
    with (REPORT/name).open('x') as log:
        child=subprocess.Popen(['nice','-n','10']+command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=development.template.group_rss(child.pid);other=development.template.parent.resources.other_rss({child.pid});free=shutil.disk_usage(u.ROOT).free
                high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,observed=binding.stamp()))
                if high>=3 or free<8*2**30:raise RuntimeError('Original24GiB/8GiB guards reached; partial full-tree artifacts preserved')
                time.sleep(5)
        except BaseException:development.template.stop(child);raise
        if child.returncode:raise RuntimeError('Full-tree subprocess failed; original artifacts preserved')

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    binding.verify_bindings(u.read(REPORT/'recovery.json')['source_sha256'])
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'declaration.json').exists():raise ValueError('Existing full-tree queue declaration preserved')
        fixture=u.ROOT/'experiments/upstream-item154-tree-stream-fixture-recovered/verification.json'
        f=u.read(fixture)
        if not f['actual_saved_tree_streaming_bitwise_equal'] or f['code_sha256']!=code_hashes():raise ValueError('Actual saved-tree streaming fixture required')
        threshold=u.read(REFERENCE)['rmse']
        u.write(REPORT/'declaration.json',dict(created=binding.stamp(),code_sha256=code_hashes(),reference_rmse=threshold,reference_sha256=u.sha(REFERENCE),fixture_sha256=u.sha(fixture),
            rule='Separate conditional production follow-up. Wait for all five independently verified original tree folds and actual model replays. Fit only if their predeclared half mean improves the reference known CV score. Full original model/data/budgets and old NN unchanged. Reserve8GiB below24GiB total; no GPU lock held by this CPU fit. Freeze before recovered-label access.',qualifying_labels_read=False))
        u.write(REPORT/'status.json',dict(state='waiting_for_complete_independent_development',observed=binding.stamp(),initial_admission_hold=True))
        os.kill(os.getpid(),signal.SIGSTOP)
        while not SUPERVISOR.exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_complete_independent_development',observed=binding.stamp()));time.sleep(20)
        declaration=u.read(REPORT/'declaration.json')
        if declaration['code_sha256']!=code_hashes() or declaration['reference_sha256']!=u.sha(REFERENCE):raise ValueError('Original production declaration changed')
        row=selection();selected=row['rmse']<threshold
        u.write(REPORT/'selection.json',dict(state='selected_on_complete_known_cv' if selected else 'complete_not_selected_no_full_fit',development=row,reference_rmse=threshold,selected=selected,qualifying_labels_read=False))
        if not selected:return
        good=0;previous=development.resources.memory.memory()
        while good<2:
            current=development.resources.memory.memory();other=development.template.parent.resources.other_rss({os.getpgrp()});free=shutil.disk_usage(u.ROOT).free
            good=good+1 if other<16*2**30 and current['compressor_bytes']<5*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>10*2**30 else 0
            u.write(REPORT/'status.json',dict(state='waiting_for_full_tree_memory',other_rss_bytes=other,memory=current,disk_free_bytes=free,good_samples=good,observed=binding.stamp()))
            if good<2:previous=current;time.sleep(20)
        prepare(row)
        for replay in [False,True]:guarded([sys.executable,'-u',str(Path(__file__)),'--worker']+(['--replay'] if replay else []),'tree-replay.log' if replay else 'tree.log')
        configure();core.verify()
        a=np.fromfile(RUN/'tree/qualifying.f32',dtype='<f4').astype(np.float64);b=np.fromfile(preceding.RUN/'nn/qualifying.f32',dtype='<f4').astype(np.float64)
        if a.shape!=(2817131,) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Complete finite family vectors required')
        ((a+b)/2).astype('<f4').tofile(REPORT/'qualifying-final.f32')
        u.write(REPORT/'report.json',dict(state='complete_frozen_tree154_nn154_half_mean',observed=binding.stamp(),plan_sha256=u.sha(REPORT/'plan.json'),
            frozen_prediction_sha256=u.sha(REPORT/'qualifying-final.f32'),new_tree_replay_sha256=u.sha(REPORT/'tree-replay.json'),
            reused_nn_replay_sha256=u.sha(preceding.REPORT/'nn-replay.json'),qualifying_labels_read=False,weights=[.5,.5]))
        subprocess.run([sys.executable,str(u.ROOT/'scripts/verify_upstream_item154_tree_production_recovered.py')],cwd=u.ROOT,check=True)
        u.write(REPORT/'status.json',dict(state='complete_full_tree_replay_frozen_mean_and_separate_audit',observed=binding.stamp()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.worker:configure();core.worker('tree',args.replay)
    else:
        try:run()
        except BaseException as error:
            u.write(REPORT/'failure.json',dict(state='incomplete_preserved',error=repr(error),observed=binding.stamp()));raise
