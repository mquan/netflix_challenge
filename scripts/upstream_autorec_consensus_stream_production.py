#!/usr/bin/env python3
"""Fit own109 NN and freeze the known-selected rounded ensemble consensus."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_production as core
import upstream_autorec109_mps_context as neural
import upstream_autorec109_stream_bank as bank
import upstream_autorec109_stream_infer as streaming
import upstream_latent90_local as resources

u,n=core.u,core.n
RUN=u.ROOT/'artifacts/upstream-autorec-consensus-stream-production'
REPORT=u.ROOT/'experiments/upstream-autorec-consensus-stream-production'
DEVELOPMENT=u.ROOT/'experiments/upstream-autorec-consensus'
OLD_RUN=u.ROOT/'artifacts/upstream-rbm50-tree-latent90-production'
OLD_REPORT=u.ROOT/'experiments/upstream-rbm50-tree-latent90-production'

def code_hashes():
    paths=[Path(__file__),Path(core.__file__),Path(bank.__file__),Path(resources.__file__),Path(streaming.__file__),
        u.ROOT/'scripts/upstream_autorec109_stream_bank.py',u.ROOT/'scripts/verify_upstream_autorec_consensus_stream_production.py',
        u.ROOT/'scripts/upstream_autorec_consensus.py',u.ROOT/'scripts/verify_upstream_autorec_consensus.py',u.ROOT/'scripts/audit_qualifying.py']
    return {**neural.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}

def configure():
    neural.configure();n.RUN,n.REPORT=RUN,REPORT
    core.RUN,core.REPORT,core.bank,core.code_hashes=RUN,REPORT,bank,code_hashes
    core.replay.neural_model=neural.neural_model;core.infer=streaming.infer

def declaration():
    vp=DEVELOPMENT/'verification.json';cp=DEVELOPMENT/'independent-verification.json'
    proof=u.read(vp);check=u.read(cp)
    if (proof['state']!='complete_fixed_post_study_consensus' or not proof['improves_previous'] or
            proof['qualifying_labels_read'] is not False or proof['weights']!=[.5,.5] or
            check['state']!='complete_independent_consensus_and_score' or check['verification_sha256']!=u.sha(vp) or
            not check['stored_consensus_bitwise_equal'] or abs(check['rmse']-proof['rmse'])>1e-12):
        raise ValueError('Independent improved known-only consensus required')
    sources=dict(proof['source_sha256'])
    for p in [vp,cp]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    records=[]
    for fold in range(5):
        path=neural.REPORT/'nn'/f'{fold}.json';r=u.read(path)
        if r['manifest_sha256']!=u.sha(neural.REPORT/'manifest.json') or r['fold']!=fold or r['family']!='nn':
            raise ValueError('Actual original neural budget receipt required')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path);records.append(r['chosen_iterations'])
    counts=core.budgets(records,neural.CONFIG['nn_epochs'])
    return proof,counts,sources

def reused_sources():
    plan=u.read(OLD_REPORT/'plan.json');frozen=u.read(OLD_REPORT/'report.json')
    if (plan['qualifying_labels_read'] is not False or frozen['state']!='complete' or
            frozen['qualifying_labels_read'] is not False or frozen['plan_sha256']!=u.sha(OLD_REPORT/'plan.json')):
        raise ValueError('Original frozen qualifying source required')
    sources={**plan['source_sha256'],**plan['code_sha256']}
    arrays={}
    for family in ['tree','nn']:
        path=OLD_REPORT/(family+'.json');r=u.read(path);rp=OLD_REPORT/(family+'-replay.json');replay=u.read(rp)
        if (r['plan_sha256']!=frozen['plan_sha256'] or not replay['bitwise_equal'] or
                replay['family_receipt_sha256']!=u.sha(path) or frozen['replay_receipts'][family]!=u.sha(rp)):
            raise ValueError('Original actual full-family replay differs')
        for name,digest in r['outputs'].items():sources[str((OLD_RUN/family/name).relative_to(u.ROOT))]=digest
        for p in [path,rp]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
        arrays[family]=np.fromfile(OLD_RUN/family/'qualifying.f32',dtype='<f4').astype(np.float64)
    old=np.fromfile(OLD_REPORT/'qualifying-final.f32',dtype='<f4')
    if (old.shape!=(2817131,) or not np.array_equal(old.view(np.uint32),((arrays['tree']+arrays['nn'])/2).astype('<f4').view(np.uint32)) or
            u.sha(OLD_REPORT/'qualifying-final.f32')!=frozen['frozen_prediction_sha256']):raise ValueError('Original rounded mean differs')
    for p in [OLD_REPORT/'plan.json',OLD_REPORT/'report.json',OLD_REPORT/'qualifying-final.f32']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    return sources

def prepare():
    configure()
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial production artifacts preserved')
    proof,counts,sources=declaration();sources.update(reused_sources())
    m=u.read(bank.REPORT/'manifest.json');p=u.read(bank.REPORT/'column-verification.json')
    if (p['state']!='complete' or p['manifest_sha256']!=u.sha(bank.REPORT/'manifest.json') or
            p['verifier_sha256']!=u.sha(u.ROOT/'scripts/upstream_autorec109_stream_bank.py') or
            not p['all_retained_columns_checked'] or not p['all_two_raw_transforms_checked'] or
            not p['added_known_columns_bitwise_match_development'] or m['design_columns']!=185 or m['nonlinear_features']!=335):
        raise ValueError('Complete exact paired109 cache required')
    for collection in [m['source_sha256'],m['code_sha256']]:
        for name,digest in collection.items():
            if name in sources and sources[name]!=digest:raise ValueError('Conflicting source lineage')
            sources[name]=digest
    for name,digest in m['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    for path in [bank.REPORT/'manifest.json',bank.REPORT/'column-verification.json']:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    z=np.load(bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(bank.RUN/'labels.npy').astype(np.float64)
    gram=np.zeros((185,185));rhs=np.zeros(185)
    for lo in range(0,len(y),8192):
        a=z[lo:lo+8192];gram+=a.T@a;rhs+=a.T@y[lo:lo+8192]
    weights=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    RUN.mkdir();np.save(RUN/'ridge.npy',weights)
    plan=dict(state='fixed_consensus_declared_before_new_full_fit_and_qualifying_outcomes',created=resources.stamp(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,iterations={'nn':counts},
        seeds=n.CONFIG['seeds'],qualifying_rows=2817131,qualifying_labels_read=False,ridge_sha256=u.sha(RUN/'ridge.npy'),
        prediction_chunk_rows=core.CHUNK,development_rmse=proof['rmse'],weights=[.5,.5],
        new_family_schema=dict(prediction_columns=109,learned_support_columns=90,features_plus_ridge=336),
        iteration_rule='Per-seed medians of five actual own109 nested user-fold budget receipts.',
        final_rule='Reuse actual replayed full tree50 and NN90. Fit/replay full own109 NN. Round each original family mean to float32, then average those two means in float64 and store float32, exactly as in development. Freeze before recovered-label access.',
        reused_original_full_models=True,raw_full50_completion_claimed=False,
        parent_realization_note='Retain previously disclosed paired90 parent source differences. Newly added raw5/10 known values match development exactly. No public numerical features or qualifying-based fitting/selection.')
    u.write(REPORT/'plan.json',plan);return plan

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof,counts,_=declaration()
        u.write(REPORT/'declaration.json',dict(created=resources.stamp(),development_rmse=proof['rmse'],iterations={'nn':counts},
            weights=[.5,.5],qualifying_labels_read=False,development_verification_sha256=u.sha(DEVELOPMENT/'verification.json')))
        while not (bank.REPORT/'column-verification.json').exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_verified_paired109'));time.sleep(15)
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            previous=resources.memory.memory();good=0
            while good<2:
                current=resources.memory.memory();workers=resources.footprints()
                good=good+1 if (sum(r['bytes'] for r in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20) else 0
                u.write(REPORT/'status.json',dict(state='waiting_for_full_fit_memory',observed=resources.stamp(),good_samples=good,memory=current))
                if good<2:previous=current;time.sleep(30)
            prepare()
            for replay in [False,True]:
                u.write(REPORT/'status.json',dict(state='actual_new_nn_saved_model_replay' if replay else 'full_known_new_nn_fitting'))
                with (REPORT/'nn.log').open('a') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker']+(['--replay'] if replay else []),cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            configure();core.verify();reused_sources()
            receipt=u.read(REPORT/'nn.json');replay=u.read(REPORT/'nn-replay.json')
            if not replay['bitwise_equal'] or replay['family_receipt_sha256']!=u.sha(REPORT/'nn.json'):raise ValueError('Actual new NN replay required')
            newpath=RUN/'nn/qualifying.f32'
            if u.sha(newpath)!=receipt['outputs']['qualifying.f32']:raise ValueError('New NN vector changed')
            new=np.fromfile(newpath,dtype='<f4').astype(np.float64)
            tree=np.fromfile(OLD_RUN/'tree/qualifying.f32',dtype='<f4').astype(np.float64)
            old=np.fromfile(OLD_REPORT/'qualifying-final.f32',dtype='<f4').astype(np.float64)
            if new.shape!=old.shape or new.shape!=(2817131,):raise ValueError('Full query count differs')
            second=((tree+new)/2).astype('<f4');final=((old+second.astype(np.float64))/2).astype('<f4')
            final.tofile(REPORT/'qualifying-final.f32')
            u.write(REPORT/'report.json',dict(state='complete',qualifying_labels_read=False,qualifying_rows=len(final),
                frozen_prediction_sha256=u.sha(REPORT/'qualifying-final.f32'),plan_sha256=u.sha(REPORT/'plan.json'),
                new_nn_replay_sha256=u.sha(REPORT/'nn-replay.json'),scope='Exact known-selected rounded consensus; original actual replayed full models reused, new full NN actual replay complete. Frozen before this recovered audit.'))
            subprocess.run([sys.executable,str(u.ROOT/'scripts/audit_qualifying.py'),'--model-dir',str(REPORT),'--output-dir',str(REPORT/'recovered-audit')],cwd=u.ROOT,check=True)
            import verify_upstream_autorec_consensus_stream_production as verifier
            verifier.verify()
            u.write(REPORT/'status.json',dict(state='complete_frozen_consensus_audit_and_independent_verification',observed=resources.stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();configure()
    if a.worker:core.worker('nn',a.replay)
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
