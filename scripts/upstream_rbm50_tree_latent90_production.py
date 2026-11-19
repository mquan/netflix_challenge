#!/usr/bin/env python3
"""Full-known CPU tree50 and MPS NN90; fixed mean declared from known CV only."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_production as core
import upstream_rbm50_tree as tree
import upstream_latent90_mps_context as neural
import upstream_rbm50_bank as tree_bank
import upstream_latent90_bank as nn_bank
import upstream_latent90_local as resources

u,n=core.u,core.n
RUN=u.ROOT/'artifacts/upstream-rbm50-tree-latent90-production'
REPORT=u.ROOT/'experiments/upstream-rbm50-tree-latent90-production'
DEVELOPMENT=u.ROOT/'experiments/upstream-rbm50-tree-latent90-mps'

def code_hashes():
    paths=[Path(__file__),Path(core.__file__),Path(tree_bank.__file__),Path(nn_bank.__file__),
        Path(resources.__file__),u.ROOT/'scripts/verify_upstream_frozen_mean.py',u.ROOT/'scripts/audit_qualifying.py',
        u.ROOT/'scripts/verify_upstream_latent90_bank.py',u.ROOT/'scripts/verify_upstream_rbm50_bank.py']
    return {**tree.code_hashes(),**neural.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}

def configure(family='nn'):
    neural.configure()
    n.RUN,n.REPORT=RUN,REPORT
    core.RUN,core.REPORT,core.bank,core.code_hashes=RUN,REPORT,tree_bank if family=='tree' else nn_bank,code_hashes
    core.replay.neural_model=neural.neural_model

def declaration():
    proof=u.read(DEVELOPMENT/'verification.json');check=u.read(DEVELOPMENT/'independent-mean-check.json')
    if (proof['state']!='complete_verified_fixed_combination' or not proof['improves_previous'] or
        not proof['original_saved_model_replays_verified'] or not proof['fixed_average_bitwise_equal'] or
        proof['origins']!=[['tree','upstream-rbm50-tree'],['nn','upstream-latent90-mps-context']] or
        proof['weights']!=[.5,.5] or proof['qualifying_labels_read'] is not False or
        check['verification_sha256']!=u.sha(DEVELOPMENT/'verification.json') or not check['fixed_mean_bitwise_equal'] or
        abs(check['rmse']-proof['rmse'])>1e-12):raise ValueError('Complete verified selected development candidate required')
    sources=dict(proof['source_sha256'])
    for p in [DEVELOPMENT/'verification.json',DEVELOPMENT/'independent-mean-check.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    iterations={}
    for family,mod in [('tree',tree),('nn',neural)]:
        records=[]
        for fold in range(5):
            path=mod.REPORT/family/f'{fold}.json';receipt=u.read(path)
            if receipt['manifest_sha256']!=u.sha(mod.REPORT/'manifest.json') or receipt['fold']!=fold or receipt['family']!=family:
                raise ValueError('Actual originating budget receipt required')
            sources[str(path.relative_to(u.ROOT))]=u.sha(path)
            for name,digest in receipt['outputs'].items():
                artifact=mod.RUN/family/str(fold)/name
                if u.sha(artifact)!=digest:raise ValueError('Budget model changed')
                sources[str(artifact.relative_to(u.ROOT))]=digest
            records.append(receipt['chosen_iterations'])
        iterations[family]=core.budgets(records,neural.CONFIG['tree_rounds' if family=='tree' else 'nn_epochs'])
    return proof,iterations,sources

def prepare():
    configure()
    if (REPORT/'plan.json').exists():return core.verify()
    if RUN.exists():raise ValueError('Partial production preserved')
    proof,iterations,sources=declaration()
    for bank,verifier,width in [(tree_bank,'verify_upstream_rbm50_bank.py',293),(nn_bank,'verify_upstream_latent90_bank.py',333)]:
        mp=bank.REPORT/'manifest.json';bp=bank.REPORT/'column-verification.json';m=u.read(mp);p=u.read(bp)
        if (p['state']!='complete' or p['manifest_sha256']!=u.sha(mp) or
            p['verifier_sha256']!=u.sha(u.ROOT/'scripts'/verifier) or not p['all_retained_columns_checked'] or
            not p['added_known_columns_bitwise_match_development'] or m['nonlinear_features']!=width or m['design_columns']!=183):
            raise ValueError('Exact independently verified paired family bank required')
        for collection in [m['source_sha256'],m['code_sha256']]:
            for name,digest in collection.items():
                if name in sources and sources[name]!=digest:raise ValueError('Conflicting source lineage')
                sources[name]=digest
        for name,digest in m['files'].items():sources[str((bank.RUN/name).relative_to(u.ROOT))]=digest
        for path in [mp,bp]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name in ['probex-design.npy','qual-design.npy','labels.npy','users.npy']:
        if u.sha(tree_bank.RUN/name)!=u.sha(nn_bank.RUN/name):raise ValueError('Family ridge and known identities differ')
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    z=np.load(tree_bank.RUN/'probex-design.npy',mmap_mode='r');y=np.load(tree_bank.RUN/'labels.npy').astype(np.float64)
    gram=np.zeros((183,183));rhs=np.zeros(183)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));a=z[lo:hi];gram+=a.T@a;rhs+=a.T@y[lo:hi]
    w=n.ridge_weights(gram[None],rhs[None],np.array([True]),n.CONFIG['ridge_lambda'])
    RUN.mkdir();np.save(RUN/'ridge.npy',w)
    plan=dict(state='fixed_mean_declared_before_full_known_fitting_and_new_qualifying_outcomes',created=resources.stamp(),
        config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,
        iterations=iterations,seeds=n.CONFIG['seeds'],weights={'tree':.5,'nn':.5},qualifying_rows=2817131,
        qualifying_labels_read=False,development_rmse=proof['rmse'],ridge_sha256=u.sha(RUN/'ridge.npy'),prediction_chunk_rows=core.CHUNK,
        family_schemas={'tree':dict(non_rating_support_columns=50,features_plus_ridge=294,backend='lightgbm_cpu'),
                        'nn':dict(non_rating_support_columns=90,features_plus_ridge=334,backend='torch_mps_float32')},
        iteration_rule='Per-seed medians of five actual originating nested user-fold receipts for each family.',
        final_rule='Fit both families on all known labels with exact originating feature order. Actual saved-model reload and full qualifying inference replay for each. Freeze the fixed float32 half mean before recovered-label access.',
        bank_scope_note='Exact paired50 tree features and paired90 neural features; common183-column ridge. Previously disclosed parent source-realization differences retained. No public numerical predictions or qualifying-outcome fitting/selection.')
    u.write(REPORT/'plan.json',plan);return plan

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof,iterations,_=declaration()
        u.write(REPORT/'declaration.json',dict(created=resources.stamp(),development_verification_sha256=u.sha(DEVELOPMENT/'verification.json'),
            development_rmse=proof['rmse'],iterations=iterations,weights={'tree':.5,'nn':.5},qualifying_labels_read=False,
            rule='Both family fits and actual replays, fixed half mean, freeze, recovered audit and separate frozen-mean scoring.'))
        while not (nn_bank.REPORT/'column-verification.json').exists():
            u.write(REPORT/'status.json',dict(state='waiting_for_independently_verified_paired90',observed=resources.stamp()));time.sleep(30)
        with (tree.neural.base.source.REPORT/'run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            previous=resources.memory.memory();good=0
            while good<2:
                current=resources.memory.memory();workers=resources.footprints()
                good=good+1 if (sum(v['bytes'] for v in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and
                    current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20) else 0
                u.write(REPORT/'status.json',dict(state='waiting_for_full_fit_memory',observed=resources.stamp(),workers=workers,memory=current,good_samples=good))
                if good<2:previous=current;time.sleep(30)
            prepare()
            for family in ['nn','tree']:
                for replay in [False,True]:
                    u.write(REPORT/'status.json',dict(state='actual_saved_model_replay' if replay else 'full_known_fitting',family=family,observed=resources.stamp()))
                    command=[sys.executable,'-u',str(Path(__file__)),'--family',family]+(['--replay'] if replay else [])
                    with (REPORT/(family+'.log')).open('a') as log:subprocess.run(command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            configure();core.verify();families=[]
            for family in ['tree','nn']:
                receipt=u.read(REPORT/(family+'.json'));replay=u.read(REPORT/(family+'-replay.json'))
                if not replay['bitwise_equal'] or replay['family_receipt_sha256']!=u.sha(REPORT/(family+'.json')):
                    raise ValueError('Actual full-family saved-model replay required')
                path=RUN/family/'qualifying.f32'
                if u.sha(path)!=receipt['outputs']['qualifying.f32']:raise ValueError('Family vector changed')
                values=np.fromfile(path,dtype='<f4').astype(np.float64)
                if values.shape!=(2817131,) or not np.isfinite(values).all():raise ValueError('Invalid family vector')
                families.append(values)
            final=((families[0]+families[1])/2).astype('<f4');final.tofile(REPORT/'qualifying-final.f32')
            u.write(REPORT/'report.json',dict(state='complete',qualifying_labels_read=False,qualifying_rows=len(final),
                frozen_prediction_sha256=u.sha(REPORT/'qualifying-final.f32'),plan_sha256=u.sha(REPORT/'plan.json'),
                replay_receipts={family:u.sha(REPORT/(family+'-replay.json')) for family in ['tree','nn']},
                scope='Known-CV-declared own tree50/NN90 fixed float32 mean, both actual full-family replays complete. Frozen before canonical recovered-label access.'))
            subprocess.run([sys.executable,str(u.ROOT/'scripts/audit_qualifying.py'),'--model-dir',str(REPORT),
                '--output-dir',str(REPORT/'recovered-audit')],cwd=u.ROOT,check=True)
            import verify_upstream_frozen_mean as verifier
            verifier.verify(Path(__file__).stem)
            u.write(REPORT/'status.json',dict(state='complete_frozen_qualifying_audit_and_separate_verification',observed=resources.stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['tree','nn']);p.add_argument('--replay',action='store_true');args=p.parse_args()
    configure(args.family or 'nn')
    core.worker(args.family,args.replay) if args.family else run()
