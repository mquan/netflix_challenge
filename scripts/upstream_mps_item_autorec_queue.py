#!/usr/bin/env python3
"""After the current raw run, prepare/verify real transposed data and fit50/50."""
from pathlib import Path
import fcntl
import importlib.metadata
import subprocess
import sys
import time
import upstream_mps_autorec as preceding
import upstream_mps_item_autorec as model

u=model.u
REPORT=u.ROOT/'experiments/upstream-mps-item-autorec-queue'

def verify_prerequisite():
    preceding.verify_plan();path=preceding.REPORT/'completion.json';done=u.read(path)
    if (done['state']!='complete_both_originally_declared50_epoch_own_raw_phases' or
            done['plan_sha256']!=u.sha(preceding.REPORT/'plan.json') or done['epochs_per_phase']!=50 or
            not done['all_declared_snapshot_pairs_complete'] or done['qualifying_labels_read'] is not False):
        raise ValueError('Actual complete preceding50/50 run required, not an ETA or epoch log')
    for epoch in preceding.CONFIG['snapshot_epochs']:
        pair_path=preceding.REPORT/f'pair-{epoch:05}.json';pair=u.read(pair_path)
        if pair['epoch']!=epoch or pair['plan_sha256']!=done['plan_sha256'] or pair['qualifying_labels_read'] is not False:
            raise ValueError('Preceding complete fixed snapshot pair differs')
        if epoch==50 and not pair['full50_epoch_model_complete']:raise ValueError('Both original50-epoch phases required')
        for phase,(query,_,count) in preceding.data.PHASES.items():
            rp=preceding.REPORT/f'{phase}-snapshot-{epoch:05}.json';r=u.read(rp)
            root=preceding.RUN/phase/'snapshots'/f'epoch-{epoch:05}'
            if (pair['receipts'][phase]!=u.sha(rp) or r['epoch']!=epoch or r['phase']!=phase or r['rows']!=count or
                    not r['actual_saved_model_replay_bitwise'] or u.sha(root/'model.pt')!=r['model_sha256'] or
                    u.sha(root/'prediction.npy')!=r['prediction_sha256']):raise ValueError('Original actual raw snapshot evidence changed')
    return u.sha(path)

def execute(script,logname):
    with (REPORT/logname).open('x') as log:
        subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing item-oriented queue declaration preserved')
        benchmark=u.ROOT/'experiments/upstream-mps-item-autorec-benchmark-v2/verification.json'
        b=u.read(benchmark);fixture=u.ROOT/'experiments/upstream-mps-item-autorec-fixture/verification.json';f=u.read(fixture)
        if (b['state']!='complete_synthetic_benchmark_report_bindings' or not b['all_four_actual_saved_model_replay_reports_verified'] or
                f['state']!='complete_isolated_item_oriented_fixture' or not f['high_user_ids_uint32_preserved'] or
                not f['independent_wrong_column_rejected'] or not f['query_leakage_rejected'] or
                f['actual_saved_snapshot_query_replays']!=6 or not f['independent_original_query_gather_bitwise']):
            raise ValueError('Actual full-width benchmark and complete isolated transpose/resume fixture required')
        benchmark_root=benchmark.parent
        if b['versions']!={name:importlib.metadata.version(name) for name in ['numpy','torch']}:
            raise ValueError('Benchmarked library versions changed')
        for name,digest in b['report_sha256'].items():
            if u.sha(benchmark_root/name)!=digest:raise ValueError('Benchmarked report binding changed')
        if b['script_sha256']!=u.sha(u.ROOT/'scripts/benchmark_upstream_mps_item_autorec_v2.py'):
            raise ValueError('Benchmarked full-width implementation changed')
        for name,digest in f['code_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Tested model/data/check implementation changed')
        paths=[Path(__file__),Path(model.__file__),Path(model.data.__file__),
            u.ROOT/'scripts/verify_upstream_mps_item_autorec_data.py',u.ROOT/'scripts/verify_upstream_mps_item_autorec_snapshot.py']
        hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
        u.write(REPORT/'plan.json',dict(created=model.stamp(),config=model.CONFIG,code_sha256=hashes,
            versions=b['versions'],
            benchmark_verification_sha256=u.sha(benchmark),fixture_verification_sha256=u.sha(fixture),
            prerequisite='Actual preceding raw50/50 completion and all original saved snapshot replay/model bindings. Serialize to preserve disk headroom.',
            rule='Losslessly transpose both complete canonical training histories. Separately replay every tuple and query exclusion. Fit own480189→128→480189 item-oriented raw model, batch32,50 epochs/phase, fixed snapshots5/10/20/30/40/50. One durable atomic optimizer/RNG checkpoint per phase; preserve every snapshot weight/query vector. Actual saved-model full query inference replay required.',
            numerical_public_features_read=False,qualifying_labels_read=False,known_query_labels_used_for_stopping_or_configuration=False,
            model_capacity_or_data_subsampling=False,original_native_recipe_parity_claimed=False))
        try:
            while not (preceding.REPORT/'completion.json').exists():
                u.write(REPORT/'status.json',dict(state='waiting_for_actual_preceding_raw50_50_completion',observed=model.stamp()));time.sleep(30)
            digest=verify_prerequisite();u.write(REPORT/'prerequisite-verification.json',dict(state='complete_original50_50_and_all_snapshot_replay_bindings',completion_sha256=digest,qualifying_labels_read=False))
            if b['versions']!={name:importlib.metadata.version(name) for name in ['numpy','torch']}:
                raise ValueError('Predeclared library versions changed while queued')
            for name,sha in hashes.items():
                if u.sha(u.ROOT/name)!=sha:raise ValueError('Predeclared item-oriented implementation changed')
            u.write(REPORT/'status.json',dict(state='preparing_lossless_item_oriented_original_histories',observed=model.stamp()))
            execute('upstream_mps_item_autorec_data.py','data.log')
            execute('verify_upstream_mps_item_autorec_data.py','independent-data.log')
            for phase in model.data.PHASES:
                r=u.read(model.data.REPORT/(phase+'-verification.json'))
                if not r['every_training_tuple_replayed'] or not r['query_pairs_absent_from_training']:raise ValueError('Complete original tuple/query verification required')
            u.write(REPORT/'status.json',dict(state='fitting_real_item_oriented50_50_recipe',observed=model.stamp()))
            execute('upstream_mps_item_autorec.py','training.log')
            for epoch in model.CONFIG['snapshot_epochs']:
                with (REPORT/f'independent-snapshot-{epoch:05}.log').open('x') as log:
                    subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_mps_item_autorec_snapshot.py'),'--epoch',str(epoch)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            u.write(REPORT/'status.json',dict(state='complete_real50_50_and_all_snapshot_independent_checks',observed=model.stamp(),qualifying_labels_read=False,goal_complete=False))
        except BaseException as error:
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error),observed=model.stamp()));raise

if __name__=='__main__':run()
