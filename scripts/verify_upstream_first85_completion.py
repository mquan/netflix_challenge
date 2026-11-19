#!/usr/bin/env python3
"""Read-only completion audit of the first own-trained85 qualifying production."""
import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_recovered_temporal_production as adapter

u=adapter.u
REPORT=u.ROOT/'experiments/upstream-first85-completion-audit'
LABEL_BLOB='47ed4788ffba11aad029854a8e60d4dee04fc899'
LABEL_BYTES=1579643
TARGET=.85725
ROWS=2817131


def average_contract(tree,neural,final):
    arrays=[np.asarray(a) for a in [tree,neural,final]]
    if any(a.ndim!=1 or a.dtype!=np.dtype('<f4') or a.shape!=arrays[0].shape or
           not np.isfinite(a).all() or np.any(a<np.float32(1.05)) or np.any(a>np.float32(4.95)) for a in arrays):
        raise ValueError('Invalid clipped float32 family/final vectors')
    expected=((arrays[0].astype(np.float64)+arrays[1].astype(np.float64))/2).astype('<f4')
    if expected.tobytes()!=arrays[2].tobytes():raise ValueError('Frozen final is not the exact fixed family average')
    return True


def score_groups(predictions,labels,expected_counts=(1408342,1408789)):
    if (labels.shape!=(len(predictions),2) or not np.issubdtype(labels.dtype,np.integer) or
            not np.all((labels[:,0]>=1)&(labels[:,0]<=5)) or not np.all(np.isin(labels[:,1],[0,1]))):
        raise ValueError('Invalid recovered label shape/domain')
    groups={}
    for value,name in [(0,'quiz'),(1,'test')]:
        mask=labels[:,1]==value;count=int(mask.sum())
        if not count:raise ValueError('Empty recovered source group')
        rmse=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(predictions[mask],labels[mask,0]))/count)
        groups[name]=dict(rows=count,rmse=rmse,below_target=rmse<TARGET,
            improvement_percent_over_cinematch=100*(1-rmse/.9525))
    if sorted(g['rows'] for g in groups.values())!=sorted(expected_counts):
        raise ValueError('Recovered source groups do not cover the required populations')
    return groups


def producer_live(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],text=True,capture_output=True)
    row=result.stdout.strip()
    return bool(row and not row.split()[0].startswith('Z') and
                'scripts/upstream_recovered_temporal_production.py' in row and '--family' not in row)


def provenance(bank,plan):
    jobs=u.read(u.BASE/'registry.json')['jobs'];records={}
    if len(bank['columns'])!=85 or bank['qualifying_labels_read'] is not False:
        raise ValueError('Own paired85 source bank required')
    adapter.bank.temporal.verify_receipt()
    frozen_at=dt.datetime.fromisoformat(plan['created'])
    for column in bank['columns']:
        path=u.ROOT/column['receipt'];record=u.read(path)
        if u.sha(path)!=column['receipt_sha256']:raise ValueError('Source receipt changed')
        if column['kind']=='our-completed-remote-native-job':
            if column['name'] not in adapter.bank.temporal.NAMES or record['epochs_per_phase']!=25:
                raise ValueError('Unexpected recovered temporal producer')
            continue
        if column['kind']!='native':raise ValueError('Unexpected source bank provenance')
        name=column['producer']
        if (record['job']!=name or record['source_commit']!=u.PIN or record['command']!=jobs[name]['command'] or
                not record.get('finished') or dt.datetime.fromisoformat(record['finished'])>frozen_at or
                not set(jobs[name]['outputs']).issubset(record['outputs'])):
            raise ValueError('Complete pinned-source job must precede fitting: '+name)
        for split in ['probex','qual']:
            expected='preds_new/'+column['name']+'.'+split+'.npy'
            if (column['files'][split]!=str((u.FRESH/expected).relative_to(u.ROOT)) or
                    record['outputs'].get(expected)!=column['sha256'][split]):
                raise ValueError('Selected source pair is not owned by the complete native job')
        if name in records:continue
        for relative,digest in record['outputs'].items():
            if u.sha(u.FRESH/relative)!=digest:raise ValueError('Original native output changed: '+relative)
        origin=record.get('imported_from')
        if origin:
            original=Path(origin['native_receipt'])
            if u.sha(original)!=origin['native_receipt_sha256']:raise ValueError('Own imported native receipt changed')
            source=u.read(original)
            if (source['job']!=name or source['command']!=record['command'] or source['source_commit']!=u.PIN or
                    source['outputs']!=record['outputs']):
                raise ValueError('Imported execution provenance differs')
        records[name]=dict(receipt=column['receipt'],sha256=u.sha(path),outputs_rehashed=len(record['outputs']),
            own_auxiliary_import=bool(origin))
    return records


def budget_evidence(plan):
    origin=adapter.core.voting;manifest=origin.REPORT/'manifest.json';bound={manifest:u.sha(manifest)}
    for family in ['tree','nn']:
        choices=[]
        for fold in range(5):
            path=origin.REPORT/family/(str(fold)+'.json');receipt=u.read(path)
            if (receipt['manifest_sha256']!=bound[manifest] or receipt['family']!=family or receipt['fold']!=fold or
                    plan['source_sha256'].get(str(path.relative_to(u.ROOT)))!=u.sha(path)):
                raise ValueError('Actual development budget receipt binding differs')
            counts=receipt['chosen_iterations']
            if len(counts)!=3 or any(type(v) is not int or v<0 for v in counts):raise ValueError('Invalid seed budgets')
            choices.append(counts);bound[path]=u.sha(path)
            for name,digest in receipt['outputs'].items():
                artifact=origin.RUN/family/str(fold)/name
                if u.sha(artifact)!=digest:raise ValueError('Original budget-source model changed')
                bound[artifact]=digest
        if [sorted(row[seed] for row in choices)[2] for seed in range(3)]!=plan['iterations'][family]:
            raise ValueError('Full refit budgets differ from actual per-seed development medians')
    return bound


def replay_family(family):
    # LightGBM and PyTorch ship different OpenMP runtimes on this host. Each
    # replay must own a separate process, as in the numerical production runner.
    adapter.configure();plan=adapter.core.verify();root=adapter.REPORT;cache=adapter.RUN/family
    receipt=u.read(root/(family+'.json'));bound={}
    if receipt['plan_sha256']!=u.sha(root/'plan.json'):raise ValueError('Saved-family plan differs')
    for name,digest in receipt['outputs'].items():
        if u.sha(cache/name)!=digest:raise ValueError('Saved-family artifact changed before replay')
        bound[name]=digest
    saved=np.fromfile(cache/'qualifying.f32',dtype='<f4')
    if saved.shape!=(ROWS,):raise ValueError('Independent replay needs complete qualifying coverage')
    actual=adapter.core.infer(family,plan,cache)
    if actual.tobytes()!=saved.tobytes():raise ValueError('Independent saved-model inference differs')
    result=dict(state='complete',family=family,rows=ROWS,bitwise_equal=True,
        plan_sha256=u.sha(root/'plan.json'),family_receipt_sha256=u.sha(root/(family+'.json')),
        prediction_sha256=u.sha(cache/'qualifying.f32'),saved_outputs=bound,
        verifier_sha256=u.sha(Path(__file__)),model_fitting_performed=False)
    u.write(REPORT/(family+'-replay.json'),result);return result


def verify():
    adapter.configure();root=adapter.REPORT;run=adapter.RUN
    # No qualifying labels are read until the frozen plan, full bank, source
    # ownership, all saved models and both independent inference passes verify.
    plan=adapter.core.verify();bank=adapter.bank.build();u.validate_fresh()
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):
        raise ValueError('Native qualifying training targets are not zero')
    model=u.read(root/'report.json');audit=u.read(root/'recovered-audit/report.json')
    primary=u.read(root/'independent-qualifying-verification.json')
    model_sha=u.sha(root/'report.json');plan_sha=u.sha(root/'plan.json')
    if (model['state']!='complete' or model['qualifying_labels_read'] is not False or model['qualifying_rows']!=ROWS or
            model['plan_sha256']!=plan_sha or model['paired_bank_manifest_sha256']!=u.sha(adapter.bank.REPORT/'manifest.json') or
            audit['source']['model_report_sha256']!=model_sha or
            audit['source']['prediction_sha256']!=model['frozen_prediction_sha256'] or
            primary['audit_report_sha256']!=u.sha(root/'recovered-audit/report.json')):
        raise ValueError('Frozen model/plan/bank/audit binding differs')
    native=provenance(bank,plan)
    bound={u.ROOT/name:digest for name,digest in {**plan['source_sha256'],**plan['code_sha256']}.items()}
    bound.update({root/'report.json':model_sha,root/'plan.json':plan_sha,**budget_evidence(plan)})
    family_vectors={}
    for family in ['tree','nn']:
        receipt_path=root/(family+'.json');receipt=u.read(receipt_path)
        replay_path=root/(family+'-replay.json');replay=u.read(replay_path)
        if (receipt['plan_sha256']!=plan_sha or receipt['qualifying_labels_read'] is not False or
                replay['plan_sha256']!=plan_sha or not replay['bitwise_equal'] or
                model['replay_receipts'][family]!=u.sha(replay_path)):
            raise ValueError('Full-family replay receipt binding differs')
        for name,digest in receipt['outputs'].items():
            path=run/family/name
            if u.sha(path)!=digest:raise ValueError('Saved full-family artifact changed')
            bound[path]=digest
        for path in [receipt_path,replay_path]:bound[path]=u.sha(path)
        values=np.fromfile(run/family/'qualifying.f32',dtype='<f4')
        if values.shape!=(ROWS,):raise ValueError('Full qualifying family coverage required')
        subprocess.run([sys.executable,str(Path(__file__)),'--family',family],cwd=u.ROOT,check=True)
        own_path=REPORT/(family+'-replay.json');own=u.read(own_path)
        if (own['state']!='complete' or not own['bitwise_equal'] or own['family']!=family or own['rows']!=ROWS or
                own['plan_sha256']!=plan_sha or own['family_receipt_sha256']!=u.sha(receipt_path) or
                own['prediction_sha256']!=u.sha(run/family/'qualifying.f32') or
                own['verifier_sha256']!=u.sha(Path(__file__)) or own['saved_outputs']!=receipt['outputs']):
            raise ValueError('Isolated independent replay proof differs')
        bound[own_path]=u.sha(own_path)
        family_vectors[family]=values
        u.write(REPORT/'status.json',dict(state='saved_family_verified',family=family,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
    final_path=root/'qualifying-final.f32';final=np.fromfile(final_path,dtype='<f4');digest=u.sha(final_path)
    if final.shape!=(ROWS,) or digest!=model['frozen_prediction_sha256'] or digest!=primary['prediction_sha256']:
        raise ValueError('Frozen final prediction changed')
    average_contract(family_vectors['tree'],family_vectors['nn'],final);bound[final_path]=digest
    payload_path=root/'recovered-audit/qual_ratings.csv.gz';payload=payload_path.read_bytes()
    blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    if (len(payload)!=LABEL_BYTES or blob!=LABEL_BLOB or audit['label_git_blob_sha1']!=LABEL_BLOB or
            u.sha(payload_path)!=audit['label_sha256'] or audit['source']['source_commit']!=u.PIN):
        raise ValueError('Recovered label bytes differ from immutable pinned source blob')
    with gzip.open(payload_path,'rt') as stream:
        if stream.readline().strip()!='rating,is_test':raise ValueError('Recovered label header differs')
        labels=np.loadtxt(stream,delimiter=',',dtype=np.int16)
    groups=score_groups(final,labels)
    for name,value in groups.items():
        if (value['rows']!=primary['groups'][name]['rows'] or value['rows']!=audit[name]['rows'] or
                abs(value['rmse']-primary['groups'][name]['rmse'])>1e-12 or
                abs(value['rmse']-audit[name]['rmse'])>1e-12):
            raise ValueError('Independent recovered scoring differs')
    passed=all(g['below_target'] for g in groups.values())
    if passed!=primary['both_groups_below_target']:raise ValueError('Strict goal decision differs')
    for path in [payload_path,root/'recovered-audit/report.json',root/'independent-qualifying-verification.json',Path(__file__)]:
        bound[path]=u.sha(path)
    for path,expected in bound.items():
        if u.sha(path)!=expected:raise ValueError('Frozen evidence changed during independent completion audit')
    result=dict(state='complete_independent_completion_audit',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        groups=groups,both_groups_strictly_below_target=passed,goal_achieved=passed,target=TARGET,
        all_saved_models_independently_replayed=True,fixed_equal_average_bitwise_verified=True,
        actual_development_budget_models_and_medians_verified=True,
        own_native_source_jobs=native,paired_prediction_columns=85,qualifying_rows=ROWS,
        qualifying_labels_used_in_fitting_or_selection=False,downloaded_pretrained_prediction_features=False,
        source_sha256={str(p.relative_to(u.ROOT)):v for p,v in bound.items()},
        limitation='Adaptive community-recovered source-group audit. Published Quiz/Test counts are reversed in the source; neither group is claimed as official or untouched Netflix Test.')
    u.write(REPORT/'verification.json',result);u.write(REPORT/'status.json',dict(state='complete',goal_achieved=passed));return result


def main(watch=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if watch:
            producer=u.read(adapter.REPORT/'remote-intake-handoff.json')['pid']
            while not (adapter.REPORT/'independent-qualifying-verification.json').exists():
                if not producer_live(producer):raise RuntimeError('Verified producer exited before its qualifying audit; inspect preserved evidence')
                u.write(REPORT/'status.json',dict(state='waiting_for_live_first85_producer',producer_pid=producer,
                    observed=dt.datetime.now(dt.timezone.utc).isoformat(),process_observed_live=True))
                time.sleep(30)
        return verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    parser.add_argument('--family',choices=['tree','nn']);args=parser.parse_args()
    try:print(replay_family(args.family) if args.family else main(args.watch))
    except Exception as error:
        u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise
