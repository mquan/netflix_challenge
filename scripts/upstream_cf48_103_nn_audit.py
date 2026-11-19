#!/usr/bin/env python3
"""Predeclare the stronger known-CV NN103 candidate; audit its frozen full fit."""
import datetime as dt
import gzip
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import upstream_cf48_103_production as production

u=production.u
REPORT=u.ROOT/'experiments/upstream-cf48-103-nn-audit'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def owner(pid):
    return subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()

def run(pid):
    if REPORT.exists():raise ValueError('Existing NN-only audit preserved')
    if (production.REPORT/'recovered-audit/report.json').exists():raise ValueError('Predeclare before either new qualifying outcome')
    identity=owner(pid)
    if not identity or 'upstream_cf48_103_production.py' not in identity:raise ValueError('Actual live full-fit owner required')
    production.development.configure()
    report=production.development.REPORT;cache=production.development.RUN
    ep,pp,rp=[report/name for name in ['evaluation.json','independent-verification.json','nn-replay.json']]
    e,p,r=[u.read(path) for path in [ep,pp,rp]]
    nn=next(row for row in e['records'] if row['arm']=='nn')
    checked=next(row for row in p['records'] if row['arm']=='nn')
    if (p['evaluation_sha256']!=u.sha(ep) or abs(nn['rmse']-checked['independent_rmse'])>1e-12 or
            nn['rmse']>=min(row['rmse'] for row in e['records'] if row['arm']!='nn') or
            nn['rmse']>=e['previous_best_rmse'] or [row['fold'] for row in r['records']]!=list(range(5)) or
            not all(row['bitwise_equal'] for row in r['records']) or u.sha(cache/'nn.f32')!=checked['prediction_sha256']):
        raise ValueError('Independently verified stronger standalone NN required')
    paths=[Path(__file__),ep,pp,rp,report/'manifest.json',cache/'nn.f32']
    for fold in range(5):
        receiptpath=report/'nn'/f'{fold}.json';receipt=u.read(receiptpath);paths.append(receiptpath)
        if receipt['manifest_sha256']!=u.sha(report/'manifest.json'):raise ValueError('Nested receipt mismatch')
        for name,digest in receipt['outputs'].items():
            artifact=cache/'nn'/str(fold)/name
            if u.sha(artifact)!=digest:raise ValueError('Nested model changed')
            paths.append(artifact)
    REPORT.mkdir()
    plan=dict(state='nn_only_selected_from_known_cv_before_new_qualifying_outcomes',created=stamp(),
        development_rmse=nn['rmse'],previous_best_rmse=e['previous_best_rmse'],weights=dict(nn=1.0,tree=0.0),
        sources={str(path.relative_to(u.ROOT)):u.sha(path) for path in paths},
        full_fit_owner=dict(pid=pid,identity=identity),qualifying_labels_read=False,
        rule='Use exactly the new full-fit NN103 family vector with actual saved-model replay, no tree mixing or qualifying-fitted weights. Choice is fixed before either new qualifying outcome.')
    u.write(REPORT/'plan.json',plan)
    while not (production.REPORT/'nn-replay.json').exists():
        if owner(pid)!=identity:raise RuntimeError('Full-fit owner ended before NN replay; outputs preserved')
        u.write(REPORT/'status.json',dict(state='waiting_for_actual_full_nn_replay',observed=stamp(),pid=pid));time.sleep(20)
    production.configure();full=production.core.verify()
    receiptpath=production.REPORT/'nn.json';replaypath=production.REPORT/'nn-replay.json'
    receipt,replay=u.read(receiptpath),u.read(replaypath);fullpath=production.REPORT/'plan.json'
    if (receipt['plan_sha256']!=u.sha(fullpath) or replay['plan_sha256']!=u.sha(fullpath) or
            replay['family_receipt_sha256']!=u.sha(receiptpath) or not replay['bitwise_equal'] or
            replay['prediction_sha256']!=receipt['outputs']['qualifying.f32'] or receipt['qualifying_labels_read'] is not False):
        raise ValueError('Actual full-fit NN replay lineage mismatch')
    for name,digest in receipt['outputs'].items():
        if u.sha(production.RUN/'nn'/name)!=digest:raise ValueError('Full-fit NN artifact changed')
    source=production.RUN/'nn/qualifying.f32';frozen=REPORT/'qualifying-final.f32';u.link_or_copy(source,frozen)
    if u.sha(frozen)!=receipt['outputs']['qualifying.f32']:raise ValueError('Frozen standalone NN differs')
    for name,digest in plan['sources'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Predeclared selection evidence changed')
    u.write(REPORT/'report.json',dict(state='complete',qualifying_labels_read=False,qualifying_rows=2817131,
        frozen_prediction_sha256=u.sha(frozen),plan_sha256=u.sha(REPORT/'plan.json'),
        full_fit_plan_sha256=u.sha(fullpath),nn_receipt_sha256=u.sha(receiptpath),nn_replay_sha256=u.sha(replaypath),
        scope='Predeclared known-CV-selected standalone NN103, exact full-fit own vector frozen after actual saved-model replay and before this audit. No fitting performed here.'))
    subprocess.run([sys.executable,str(u.ROOT/'scripts/audit_qualifying.py'),'--model-dir',str(REPORT),
        '--output-dir',str(REPORT/'recovered-audit')],check=True,cwd=u.ROOT)
    labelpath=REPORT/'recovered-audit/qual_ratings.csv.gz';blob=labelpath.read_bytes()
    if hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()!='47ed4788ffba11aad029854a8e60d4dee04fc899':
        raise ValueError('Canonical recovered label blob mismatch')
    with gzip.open(labelpath,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    values=np.fromfile(frozen,dtype='<f4');audit=u.read(REPORT/'recovered-audit/report.json')
    if labels.shape!=(2817131,2) or values.shape!=(2817131,) or not np.isfinite(values).all():raise ValueError('Full query coverage mismatch')
    groups={}
    for flag,name,count in [(0,'source0',1408789),(1,'source1',1408342)]:
        mask=labels[:,1]==flag
        if int(mask.sum())!=count:raise ValueError('Recovered group count mismatch')
        score=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(values[mask],labels[mask,0]))/count)
        if abs(score-audit['quiz' if flag==0 else 'test']['rmse'])>1e-12:raise ValueError('Independent compensated score mismatch')
        groups[name]=dict(rows=count,rmse=score,below_target=score<.85725)
    if u.sha(frozen)!=receipt['outputs']['qualifying.f32'] or u.sha(source)!=u.sha(frozen):raise ValueError('Frozen NN changed during audit')
    production.core.verify()
    proof=dict(state='complete_standalone_nn_frozen_vector_and_score_verification',observed=stamp(),groups=groups,
        both_groups_below_target=all(row['below_target'] for row in groups.values()),prediction_sha256=u.sha(frozen),
        plan_sha256=u.sha(REPORT/'plan.json'),full_fit_plan_sha256=u.sha(fullpath),nn_receipt_sha256=u.sha(receiptpath),
        nn_replay_sha256=u.sha(replaypath),audit_report_sha256=u.sha(REPORT/'recovered-audit/report.json'),
        label_sha256=u.sha(labelpath),verifier_sha256=u.sha(Path(__file__)),standalone_vector_matches_saved_family_bitwise=True,
        qualifying_labels_used_for_fitting_or_selection=False,
        scope='Predeclared known-CV-only standalone choice; actual original NN model/scaler/output hashes and saved-inference replay receipt verified. Canonical full recovered groups independently scored with compensated summation. No inference rerun here; adaptive recovered-label audit, not official/untouched Test.')
    u.write(REPORT/'verification.json',proof);print(proof,flush=True)

if __name__=='__main__':run(int(sys.argv[1]))
