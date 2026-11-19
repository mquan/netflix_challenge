#!/usr/bin/env python3
"""Recheck frozen full-family means and recovered scores without fitting."""
import argparse
import datetime as dt
import gzip
import hashlib
import importlib
import math
from pathlib import Path
import numpy as np

ADAPTERS=['upstream_cf12_matching_production','upstream_tree91_context86_production']
LABEL_BLOB='47ed4788ffba11aad029854a8e60d4dee04fc899'


def verify(name):
    adapter=importlib.import_module(name);adapter.configure();u=adapter.u
    plan=adapter.core.verify();root,report=adapter.RUN,adapter.REPORT
    plan_path=report/'plan.json';plan_digest=u.sha(plan_path);frozen=u.read(report/'report.json')
    if (frozen['state']!='complete' or frozen['qualifying_labels_read'] is not False or
            frozen['plan_sha256']!=plan_digest or frozen['qualifying_rows']!=2817131):
        raise ValueError('Complete frozen prediction report required')
    bindings={str(plan_path.relative_to(u.ROOT)):plan_digest};families={}
    for family in ['tree','nn']:
        receipt_path=report/(family+'.json');receipt=u.read(receipt_path)
        replay_path=report/(family+'-replay.json');replay=u.read(replay_path)
        if (receipt['plan_sha256']!=plan_digest or receipt['qualifying_labels_read'] is not False or
                not replay['bitwise_equal'] or replay['plan_sha256']!=plan_digest or
                replay['family_receipt_sha256']!=u.sha(receipt_path) or
                replay['prediction_sha256']!=receipt['outputs']['qualifying.f32'] or
                frozen['replay_receipts'][family]!=u.sha(replay_path)):
            raise ValueError('Saved-family replay binding differs')
        for filename,digest in receipt['outputs'].items():
            path=root/family/filename
            if u.sha(path)!=digest:raise ValueError('Saved-family artifact changed')
            bindings[str(path.relative_to(u.ROOT))]=digest
        for path in [receipt_path,replay_path]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
        values=np.fromfile(root/family/'qualifying.f32',dtype='<f4')
        if values.shape!=(2817131,) or not np.isfinite(values).all():raise ValueError('Invalid family vector')
        families[family]=values.astype(np.float64)
    final_path=report/'qualifying-final.f32';final=np.fromfile(final_path,dtype='<f4')
    expected=((families['tree']+families['nn'])/2).astype('<f4')
    if (not np.array_equal(final.view(np.uint32),expected.view(np.uint32)) or
            u.sha(final_path)!=frozen['frozen_prediction_sha256']):raise ValueError('Frozen float32 mean differs')
    label_path=report/'recovered-audit/qual_ratings.csv.gz';blob=label_path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()!=LABEL_BLOB:
        raise ValueError('Recovered label Git blob differs')
    audit_path=report/'recovered-audit/report.json';audit=u.read(audit_path)
    if (audit['source']['prediction_sha256']!=u.sha(final_path) or
            audit['label_sha256']!=u.sha(label_path)):raise ValueError('Audit source binding differs')
    with gzip.open(label_path,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if (labels.shape!=(2817131,2) or np.any((labels[:,0]<1)|(labels[:,0]>5)) or
            set(np.unique(labels[:,1]))!={0,1}):raise ValueError('Recovered row coverage differs')
    groups={}
    for value,name,rows in [(0,'source0',1408789),(1,'source1',1408342)]:
        mask=labels[:,1]==value
        if int(mask.sum())!=rows:raise ValueError('Recovered group size differs')
        score=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(final[mask],labels[mask,0]))/rows)
        published=audit['quiz' if value==0 else 'test']['rmse']
        if abs(score-published)>1e-12:raise ValueError('Compensated recovered score differs')
        groups[name]=dict(rows=rows,rmse=score,below_target=score<.85725)
    for path in [Path(__file__),report/'report.json',final_path,label_path,audit_path]:
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    for relative,digest in bindings.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Evidence changed during verification')
    adapter.core.verify()
    result=dict(state='complete_frozen_mean_and_score_verification',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        adapter=adapter.__name__,groups=groups,fixed_float32_mean_bitwise_equal=True,
        saved_family_replay_receipts_verified=True,source_inputs_verified=len(plan['source_sha256']),
        code_files_verified=len(plan['code_sha256']),prediction_sha256=u.sha(final_path),label_git_blob=LABEL_BLOB,
        bindings=bindings,both_groups_below_target=all(r['below_target'] for r in groups.values()),
        model_fitting_performed=False,qualifying_labels_used_for_fitting_or_selection=False,
        scope='Separate frozen-mean and compensated-score recheck. Actual saved-family replay receipts and model hashes verified; this check does not rerun model inference. Adaptive community-recovered groups, not official or untouched Netflix Test.')
    u.write(report/'frozen-mean-score-verification.json',result);print(dict(groups=groups,both_groups_below_target=result['both_groups_below_target']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--adapter',required=True,choices=ADAPTERS)
    verify(p.parse_args().adapter)
