#!/usr/bin/env python3
"""Recheck standalone full-fit bindings, frozen vector and compensated scores."""
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_rbm50_production as production

def verify():
    production.configure();u=production.u;plan=production.core.verify();root=production.REPORT
    receipt=u.read(root/'nn.json');replay=u.read(root/'nn-replay.json');report=u.read(root/'report.json')
    digest=u.sha(root/'plan.json')
    if (receipt['plan_sha256']!=digest or replay['plan_sha256']!=digest or report['plan_sha256']!=digest or
        replay['family_receipt_sha256']!=u.sha(root/'nn.json') or not replay['bitwise_equal'] or
        replay['prediction_sha256']!=receipt['outputs']['qualifying.f32'] or receipt['qualifying_labels_read'] is not False or
        plan['weights']!={'nn':1.0} or report['qualifying_labels_read'] is not False):
        raise ValueError('Predeclared original full-model/replay lineage differs')
    for name,expected in receipt['outputs'].items():
        if u.sha(production.RUN/'nn'/name)!=expected:raise ValueError('Saved full-model artifact changed')
    frozen=root/'qualifying-final.f32';source=production.RUN/'nn/qualifying.f32'
    if u.sha(frozen)!=u.sha(source) or u.sha(frozen)!=report['frozen_prediction_sha256']:
        raise ValueError('Standalone frozen vector differs')
    label=root/'recovered-audit/qual_ratings.csv.gz';payload=label.read_bytes()
    if hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()!='47ed4788ffba11aad029854a8e60d4dee04fc899':
        raise ValueError('Canonical recovered-label blob differs')
    with gzip.open(label,'rt') as f:labels=np.loadtxt(f,delimiter=',',skiprows=1,dtype=np.int16)
    values=np.fromfile(frozen,dtype='<f4');audit=u.read(root/'recovered-audit/report.json')
    if labels.shape!=(2817131,2) or values.shape!=(2817131,) or not np.isfinite(values).all():raise ValueError('Full query coverage differs')
    groups={}
    for flag,name,count in [(0,'source0',1408789),(1,'source1',1408342)]:
        mask=labels[:,1]==flag
        if int(mask.sum())!=count:raise ValueError('Canonical group count differs')
        score=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(values[mask],labels[mask,0]))/count)
        if abs(score-audit['quiz' if flag==0 else 'test']['rmse'])>1e-12:raise ValueError('Independent score differs')
        groups[name]=dict(rows=count,rmse=score,below_target=score<.85725)
    production.core.verify()
    proof=dict(state='complete_standalone_nn_frozen_vector_and_score_verification',observed=production.resources.stamp(),
        groups=groups,both_groups_below_target=all(v['below_target'] for v in groups.values()),
        prediction_sha256=u.sha(frozen),plan_sha256=digest,nn_receipt_sha256=u.sha(root/'nn.json'),
        nn_replay_sha256=u.sha(root/'nn-replay.json'),audit_report_sha256=u.sha(root/'recovered-audit/report.json'),
        label_sha256=u.sha(label),verifier_sha256=u.sha(Path(__file__)),standalone_vector_matches_saved_family_bitwise=True,
        qualifying_labels_used_for_fitting_or_selection=False,
        scope='Separate predeclared standalone/model/scaler/source hash and saved-inference-replay receipt check, exact frozen family identity, full canonical group counts and compensated scores. Inference was actually replayed by the full-fit worker; this checker does not repeat inference. Adaptive recovered audit, not official/untouched Test.')
    u.write(root/'verification.json',proof);print(proof,flush=True)

if __name__=='__main__':verify()
