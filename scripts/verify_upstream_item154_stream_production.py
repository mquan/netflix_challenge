#!/usr/bin/env python3
"""Independently check frozen consensus arithmetic and canonical group scores."""
import datetime as dt
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_item154_stream_production as production

u=production.u
LABEL_BLOB='47ed4788ffba11aad029854a8e60d4dee04fc899'

def verify():
    production.configure();plan=production.core.verify();root,report=production.RUN,production.REPORT
    frozen=u.read(report/'report.json');receipt=u.read(report/'nn.json');replay=u.read(report/'nn-replay.json')
    digest=u.sha(report/'plan.json')
    if (frozen['state']!='complete' or frozen['qualifying_labels_read'] is not False or frozen['plan_sha256']!=digest or
            receipt['plan_sha256']!=digest or replay['plan_sha256']!=digest or not replay['bitwise_equal'] or
            replay['family_receipt_sha256']!=u.sha(report/'nn.json') or frozen['new_nn_replay_sha256']!=u.sha(report/'nn-replay.json')):
        raise ValueError('Frozen plan and actual new full-model replay must match')
    bindings={str((report/'plan.json').relative_to(u.ROOT)):digest}
    for name,d in receipt['outputs'].items():
        p=root/'nn'/name
        if u.sha(p)!=d:raise ValueError('New saved neural artifact differs')
        bindings[str(p.relative_to(u.ROOT))]=d
    reused=production.reused_sources()
    for name,d in reused.items():
        if u.sha(u.ROOT/name)!=d:raise ValueError('Original full-model source changed')
    bindings.update(reused)
    tree=np.fromfile(production.OLD_RUN/'tree/qualifying.f32',dtype='<f4').astype(np.float64)
    previous=np.fromfile(production.OLD_REPORT/'qualifying-final.f32',dtype='<f4').astype(np.float64)
    new=np.fromfile(root/'nn/qualifying.f32',dtype='<f4').astype(np.float64)
    for a in [tree,previous,new]:
        if a.shape!=(2817131,) or not np.isfinite(a).all():raise ValueError('Complete finite query vectors required')
    intermediate=np.multiply(np.add(tree,new),.5).astype('<f4')
    expected=intermediate
    final_path=report/'qualifying-final.f32';stored=np.fromfile(final_path,dtype='<f4')
    if not np.array_equal(stored.view(np.uint32),expected.view(np.uint32)) or u.sha(final_path)!=frozen['frozen_prediction_sha256']:
        raise ValueError('Frozen exact rounded consensus differs')
    label_path=report/'recovered-audit/qual_ratings.csv.gz';payload=label_path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()!=LABEL_BLOB:raise ValueError('Canonical recovered label blob differs')
    audit=u.read(report/'recovered-audit/report.json')
    if audit['source']['prediction_sha256']!=u.sha(final_path) or audit['label_sha256']!=u.sha(label_path):raise ValueError('Audit source differs')
    with gzip.open(label_path,'rt') as stream:
        if stream.readline().strip()!='rating,is_test':raise ValueError('Label header differs')
        labels=np.loadtxt(stream,delimiter=',',dtype=np.int16)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,1],[0,1]).all() or not ((labels[:,0]>=1)&(labels[:,0]<=5)).all():
        raise ValueError('Canonical complete label domain differs')
    groups={}
    for value,count in [(0,1408789),(1,1408342)]:
        ids=np.flatnonzero(labels[:,1]==value)
        if len(ids)!=count:raise ValueError('Canonical group count differs')
        score=math.sqrt(math.fsum((float(stored[i])-float(labels[i,0]))**2 for i in ids)/count)
        old_name='quiz' if value==0 else 'test'
        if abs(audit[old_name]['rmse']-score)>1e-12:raise ValueError('Independent compensated group score differs')
        groups['source'+str(value)]={'rows':count,'rmse':score}
    for p in [report/'report.json',report/'nn.json',report/'nn-replay.json',final_path,label_path,report/'recovered-audit/report.json']:
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,d in plan['source_sha256'].items():
        if u.sha(u.ROOT/name)!=d:raise ValueError('Frozen own source changed during verification')
    result=dict(state='complete_independent_frozen_consensus_and_scores',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        groups=groups,weights=[.5,.5],label_git_blob=LABEL_BLOB,exact_rounded_consensus_bitwise_equal=True,
        actual_new_full_neural_replay_verified=True,original_full_model_replays_reused_and_verified=True,
        prediction_sha256=u.sha(final_path),plan_sha256=digest,bindings=bindings,verifier_sha256=u.sha(Path(__file__)),
        both_groups_below_target=all(r['rmse']<.85725 for r in groups.values()),qualifying_labels_used_for_fitting_or_selection=False,
        source_inputs_verified=len(plan['source_sha256']),scope='Independent fixed rounded consensus, original and new actual replay/model/source bindings and canonical complete-group compensated scores. This checker does not rerun inference. Adaptive recovered audit, not official/untouched Test.')
    path=report/'independent-verification.json'
    if path.exists():raise ValueError('Existing independent production proof preserved')
    u.write(path,result);print(groups,flush=True)

if __name__=='__main__':verify()
