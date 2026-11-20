#!/usr/bin/env python3
"""Independently check the actual new tree replay, reused NN and frozen group scores."""
from pathlib import Path
import gzip
import hashlib
import math
import numpy as np
import upstream_item154_tree_production as model
u=model.u

def verify():
    model.configure();plan=model.core.verify();report,root=model.REPORT,model.RUN
    if (report/'independent-verification.json').exists():raise ValueError('Existing independent full-tree proof preserved')
    frozen=u.read(report/'report.json');pd=u.sha(report/'plan.json')
    if (frozen['state']!='complete_frozen_tree154_nn154_half_mean' or frozen['plan_sha256']!=pd or
        frozen['qualifying_labels_read'] is not False or frozen['weights']!=[.5,.5] or plan['qualifying_labels_read'] is not False):raise ValueError('Original fixed frozen mean required')
    bindings={}
    for role,rr,aa in [('tree',report,root),('nn',model.preceding.REPORT,model.preceding.RUN)]:
        receipt=u.read(rr/(role+'.json'));replay=u.read(rr/(role+'-replay.json'));origin=u.sha(rr/'plan.json')
        if (receipt['plan_sha256']!=origin or replay['plan_sha256']!=origin or not replay['bitwise_equal'] or
            replay['family_receipt_sha256']!=u.sha(rr/(role+'.json')) or replay['prediction_sha256']!=receipt['outputs']['qualifying.f32']):raise ValueError('Actual original full-family replay required')
        expected=frozen['new_tree_replay_sha256'] if role=='tree' else frozen['reused_nn_replay_sha256']
        if expected!=u.sha(rr/(role+'-replay.json')):raise ValueError('Frozen family replay binding differs')
        for name,digest in receipt['outputs'].items():
            path=aa/role/name
            if u.sha(path)!=digest:raise ValueError('Saved model/full vector changed')
            bindings[str(path.relative_to(u.ROOT))]=digest
        for path in [rr/'plan.json',rr/(role+'.json'),rr/(role+'-replay.json')]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    a=np.fromfile(root/'tree/qualifying.f32',dtype='<f4');b=np.fromfile(model.preceding.RUN/'nn/qualifying.f32',dtype='<f4');final=np.fromfile(report/'qualifying-final.f32',dtype='<f4')
    if (a.shape!=(2817131,) or b.shape!=a.shape or final.shape!=a.shape or not np.isfinite(final).all() or
        not np.array_equal(((a.astype(np.float64)+b.astype(np.float64))/2).astype('<f4').view(np.uint32),final.view(np.uint32)) or
        u.sha(report/'qualifying-final.f32')!=frozen['frozen_prediction_sha256']):raise ValueError('Complete exact frozen mean differs')
    lp=model.preceding.REPORT/'recovered-audit/qual_ratings.csv.gz';payload=lp.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered labels required')
    with gzip.open(lp,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Complete canonical label domain differs')
    groups={}
    for value,count in [(0,1408789),(1,1408342)]:
        idx=np.flatnonzero(labels[:,1]==value)
        if len(idx)!=count:raise ValueError('Canonical group row count differs')
        groups['source'+str(value)]=dict(rows=count,rmse=math.sqrt(math.fsum((float(final[i])-float(labels[i,0]))**2 for i in idx)/count))
    model.core.verify()
    for path in [Path(__file__),report/'report.json',report/'qualifying-final.f32',lp]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    u.write(report/'independent-verification.json',dict(state='complete_independent_frozen_tree154_mean_and_scores',observed=model.binding.stamp(),groups=groups,
        both_groups_below_target=all(r['rmse']<.85725 for r in groups.values()),fixed_float32_mean_bitwise_equal=True,
        actual_new_tree_and_original_full_nn_replays_verified=True,label_git_blob=blob,prediction_sha256=u.sha(report/'qualifying-final.f32'),
        bindings=bindings,qualifying_labels_used_for_fitting_or_selection=False,
        scope='Complete known-CV-selected full tree154 and unchanged actually replayed full NN154. Separate exact stored mean and compensated complete canonical groups. Adaptive recovered audit, not official/untouched Test.'))
    print(dict(groups=groups,both_groups_below_target=all(r['rmse']<.85725 for r in groups.values())),flush=True)

if __name__=='__main__':verify()
