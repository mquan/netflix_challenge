#!/usr/bin/env python3
"""Independent coefficient fit, frozen-vector arithmetic and recovered group audit."""
from pathlib import Path
import gzip
import hashlib
import math
import numpy as np
import upstream_item154_weight_calibration as experiment
u=experiment.u

def verify():
    root,report=experiment.RUN,experiment.REPORT
    if (report/'independent-verification.json').exists():raise ValueError('Existing independent proof preserved')
    plan=u.read(report/'plan.json');cal=u.read(report/'calibration.json');frozen=u.read(report/'frozen.json')
    if (cal['state']!='complete_known_fit_single_convex_coefficient' or cal['score_is_nested_cross_validation'] is not False or
        frozen['state']!='complete_frozen_own_known_calibrated_prediction' or frozen['qualifying_labels_read'] is not False or
        frozen['plan_sha256']!=u.sha(report/'plan.json') or frozen['calibration_sha256']!=u.sha(report/'calibration.json')):
        raise ValueError('Original known-only coefficient and freeze required before audit')
    for name,digest in cal['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Own fitted family/source binding differs')
    y=np.load(experiment.LABELS).astype(np.float64);tree=np.fromfile(experiment.KNOWN_TREE,dtype='<f4').astype(np.float64);nn=np.fromfile(experiment.KNOWN_NN,dtype='<f4').astype(np.float64)
    d=tree-nn;independent=float(np.clip(np.dot(d,y-nn)/np.dot(d,d),0,1));weights=cal['weights_tree_nn']
    if len(weights)!=2 or min(weights)<0 or abs(sum(weights)-1)>1e-15 or abs(weights[0]-independent)>1e-11 or weights!=frozen['weights_tree_nn']:
        raise ValueError('Independent coefficient fit differs')
    known=(weights[0]*tree+weights[1]*nn).astype('<f4');stored_known=np.fromfile(root/'known-calibration.f32',dtype='<f4')
    if not np.array_equal(known.view(np.uint32),stored_known.view(np.uint32)) or u.sha(root/'known-calibration.f32')!=cal['known_prediction_sha256']:raise ValueError('Stored known calibration differs')
    fulltree=np.fromfile(experiment.FULL_TREE,dtype='<f4').astype(np.float64);fullnn=np.fromfile(experiment.FULL_NN,dtype='<f4').astype(np.float64)
    expected=(weights[0]*fulltree+weights[1]*fullnn).astype('<f4');stored=np.fromfile(root/'qualifying.f32',dtype='<f4')
    if not np.array_equal(expected.view(np.uint32),stored.view(np.uint32)) or u.sha(root/'qualifying.f32')!=frozen['prediction_sha256']:raise ValueError('Frozen actual full-query arithmetic differs')
    # Only after proving the stored vector matches the prior known-only freeze.
    labels_path=experiment.source.REPORT/'recovered-audit/qual_ratings.csv.gz';payload=labels_path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered labels differ')
    with gzip.open(labels_path,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',dtype=np.int16,skiprows=1)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],np.arange(1,6)).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Canonical label domain differs')
    groups={}
    for group,count in [(0,1408789),(1,1408342)]:
        ids=np.flatnonzero(labels[:,1]==group)
        if len(ids)!=count:raise ValueError('Canonical complete group count differs')
        rmse=math.sqrt(math.fsum((float(stored[i])-float(labels[i,0]))**2 for i in ids)/count)
        groups['source'+str(group)]=dict(rows=count,rmse=rmse)
    previous=u.read(experiment.source.REPORT/'independent-verification.json')['groups']
    for name,digest in cal['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source changed during audit')
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [report/'plan.json',report/'calibration.json',report/'frozen.json',root/'known-calibration.f32',root/'qualifying.f32',labels_path,Path(__file__)]}
    u.write(report/'independent-verification.json',dict(state='complete_independent_known_calibration_and_frozen_group_scores',observed=experiment.stamp(),
        weights_tree_nn=weights,groups=groups,previous_groups=previous,both_groups_improved=all(groups[k]['rmse']<previous[k]['rmse'] for k in groups),
        both_groups_below_target=all(r['rmse']<.85725 for r in groups.values()),known_fit_rmse=cal['known_fit_rmse'],known_score_is_nested_cross_validation=False,
        independently_refitted_coefficient=independent,exact_known_and_full_stored_vectors_bitwise=True,actual_full_family_replays_reused=True,
        label_git_blob=blob,prediction_sha256=u.sha(root/'qualifying.f32'),bindings=bindings,
        qualifying_labels_used_for_fitting_or_selection=False,scope='One coefficient fitted to own known OOF predictions; apparent calibration score is not nested CV. Existing actual replayed own full family models reused. Frozen vector independently reconstructed before canonical recovered group audit. Adaptive audit, not official/untouched Test.'))
    print(groups,flush=True)
if __name__=='__main__':verify()
