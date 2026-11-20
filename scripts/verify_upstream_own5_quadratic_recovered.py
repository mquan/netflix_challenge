#!/usr/bin/env python3
"""Independent direct SVD fit, exact quadratic vectors and canonical group scores."""
from pathlib import Path
import gzip
import hashlib
import math
import numpy as np
import upstream_own5_quadratic as experiment
parent=experiment.parent
u=parent.u

def independent_design(arrays,lo,hi,center,scale):
    last=arrays[-1][lo:hi].astype(np.float64)
    coordinates=[arrays[i][lo:hi].astype(np.float64)-last for i in range(4)]+[last-3.6]
    columns=coordinates[:]
    for i in range(5):
        for j in range(i,5):columns.append(coordinates[i]*coordinates[j])
    z=np.ones((hi-lo,21))
    for j in range(20):z[:,j+1]=(columns[j]-center[j])/scale[j]
    return z

def check_vector(path,arrays,offset,center,scale,beta):
    stored=np.memmap(path,dtype='<f4',mode='r')
    if len(stored)!=len(offset):raise ValueError('Stored row count differs')
    for lo in range(0,len(stored),4096):
        hi=min(lo+4096,len(stored));z=independent_design(arrays,lo,hi,center,scale);delta=np.zeros(hi-lo)
        for j in range(21):delta=np.add(delta,np.multiply(z[:,j],float(beta[j])))
        expected=np.clip(offset[lo:hi].astype(np.float64)+delta,1.05,4.95).astype('<f4')
        if not np.isfinite(expected).all() or not np.array_equal(expected.view(np.uint32),stored[lo:hi].view(np.uint32)):raise ValueError('Exact complete stored-vector reconstruction differs')
    return stored

def verify():
    report,root=experiment.REPORT,experiment.RUN
    if (report/'independent-verification.json').exists():raise ValueError('Existing independent proof preserved')
    plan=u.read(report/'plan.json');cal=u.read(report/'calibration.json');freeze=u.read(report/'frozen.json')
    if (cal['plan_sha256']!=u.sha(report/'plan.json') or freeze['plan_sha256']!=u.sha(report/'plan.json') or
        freeze['calibration_sha256']!=u.sha(report/'calibration.json') or freeze['qualifying_labels_read'] is not False or
        cal['known_score_is_nested_cross_validation'] is not False or cal['penalty']!=1000. or plan['penalty']!=1000.):raise ValueError('Original fixed known-only fit and freeze required')
    for name,sha in plan['code_sha256'].items():
        if u.sha(u.ROOT/name)!=sha:raise ValueError('Declared fitting/checking code changed')
    parent.verify_bindings(cal['source_sha256'])
    rows=cal['families']
    if [(r['id'],r['family'],r['development'],r['production']) for r in rows]!=parent.FAMILIES:raise ValueError('Fixed own five-family list differs')
    arrays=[np.memmap(u.ROOT/r['known_prediction'],dtype='<f4',mode='r') for r in rows]
    offset=np.memmap(parent.RUN/'known-fit.f32',dtype='<f4',mode='r');y=np.load(parent.LABELS).astype(np.float64)
    center=np.asarray(cal['center']);scale=np.asarray(cal['scale']);beta=np.asarray(cal['beta'])
    # Direct rectangular least-squares via SVD, independent of the producer's
    # chunked normal-equation solve. Additional rows implement its fixed ridge.
    matrix=independent_design(arrays,0,len(y),np.zeros(20),np.ones(20))
    raw=matrix[:,1:];means=raw.mean(axis=0);std=raw.std(axis=0)
    if np.max(np.abs(means-center))>1e-10 or np.max(np.abs(std-scale))>1e-10:raise ValueError('Independent population standardization differs')
    matrix[:,1:]=(raw-center)/scale
    regularizer=np.diag([0.]+[math.sqrt(1000.)]*20)
    solved,_,rank,_=np.linalg.lstsq(np.vstack([matrix,regularizer]),np.r_[y-offset.astype(np.float64),np.zeros(21)],rcond=None)
    if rank!=21 or np.max(np.abs(solved-beta))>1e-8:raise ValueError('Independent direct SVD solution differs')
    del matrix,raw
    known=check_vector(root/'known-fit.f32',arrays,offset,center,scale,beta)
    score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(known,y))/len(y))
    if abs(score-cal['known_fit_rmse'])>1e-12 or u.sha(root/'known-fit.f32')!=freeze['known_prediction_sha256']:raise ValueError('Original known-fit vector/score differs')
    full=[np.memmap(u.ROOT/r['qualifying_prediction'],dtype='<f4',mode='r') for r in rows]
    offset=np.memmap(parent.RUN/'qualifying.f32',dtype='<f4',mode='r')
    stored=check_vector(root/'qualifying.f32',full,offset,center,scale,beta)
    if u.sha(root/'qualifying.f32')!=freeze['prediction_sha256']:raise ValueError('Original frozen vector differs')
    lp=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz';payload=lp.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered labels differ')
    with gzip.open(lp,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',dtype=np.int16,skiprows=1)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Canonical full label domain differs')
    groups={}
    for group,count in [(0,1408789),(1,1408342)]:
        idx=np.flatnonzero(labels[:,1]==group)
        if len(idx)!=count:raise ValueError('Complete canonical group required')
        groups['source'+str(group)]=dict(rows=count,rmse=math.sqrt(math.fsum((float(stored[i])-float(labels[i,0]))**2 for i in idx)/count))
    prior=u.read(parent.REPORT/'independent-verification.json')['groups']
    parent.verify_bindings(cal['source_sha256'])
    paths=[report/'plan.json',report/'calibration.json',report/'frozen.json',root/'known-fit.f32',root/'qualifying.f32',lp,report/'verification-recovery.json',Path(__file__)]
    u.write(report/'independent-verification.json',dict(state='complete_independent_fixed_quadratic_fit_and_frozen_group_scores',observed=parent.stamp(),
        groups=groups,previous_groups=prior,both_groups_improved=all(groups[k]['rmse']<prior[k]['rmse'] for k in groups),
        both_groups_below_target=all(r['rmse']<.85725 for r in groups.values()),known_fit_rmse=score,
        known_score_is_nested_cross_validation=False,independent_direct_svd_rank=int(rank),independent_coefficient_max_error=float(np.max(np.abs(solved-beta))),
        exact_known_and_full_stored_vectors_bitwise=True,original_own_family_source_bindings_reverified=True,
        label_git_blob=blob,prediction_sha256=u.sha(root/'qualifying.f32'),qualifying_labels_used_for_fitting_or_selection=False,
        bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths},
        scope='Fixed21-coefficient quadratic residual ridge of five complete own families; no tuning. Separate full direct SVD solution, exact vector reconstruction and compensated complete-group scoring. Apparent known fit is not nested CV; recovered audit is adaptive, not official/untouched Test.'))
    print(dict(groups=groups,both_groups_improved=all(groups[k]['rmse']<prior[k]['rmse'] for k in groups)),flush=True)

if __name__=='__main__':
    recovery=u.read(experiment.REPORT/'verification-recovery.json')
    for name,digest in recovery['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original recovery evidence changed')
    verify()
