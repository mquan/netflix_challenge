#!/usr/bin/env python3
"""Independent simplex solution, exact vector reconstruction and recovered-group audit."""
from pathlib import Path
import gzip
import hashlib
import itertools
import math
import numpy as np
import upstream_own6_tight_calibration as experiment
u=experiment.u

def independent_weights(arrays,y):
    columns=[a.astype(np.float64)-arrays[-1].astype(np.float64) for a in arrays];r=y-arrays[-1]
    gram=np.empty((6,6));rhs=np.empty(6)
    for i in range(6):
        rhs[i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],r))/len(y)
        for j in range(i+1):gram[i,j]=gram[j,i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],columns[j]))/len(y)
    candidates=[]
    for size in range(1,7):
        for active in itertools.combinations(range(6),size):
            ids=list(active);g=gram[np.ix_(ids,ids)];system=np.zeros((size+1,size+1));system[:size,:size]=g;system[:size,-1]=1;system[-1,:size]=1
            try:solution=np.linalg.solve(system,np.r_[rhs[ids],1.])
            except np.linalg.LinAlgError:continue
            w=np.zeros(6);w[ids]=solution[:-1]
            if w.min()<-1e-9:continue
            gradient=gram@w-rhs;inactive=[j for j in range(6) if j not in ids]
            if inactive and min(gradient[inactive]+solution[-1])<-1e-9:continue
            candidates.append((float(w@gram@w-2*rhs@w),w))
    if not candidates:raise ValueError('Independent constrained quadratic solution missing')
    return min(candidates,key=lambda r:r[0])[1],gram,rhs

def check_vector(path,arrays,w):
    stored=np.fromfile(path,dtype='<f4')
    if len(stored)!=len(arrays[0]):raise ValueError('Stored vector shape differs')
    for lo in range(0,len(stored),4096):
        hi=min(lo+4096,len(stored));expected=np.zeros(hi-lo,dtype=np.float64)
        for j in range(6):expected=expected+float(w[j])*np.asarray(arrays[j][lo:hi],dtype=np.float64)
        if not np.array_equal(expected.astype('<f4').view(np.uint32),stored[lo:hi].view(np.uint32)):raise ValueError('Every exact stored vector value must reconstruct')
    return stored

def verify():
    report,root=experiment.REPORT,experiment.RUN
    if (report/'independent-verification.json').exists():raise ValueError('Existing independent proof preserved')
    plan=u.read(report/'plan.json');cal=u.read(report/'calibration.json');freeze=u.read(report/'frozen.json')
    if (cal['state']!='complete_own_six_family_known_calibration' or freeze['state']!='complete_own_six_family_prediction_frozen_before_audit' or
        freeze['plan_sha256']!=u.sha(report/'plan.json') or freeze['calibration_sha256']!=u.sha(report/'calibration.json') or
        freeze['qualifying_labels_read'] is not False or cal['known_score_is_nested_cross_validation'] is not False or freeze['weights']!=cal['weights']):raise ValueError('Prior own known-only calibration and freeze required')
    experiment.verify_bindings(cal['source_sha256']);rows=cal['families']
    if [(r['id'],r['family'],r['development'],r['production']) for r in rows]!=experiment.FAMILIES:raise ValueError('Predeclared complete six-family list differs')
    known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows];y=np.load(experiment.LABELS).astype(np.float64);w=np.asarray(cal['weights'])
    exact,gram,rhs=independent_weights(known,y)
    if len(w)!=6 or min(w)<0 or abs(sum(w)-1)>1e-12 or np.max(np.abs(w-exact))>1e-5:raise ValueError('Independent global simplex solution differs')
    if np.max(np.abs(gram-np.asarray(cal['gram'])))>1e-12 or np.max(np.abs(rhs-np.asarray(cal['rhs'])))>1e-12:raise ValueError('Independent sufficient statistics differ')
    kp=check_vector(root/'known-fit.f32',known,w)
    score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(kp,y))/len(y))
    if abs(score-cal['known_fit_rmse'])>1e-12 or u.sha(root/'known-fit.f32')!=cal['known_prediction_sha256']:raise ValueError('Apparent known-fit score differs')
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows];stored=check_vector(root/'qualifying.f32',full,w)
    if u.sha(root/'qualifying.f32')!=freeze['prediction_sha256']:raise ValueError('Original frozen vector differs')
    lp=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz';payload=lp.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered labels differ')
    with gzip.open(lp,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',dtype=np.int16,skiprows=1)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Canonical complete label domain differs')
    groups={}
    for group,count in [(0,1408789),(1,1408342)]:
        idx=np.flatnonzero(labels[:,1]==group)
        if len(idx)!=count:raise ValueError('Complete recovered group count differs')
        rmse=math.sqrt(math.fsum((float(stored[i])-float(labels[i,0]))**2 for i in idx)/count);groups['source'+str(group)]=dict(rows=count,rmse=rmse)
    prior=u.read(u.ROOT/'experiments/upstream-own5-calibration/independent-verification.json')['groups']
    experiment.verify_bindings(cal['source_sha256'])
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [report/'plan.json',report/'calibration.json',report/'frozen.json',root/'known-fit.f32',root/'qualifying.f32',lp,Path(__file__)]}
    u.write(report/'independent-verification.json',dict(state='complete_independent_own6_calibration_and_frozen_group_scores',observed=experiment.stamp(),weights=w.tolist(),independent_simplex_weights=exact.tolist(),
        groups=groups,previous_groups=prior,both_groups_improved=all(groups[k]['rmse']<prior[k]['rmse'] for k in groups),both_groups_below_target=all(r['rmse']<.85725 for r in groups.values()),
        known_fit_rmse=score,known_score_is_nested_cross_validation=False,exact_known_and_full_stored_vectors_bitwise=True,all_original_development_and_full_family_replays_verified=True,
        independently_solved_full_simplex=True,label_git_blob=blob,prediction_sha256=u.sha(root/'qualifying.f32'),bindings=bindings,qualifying_labels_used_for_fitting_or_selection=False,
        scope='Six fully replayed own families, one known-only convex coefficient fit. Independent active-set solution and compensated statistics, exact frozen vectors, full canonical recovered groups. Apparent calibration is not nested CV; adaptive audit is not official/untouched Test.'))
    print(dict(groups=groups,both_groups_improved=all(groups[k]['rmse']<prior[k]['rmse'] for k in groups)),flush=True)
if __name__=='__main__':verify()
