#!/usr/bin/env python3
"""Independent complete-support/normal-equation replay before recovered audit."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
import upstream_complete13_calibration as experiment

u=experiment.u
STATE='complete_independent8191_support_and_ridge10_all_frozen_arithmetic_before_audit'


def simplex(arrays,y):
    size=len(arrays)
    if size!=13:raise ValueError('All13 original families required')
    reference=arrays[-1].astype(np.float64)
    columns=[a.astype(np.float64)-reference for a in arrays];residual=y-reference
    gram=np.empty((size,size));rhs=np.empty(size)
    for i in range(size):
        rhs[i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],residual))/len(y)
        for j in range(i+1):
            gram[i,j]=gram[j,i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],columns[j]))/len(y)
    candidates=[];examined=0
    for count in range(1,size+1):
        for support in itertools.combinations(range(size),count):
            examined+=1;ids=list(support);system=np.zeros((count+1,count+1))
            system[:count,:count]=gram[np.ix_(ids,ids)];system[:count,-1]=1;system[-1,:count]=1
            try:solution=np.linalg.solve(system,np.r_[rhs[ids],1.])
            except np.linalg.LinAlgError:continue
            weights=np.zeros(size);weights[ids]=solution[:-1]
            if weights.min()<-1e-9:continue
            inactive=[j for j in range(size) if j not in ids];gradient=gram@weights-rhs
            if inactive and min(gradient[inactive]+solution[-1])<-1e-9:continue
            candidates.append((float(weights@gram@weights-2*rhs@weights),weights))
    if examined!=8191 or not candidates:raise ValueError('Complete independent simplex solution missing')
    return min(candidates,key=lambda v:v[0])[1],gram,rhs


def stored(path,arrays,weights,intercept=0.):
    actual=np.fromfile(path,dtype='<f4')
    if actual.shape!=arrays[0].shape or not np.isfinite(actual).all():raise ValueError('Complete finite stored vector required')
    for lo in range(0,len(actual),4096):
        hi=min(lo+4096,len(actual));expected=np.full(hi-lo,float(intercept),dtype=np.float64)
        for a,w in zip(arrays,weights):expected=expected+float(w)*a[lo:hi].astype(np.float64)
        expected=np.clip(expected,1.05,4.95).astype('<f4')
        if not np.array_equal(expected.view('u4'),actual[lo:hi].view('u4')):raise ValueError('Exact stored arithmetic differs')
    return actual


def verify():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-preaudit-verification.json'
    if output.exists():raise ValueError('Existing independent evidence retained')
    plan=u.read(report/'plan.json');cal=u.read(report/'calibration.json');frozen=u.read(report/'frozen.json')
    if (frozen['state']!='complete_both_complete13_candidates_frozen_before_any_new_audit' or
        frozen['plan_sha256']!=u.sha(report/'plan.json') or frozen['calibration_sha256']!=u.sha(report/'calibration.json') or
        frozen['qualifying_labels_read'] is not False or plan['candidates']!=experiment.CANDIDATES or
        plan['script_sha256']!=u.sha(Path(experiment.__file__)) or plan['verifier_sha256']!=u.sha(Path(__file__)) or
        plan['declaration_sha256']!=u.sha(u.ROOT/'ALL_COMPLETE13_CALIBRATION_PLAN.md') or plan['ridge_lambda']!=10.):
        raise ValueError('Original complete13 freeze required')
    rows,sources,aliases=experiment.checked_sources()
    if rows!=cal['families'] or sources!=cal['source_sha256'] or aliases!=cal['aliases']:
        raise ValueError('Original source/component expectations differ')
    known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows]
    y=np.load(experiment.own6.LABELS).astype(np.float64)
    independent,gram,rhs=simplex(known,y);weights=np.asarray(cal['simplex_weights'])
    if (weights.shape!=(13,) or weights.min()<0 or abs(weights.sum()-1)>1e-12 or
        np.max(abs(weights-independent))>1e-5 or np.max(abs(gram-np.asarray(cal['simplex_gram'])))>1e-12 or
        np.max(abs(rhs-np.asarray(cal['simplex_rhs'])))>1e-12):raise ValueError('Independent all8191-support solution differs')
    columns=[a.astype(np.float64) for a in known]+[np.ones(len(y))]
    raw=np.empty((14,14));target=np.empty(14)
    for i in range(14):
        target[i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],y))
        for j in range(i+1):raw[i,j]=raw[j,i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],columns[j]))
    penalty=np.eye(14)*10.;penalty[-1,-1]=0
    ridge=np.r_[cal['ridge_weights'],cal['ridge_intercept']]
    independent_ridge=np.linalg.solve(raw+penalty,target)
    if (cal['ridge_lambda']!=10. or ridge.shape!=(14,) or not np.isfinite(ridge).all() or
        np.max(abs(raw-np.asarray(cal['ridge_gram'])))/len(y)>1e-12 or
        np.max(abs(target-np.asarray(cal['ridge_rhs'])))/len(y)>1e-12 or
        np.max(abs(ridge-independent_ridge))>1e-7 or
        np.max(abs((raw+penalty)@ridge-target))/len(y)>1e-12):
        raise ValueError('Independent full14-dimensional released-default ridge solution differs')
    fitted=[stored(root/'known-simplex.f32',known,weights),stored(root/'known-ridge10.f32',known,ridge[:-1],ridge[-1])]
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
    stored(root/(experiment.CANDIDATES[0]+'.f32'),full,weights)
    stored(root/(experiment.CANDIDATES[1]+'.f32'),full,ridge[:-1],ridge[-1])
    for name,digest in frozen['predictions'].items():
        if u.sha(root/(name+'.f32'))!=digest:raise ValueError('Original frozen vector changed')
    experiment.own6.verify_bindings(sources)
    scores=[math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(v.astype(np.float64),1.05,4.95),y))/len(y)) for v in fitted]
    u.write(output,dict(state=STATE,observed=experiment.own6.stamp(),freeze_sha256=u.sha(report/'frozen.json'),
        calibration_sha256=u.sha(report/'calibration.json'),verifier_sha256=u.sha(Path(__file__)),
        simplex_supports_examined=8191,independent_simplex_weights=independent.tolist(),
        independent_ridge_coefficients=independent_ridge.tolist(),ridge_equations=14,ridge_lambda=10.,
        known_apparent_fit_rmse=scores,known_score_is_nested_cross_validation=False,
        all_stored_candidate_arithmetic_bitwise=True,source_sha256=sources,qualifying_labels_read=False))
    print('All8191 supports, full14dim ridge, and all four vectors verified',flush=True)


def require_paired_preaudits():
    other=experiment.current.REPORT
    for directory,state in [(experiment.REPORT,STATE),(other,'complete_independent_own352_foldbag_127_support_and_all_frozen_arithmetic_before_audit')]:
        proof=u.read(directory/'independent-preaudit-verification.json');freeze=u.read(directory/'frozen.json');plan=u.read(directory/'plan.json')
        if (proof['state']!=state or proof['qualifying_labels_read'] is not False or
            proof['freeze_sha256']!=u.sha(directory/'frozen.json') or
            proof['calibration_sha256']!=u.sha(directory/'calibration.json') or
            freeze['plan_sha256']!=u.sha(directory/'plan.json') or freeze['calibration_sha256']!=u.sha(directory/'calibration.json') or
            proof['verifier_sha256']!=plan['verifier_sha256']):raise ValueError('Both original studies must freeze and preverify before any new outcome')
        experiment.own6.verify_bindings(proof['source_sha256'])


def audit():
    import gzip
    import hashlib
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-recovered-audit.json'
    if output.exists():raise ValueError('Existing recovered evidence retained')
    require_paired_preaudits()
    frozen=u.read(report/'frozen.json');predictions={}
    for name in experiment.CANDIDATES:
        path=root/(name+'.f32')
        if u.sha(path)!=frozen['predictions'][name]:raise ValueError('Frozen vector changed')
        values=np.fromfile(path,dtype='<f4')
        if values.shape!=(2817131,) or not np.isfinite(values).all():raise ValueError('Complete finite candidate required')
        predictions[name]=np.clip(values.astype(np.float64),1.05,4.95)
    path=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered payload differs')
    with gzip.open(path,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Canonical recovered domain differs')
    results=[]
    for name,values in predictions.items():
        groups={}
        for group,count in [(0,1408789),(1,1408342)]:
            ids=np.flatnonzero(labels[:,1]==group)
            if len(ids)!=count:raise ValueError('Complete canonical group count differs')
            score=math.sqrt(math.fsum((float(values[i])-float(labels[i,0]))**2 for i in ids)/count)
            groups['source'+str(group)]=dict(rows=count,rmse=score,below_target=score<.85725)
        results.append(dict(candidate=name,groups=groups,both_groups_below_target=all(v['below_target'] for v in groups.values()),prediction_sha256=frozen['predictions'][name]))
    require_paired_preaudits()
    u.write(output,dict(state='complete_independent_two_frozen_complete13_candidate_recovered_audit',observed=experiment.own6.stamp(),
        primary=experiment.CANDIDATES[0],results=results,label_git_blob=blob,
        preaudit_verification_sha256=u.sha(report/'independent-preaudit-verification.json'),freeze_sha256=u.sha(report/'frozen.json'),
        qualifying_outcomes_used_for_fitting_or_selection=False,limitation='Adaptive community-recovered groups, not official or untouched Test. Development A and production B remain distinct.'))
    print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    audit() if args.audit else verify()
