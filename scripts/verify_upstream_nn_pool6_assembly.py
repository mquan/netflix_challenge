#!/usr/bin/env python3
"""Independent127-support coefficient check, exact arithmetic, then frozen audit."""
import argparse
import gzip
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np
import upstream_nn_pool6_assembly as experiment

u=experiment.u

def simplex(arrays,y):
    size=len(arrays);reference=arrays[-1].astype(np.float64)
    columns=[a.astype(np.float64)-reference for a in arrays];residual=y-reference
    gram=np.empty((size,size));rhs=np.empty(size)
    for i in range(size):
        rhs[i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],residual))/len(y)
        for j in range(i+1):gram[i,j]=gram[j,i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],columns[j]))/len(y)
    candidates=[];examined=0
    for count in range(1,size+1):
        for support in itertools.combinations(range(size),count):
            examined+=1;ids=list(support);system=np.zeros((count+1,count+1))
            system[:count,:count]=gram[np.ix_(ids,ids)];system[:count,-1]=1;system[-1,:count]=1
            try:solution=np.linalg.solve(system,np.r_[rhs[ids],1.])
            except np.linalg.LinAlgError:continue
            weights=np.zeros(size);weights[ids]=solution[:-1]
            if weights.min()<-1e-9:continue
            gradient=gram@weights-rhs;inactive=[j for j in range(size) if j not in ids]
            if inactive and min(gradient[inactive]+solution[-1])<-1e-9:continue
            candidates.append((float(weights@gram@weights-2*rhs@weights),weights))
    if examined!=127 or not candidates:raise ValueError('Complete independent simplex solution missing')
    return min(candidates,key=lambda item:item[0])[1],gram,rhs

def stored(path,arrays,weights):
    actual=np.fromfile(path,dtype='<f4')
    if actual.shape!=arrays[0].shape:raise ValueError('Stored vector shape differs')
    for lo in range(0,len(actual),4096):
        hi=min(lo+4096,len(actual));expected=np.zeros(hi-lo,dtype=np.float64)
        for a,w in zip(arrays,weights):expected=expected+float(w)*a[lo:hi].astype(np.float64)
        if not np.array_equal(expected.astype('<f4').view('u4'),actual[lo:hi].view('u4')):raise ValueError('Exact stored arithmetic differs')
    return actual

def verify():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-preaudit-verification.json'
    if output.exists():raise ValueError('Existing independent preaudit proof preserved')
    plan=u.read(report/'plan.json');cal=u.read(report/'calibration.json');freeze=u.read(report/'frozen.json')
    if (freeze['state']!='complete_three_predeclared_own_candidates_frozen_before_audit' or
        freeze['plan_sha256']!=u.sha(report/'plan.json') or freeze['calibration_sha256']!=u.sha(report/'calibration.json') or
        freeze['qualifying_labels_read'] is not False or plan['verifier_sha256']!=u.sha(Path(__file__))):raise ValueError('Original freeze differs')
    original,new,kpool,sources=experiment.checked_sources()
    if original!=cal['families'] or new!=cal['new_neural_full_sources'] or sources!=cal['source_sha256']:raise ValueError('Independently checked original source family list differs')
    known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in original]+[np.fromfile(kpool,dtype='<f4')]
    y=np.load(experiment.own6.LABELS).astype(np.float64);w=np.asarray(cal['weights'])
    independent,gram,rhs=simplex(known,y)
    if w.shape!=(7,) or w.min()<0 or abs(w.sum()-1)>1e-12 or np.max(abs(independent-w))>1e-5:raise ValueError('Independent simplex coefficients differ')
    if np.max(abs(gram-np.asarray(cal['gram'])))>1e-12 or np.max(abs(rhs-np.asarray(cal['rhs'])))>1e-12:raise ValueError('Independent compensated statistics differ')
    fitted=stored(root/'known-fit.f32',known,w)
    neural=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in new]
    pooled=np.fromfile(root/'equal-six-nn.f32',dtype='<f4')
    # The declared pool adds all six in order, then divides once before f32.
    for lo in range(0,len(pooled),4096):
        hi=min(lo+4096,len(pooled));expected=(sum(a[lo:hi].astype(np.float64) for a in neural)/6).astype('<f4')
        if not np.array_equal(expected.view('u4'),pooled[lo:hi].view('u4')):raise ValueError('Equal pool arithmetic differs')
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in original]
    tree=full[next(i for i,r in enumerate(original) if r['id']=='tree154')]
    stored(root/'fixed-tree25-pool75.f32',[tree,pooled],[.25,.75])
    stored(root/'own7-known-calibration.f32',[*full,pooled],w)
    for name,digest in freeze['predictions'].items():
        if u.sha(root/(name+'.f32'))!=digest:raise ValueError('Frozen candidate changed')
    experiment.own6.verify_bindings(sources)
    apparent=math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(np.clip(fitted.astype(np.float64),1.05,4.95),y))/len(y))
    u.write(output,dict(state='complete_independent127_support_and_all_frozen_arithmetic_before_audit',
        observed=experiment.stamp(),weights=w.tolist(),independent_weights=independent.tolist(),simplex_supports_examined=127,
        known_apparent_fit_rmse=apparent,known_score_is_nested_cross_validation=False,
        all_stored_candidate_arithmetic_bitwise=True,qualifying_labels_read=False,
        freeze_sha256=u.sha(report/'frozen.json'),calibration_sha256=u.sha(report/'calibration.json'),
        verifier_sha256=u.sha(Path(__file__)),source_sha256=sources))
    print('Independent127-support calibration and all frozen arithmetic verified',flush=True)

def audit():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-recovered-audit.json'
    if output.exists():raise ValueError('Existing recovered audit preserved')
    pre=report/'independent-preaudit-verification.json';v=u.read(pre);freeze=u.read(report/'frozen.json')
    if (v['state']!='complete_independent127_support_and_all_frozen_arithmetic_before_audit' or
        v['freeze_sha256']!=u.sha(report/'frozen.json') or v['verifier_sha256']!=u.sha(Path(__file__)) or
        v['calibration_sha256']!=u.sha(report/'calibration.json') or
        freeze['calibration_sha256']!=u.sha(report/'calibration.json') or
        freeze['plan_sha256']!=u.sha(report/'plan.json') or
        v['qualifying_labels_read'] is not False):raise ValueError('Completed independent check before outcomes required')
    experiment.own6.verify_bindings(v['source_sha256'])
    predictions={}
    for name in ['equal-six-nn','fixed-tree25-pool75','own7-known-calibration']:
        p=root/(name+'.f32')
        if u.sha(p)!=freeze['predictions'][name]:raise ValueError('Frozen candidate changed')
        predictions[name]=np.fromfile(p,dtype='<f4')
        if predictions[name].shape!=(2817131,) or not np.isfinite(predictions[name]).all():raise ValueError('Incomplete candidate')
    lp=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload=lp.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered outcomes differ')
    with gzip.open(lp,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Outcome domain differs')
    results=[]
    for name,values in predictions.items():
        values=np.clip(values.astype(np.float64),1.05,4.95);groups={}
        for group,count in [(0,1408789),(1,1408342)]:
            ids=np.flatnonzero(labels[:,1]==group)
            if len(ids)!=count:raise ValueError('Complete recovered group count differs')
            rmse=math.sqrt(math.fsum(float(values[i]-labels[i,0])**2 for i in ids)/count)
            groups['source'+str(group)]=dict(rows=count,rmse=rmse,below_target=rmse<.85725)
        results.append(dict(candidate=name,groups=groups,both_groups_below_target=all(g['below_target'] for g in groups.values()),prediction_sha256=freeze['predictions'][name]))
    u.write(output,dict(state='complete_independent_frozen_three_candidate_recovered_audit',observed=experiment.stamp(),
        primary='own7-known-calibration',results=results,label_git_blob=blob,
        preaudit_verification_sha256=u.sha(pre),freeze_sha256=u.sha(report/'frozen.json'),
        qualifying_outcomes_used_for_fitting_or_selection=False,
        limitation='Adaptive community-recovered groups; no official or untouched Test score. Original A development and paired B production recipes remain distinct.'))
    print(results,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');a=p.parse_args()
    audit() if a.audit else verify()
