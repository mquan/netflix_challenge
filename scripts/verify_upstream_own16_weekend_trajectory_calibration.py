#!/usr/bin/env python3
"""Separate complete-support/statistic replay, then frozen-label audit/canonical."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np
import upstream_own16_weekend_trajectory_calibration as study

u=study.u
STATE='complete_independent_all65535_supports_ridge17_and_six_vectors_before_audit'


def products(columns,y):
    n=len(columns);gram=np.empty((n,n));rhs=np.empty(n)
    for i in range(n):
        rhs[i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],y))
        for j in range(i+1):gram[i,j]=gram[j,i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],columns[j]))
    return gram,rhs


def support_solution(gram,rhs):
    if gram.shape!=(16,16) or rhs.shape!=(16,):raise ValueError('All16 full families required')
    options=[];examined=0
    for count in range(1,17):
        for support in itertools.combinations(range(16),count):
            examined+=1;ids=list(support);system=np.zeros((count+1,count+1))
            system[:count,:count]=gram[np.ix_(ids,ids)];system[:count,-1]=1;system[-1,:count]=1
            try:solution=np.linalg.solve(system,np.r_[rhs[ids],1.])
            except np.linalg.LinAlgError:continue
            weights=np.zeros(16);weights[ids]=solution[:-1]
            if weights.min()<-1e-9:continue
            inactive=[j for j in range(16) if j not in ids];gradient=gram@weights-rhs
            if inactive and min(gradient[inactive]+solution[-1])<-1e-9:continue
            options.append((float(weights@gram@weights-2*rhs@weights),weights))
    if examined!=65535 or not options:raise ValueError('Full independent simplex supports missing')
    return min(options,key=lambda row:row[0])[1],examined


def verify_simplex_weights(gram,rhs,actual,independent):
    actual=np.asarray(actual,dtype=np.float64)
    if actual.shape!=(16,) or not np.isfinite(actual).all() or actual.min()<0 or abs(actual.sum()-1)>1e-12:
        raise ValueError('All16 actual finite nonnegative sum-one coefficients required')
    gradient=gram@actual-rhs;dual=-float(actual@gradient);reduced=gradient+dual
    value=float(actual@gram@actual-2*rhs@actual)
    optimum=float(independent@gram@independent-2*rhs@independent)
    gap=abs(value-optimum);complementarity=float(np.max(abs(actual*reduced)))
    if np.min(reduced)<-1e-9 or complementarity>1e-10 or gap>1e-12:
        raise ValueError('Actual coefficients must attain independently enumerated optimum and full KKT conditions')
    return dict(objective=value,independent_optimum=optimum,absolute_gap=gap,
                minimum_reduced_gradient=float(np.min(reduced)),complementarity_error=complementarity,
                nonunique_coefficients_allowed=True)


def stored(path,arrays,weights,intercept=0.):
    actual=np.fromfile(path,dtype='<f4')
    if actual.shape!=arrays[0].shape or not np.isfinite(actual).all():raise ValueError('Complete finite stored candidate required')
    for lo in range(0,len(actual),4096):
        hi=min(lo+4096,len(actual));expected=np.full(hi-lo,float(intercept),dtype=np.float64)
        for column,weight in zip(arrays,weights):expected=expected+float(weight)*column[lo:hi].astype(np.float64)
        expected=np.asarray(np.clip(expected,1.05,4.95),dtype='<f4')
        if not np.array_equal(actual[lo:hi].view('u4'),expected.view('u4')):raise ValueError('Every actual stored arithmetic bit must replay')
    return actual


def evidence():
    p,c,f=map(u.read,[study.REPORT/name for name in ['plan.json','calibration.json','frozen.json']])
    if (p['primary']!=study.CANDIDATES[0] or p['candidates']!=study.CANDIDATES or p['families']!=16 or p['ridge_lambda']!=10. or
        p['script_sha256']!=study.sha(Path(study.__file__)) or p['verifier_sha256']!=study.sha(Path(__file__)) or
        p['declaration_sha256']!=study.sha(study.DECL) or p['qualifying_labels_read'] is not False or
        f['state']!='complete_all_three_own16_candidates_frozen_before_any_new_audit' or f['primary']!=study.CANDIDATES[0] or
        f['plan_sha256']!=study.sha(study.REPORT/'plan.json') or f['calibration_sha256']!=study.sha(study.REPORT/'calibration.json') or
        f['known_completion_sha256']!=study.sha(study.REPORT/'known-completion.json') or f['qualifying_labels_read'] is not False or
        c['qualifying_labels_read'] is not False or c['known_score_is_nested_cross_validation'] is not False or
        c['family_ids']!=study.BASE_IDS+['nn-weekend-trajectory-fold-bag'] or [r['id'] for r in c['families']]!=c['family_ids'] or
        any(f['source_sha256'].get(k)!=v for k,v in c['source_sha256'].items())):
        raise ValueError('Original complete16 known-only recipe/freeze required')
    if set(f['predictions'])!=set(study.CANDIDATES+['known-simplex','known-ridge10','known-fixed-half']):
        raise ValueError('All six originally declared frozen vectors required')
    for name,digest in f['predictions'].items():
        if study.sha(study.RUN/(name+'.f32'))!=digest:raise ValueError('Original frozen prediction changed')
    study.verify_bindings(f['source_sha256'])
    return p,c,f


def verify():
    output=study.REPORT/'independent-preaudit-verification.json'
    if output.exists():raise ValueError('Existing full independent own16 preaudit retained')
    p,c,f=evidence();rows=c['families'];known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows]
    if any(a.shape!=(1128164,) or not np.isfinite(a).all() for a in known):raise ValueError('All complete known OOF families required')
    y=np.load(study.KNOWN/'labels.npy').astype(np.float64);reference=known[-1].astype(np.float64)
    gram,rhs=products([a.astype(np.float64)-reference for a in known],y-reference);gram/=len(y);rhs/=len(y)
    weights,count=support_solution(gram,rhs);actual=np.asarray(c['simplex_weights'])
    simplex_check=verify_simplex_weights(gram,rhs,actual,weights)
    if (np.max(abs(gram-np.asarray(c['simplex_gram'])))>1e-12 or np.max(abs(rhs-np.asarray(c['simplex_rhs'])))>1e-12):
        raise ValueError('All65535 support/compensated simplex statistics must match')
    raw,target=products([a.astype(np.float64) for a in known]+[np.ones(len(y))],y)
    penalty=np.eye(17)*10.;penalty[-1,-1]=0;ridge=np.r_[c['ridge_weights'],c['ridge_intercept']]
    independent_ridge=np.linalg.solve(raw+penalty,target)
    if (c['ridge_lambda']!=10. or ridge.shape!=(17,) or not np.isfinite(ridge).all() or
        np.max(abs(raw-np.asarray(c['ridge_gram'])))/len(y)>1e-12 or np.max(abs(target-np.asarray(c['ridge_rhs'])))/len(y)>1e-12 or
        np.max(abs(ridge-independent_ridge))>1e-7 or np.max(abs((raw+penalty)@ridge-target))/len(y)>1e-12):
        raise ValueError('Complete independent17-dimensional released-default ridge must match')
    fitted=[stored(study.RUN/'known-simplex.f32',known,actual),stored(study.RUN/'known-ridge10.f32',known,ridge[:-1],ridge[-1]),
        stored(study.RUN/'known-fixed-half.f32',[known[-1],known[5]],[.5,.5])]
    if study.sha(study.RUN/'known-fixed-half.f32')!=study.sha(study.KNOWN/'fixed-half-tree154.f32'):
        raise ValueError('Original canonical known fixed half must remain exact')
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
    if any(a.shape!=(2817131,) or not np.isfinite(a).all() for a in full):raise ValueError('Every complete qualifying component required')
    stored(study.RUN/(study.CANDIDATES[0]+'.f32'),full,actual)
    stored(study.RUN/(study.CANDIDATES[1]+'.f32'),full,ridge[:-1],ridge[-1])
    stored(study.RUN/(study.CANDIDATES[2]+'.f32'),[full[-1],full[5]],[.5,.5])
    study.verify_bindings(f['source_sha256'])
    scores=[math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(v.astype(np.float64),1.05,4.95),y))/len(y)) for v in fitted]
    u.write(output,dict(state=STATE,observed=study.stamp(),freeze_sha256=study.sha(study.REPORT/'frozen.json'),
        calibration_sha256=study.sha(study.REPORT/'calibration.json'),verifier_sha256=study.sha(Path(__file__)),
        simplex_supports_examined=count,independent_simplex_weights=weights.tolist(),actual_simplex_optimality=simplex_check,ridge_equations=17,ridge_lambda=10.,
        independent_ridge_coefficients=independent_ridge.tolist(),all_six_candidate_vectors_bitwise=True,
        known_apparent_fit_rmse=scores,known_score_is_nested_cross_validation=False,source_sha256=f['source_sha256'],qualifying_labels_read=False))
    print('Own16 all65535 supports/full17dim ridge/all six vectors verified',flush=True)


def require_preaudit():
    p,c,f=evidence();pre=u.read(study.REPORT/'independent-preaudit-verification.json')
    if (pre['state']!=STATE or pre['freeze_sha256']!=study.sha(study.REPORT/'frozen.json') or
        pre['calibration_sha256']!=study.sha(study.REPORT/'calibration.json') or pre['verifier_sha256']!=study.sha(Path(__file__)) or
        pre['simplex_supports_examined']!=65535 or pre['ridge_equations']!=17 or pre['ridge_lambda']!=10. or
        not pre['all_six_candidate_vectors_bitwise'] or pre['qualifying_labels_read'] is not False or
        pre['known_score_is_nested_cross_validation'] is not False):raise ValueError('Complete original preaudit required before labels')
    return p,c,f,pre


def labels():
    import gzip
    path=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered payload differs')
    with gzip.open(path,'rt') as stream:value=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if value.shape!=(2817131,2) or not np.isin(value[:,0],[1,2,3,4,5]).all() or not np.isin(value[:,1],[0,1]).all():
        raise ValueError('Full canonical recovered domain required')
    return value,blob


def audit():
    output=study.REPORT/'independent-recovered-audit.json'
    if output.exists():raise ValueError('Existing original own16 audit retained')
    p,c,f,pre=require_preaudit();y,blob=labels();results=[]
    for name in study.CANDIDATES:
        values=np.clip(np.fromfile(study.RUN/(name+'.f32'),dtype='<f4').astype(np.float64),1.05,4.95);groups={}
        for group,count in [(0,1408789),(1,1408342)]:
            ids=np.flatnonzero(y[:,1]==group)
            if len(ids)!=count:raise ValueError('Exact full group count required')
            score=math.sqrt(math.fsum((float(values[i])-float(y[i,0]))**2 for i in ids)/count)
            groups['source'+str(group)]=dict(rows=count,rmse=score)
        results.append(dict(candidate=name,groups=groups,prediction_sha256=f['predictions'][name],
            both_groups_below_target=all(g['rmse']<.85725 for g in groups.values())))
    require_preaudit()
    u.write(output,dict(state='complete_all_three_original_own16_frozen_candidate_audits',observed=study.stamp(),primary=p['primary'],results=results,
        preaudit_verification_sha256=study.sha(study.REPORT/'independent-preaudit-verification.json'),label_git_blob=blob,
        qualifying_labels_used_for_fitting_or_selection=False,official_or_untouched_test=False))


def canonical():
    output=study.REPORT/'independent-canonical-verification.json'
    if output.exists():raise ValueError('Existing separate canonical own16 evidence retained')
    p,c,f,pre=require_preaudit();aud=u.read(study.REPORT/'independent-recovered-audit.json')
    if (aud['state']!='complete_all_three_original_own16_frozen_candidate_audits' or aud['primary']!=p['primary'] or
        aud['preaudit_verification_sha256']!=study.sha(study.REPORT/'independent-preaudit-verification.json') or
        aud['qualifying_labels_used_for_fitting_or_selection'] is not False):raise ValueError('Original fixed candidate audit required')
    y,blob=labels();full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in c['families']]
    specs=[(study.CANDIDATES[0],full,c['simplex_weights'],0.),(study.CANDIDATES[1],full,c['ridge_weights'],c['ridge_intercept']),
        (study.CANDIDATES[2],[full[-1],full[5]],[.5,.5],0.)];results=[]
    for name,arrays,weights,intercept in specs:
        values=stored(study.RUN/(name+'.f32'),arrays,weights,intercept).astype(np.float64);values=np.clip(values,1.05,4.95)
        audited=next(r for r in aud['results'] if r['candidate']==name);groups={}
        for group,count in [(0,1408789),(1,1408342)]:
            ids=np.flatnonzero(y[:,1]==group)
            if len(ids)!=count:raise ValueError('Complete canonical population required')
            partial=[]
            for lo in range(0,count,16384):
                selected=ids[lo:lo+16384];errors=values[selected]-y[selected,0]
                partial.append(math.fsum(float(e)*float(e) for e in errors))
            score=math.sqrt(math.fsum(partial)/count);key='source'+str(group)
            if abs(score-audited['groups'][key]['rmse'])>1e-12:raise ValueError('Separate canonical compensated score differs')
            groups[key]=dict(rows=count,rmse=score)
        results.append(dict(candidate=name,groups=groups,prediction_sha256=f['predictions'][name],both_groups_below_target=all(g['rmse']<.85725 for g in groups.values())))
    study.verify_bindings(f['source_sha256']);primary=results[0]
    u.write(output,dict(state='complete_independent_own16_full_source_models_coefficients_and_frozen_group_scores',observed=study.stamp(),
        primary=p['primary'],groups=primary['groups'],results=results,prediction_sha256=primary['prediction_sha256'],
        both_groups_below_target=primary['both_groups_below_target'],simplex_supports_examined=65535,ridge_equations=17,
        all_six_vectors_independently_verified=True,full_inherited_source_closure_rehashed=True,
        actual_new_qualifying_models_reopened=u.read(study.BAG/'independent-verification.json')['actual_models_reopened'],source_sha256=f['source_sha256'],label_git_blob=blob,
        qualifying_labels_used_for_fitting_or_selection=False,known_score_is_nested_cross_validation=False,official_or_untouched_test=False,
        bindings={str((study.REPORT/name).relative_to(u.ROOT)):study.sha(study.REPORT/name) for name in
            ['plan.json','calibration.json','known-completion.json','frozen.json','independent-preaudit-verification.json','independent-recovered-audit.json']}))
    print('Own16 separately canonical primary:',primary['groups'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preaudit','audit','canonical']);args=parser.parse_args()
    {'preaudit':verify,'audit':audit,'canonical':canonical}[args.phase]()
