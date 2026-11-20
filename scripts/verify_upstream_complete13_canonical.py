#!/usr/bin/env python3
"""Separate all-four-vector reconstruction and compensated canonical group check."""
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_complete13_calibration as thirteen

u=thirteen.u


def main():
    studies=[(thirteen.current,False),(thirteen,True)];bindings={};work=[]
    for study,wide in studies:
        report=study.REPORT
        if (report/'independent-canonical-verification.json').exists():raise ValueError('Existing separate canonical proof retained')
        paths={name:report/(name+'.json') for name in ['plan','calibration','frozen','independent-preaudit-verification','independent-recovered-audit']}
        p,c,f,pre,a=[u.read(paths[name]) for name in paths]
        expected_pre=('complete_independent8191_support_and_ridge10_all_frozen_arithmetic_before_audit' if wide else
            'complete_independent_own352_foldbag_127_support_and_all_frozen_arithmetic_before_audit')
        expected_audit=('complete_independent_two_frozen_complete13_candidate_recovered_audit' if wide else
            'complete_independent_two_frozen_own352_foldbag_candidate_recovered_audit')
        if (pre['state']!=expected_pre or a['state']!=expected_audit or p['candidates']!=study.CANDIDATES or
            f['plan_sha256']!=u.sha(paths['plan']) or f['calibration_sha256']!=u.sha(paths['calibration']) or
            pre['freeze_sha256']!=u.sha(paths['frozen']) or pre['calibration_sha256']!=u.sha(paths['calibration']) or
            a['preaudit_verification_sha256']!=u.sha(paths['independent-preaudit-verification']) or
            a['freeze_sha256']!=u.sha(paths['frozen']) or pre['verifier_sha256']!=p['verifier_sha256'] or
            pre['qualifying_labels_read'] is not False or f['qualifying_labels_read'] is not False or
            c['qualifying_labels_read'] is not False or a['qualifying_outcomes_used_for_fitting_or_selection'] is not False or
            pre['source_sha256']!=c['source_sha256'] or pre['simplex_supports_examined']!=(8191 if wide else 127) or
            not pre['all_stored_candidate_arithmetic_bitwise']):raise ValueError('Original independent all-candidate freeze/coefficient/arithmetic/audit chain required')
        for key,value in c['source_sha256'].items():
            if key in bindings and bindings[key]!=value:raise ValueError('Original component source expectations conflict')
            bindings[key]=value
        for path in paths.values():
            key=str(path.relative_to(u.ROOT));digest=u.sha(path)
            if key in bindings and bindings[key]!=digest:raise ValueError('Original proof expectation conflicts')
            bindings[key]=digest
        work.append((study,wide,p,c,f,pre,a,paths))
    bindings[str(Path(__file__).relative_to(u.ROOT))]=u.sha(Path(__file__))
    prior_path=u.ROOT/'experiments/upstream-own330-assembly/independent-canonical-verification.json'
    prior=u.read(prior_path)
    if prior['state']!='complete_independent_own330_calibration_and_frozen_group_scores':raise ValueError('Original preceding canonical benchmark required')
    bindings[str(prior_path.relative_to(u.ROOT))]=u.sha(prior_path)
    replay_path=thirteen.current.full.REPORT/'independent-verification.json';replay=u.read(replay_path)
    if (replay['state']!='complete_independent_all15_original_own352_fold_qualifying_models_and_uniform_bag' or
        replay['original_models_reopened']!=15 or not replay['all428_578_input_operators_independently_reconstructed'] or
        not replay['uniform_five_fold_bag_bitwise_equal'] or replay['qualifying_labels_read'] is not False or
        [r['fold'] for r in replay['records']]!=list(range(5)) or
        not all(r['actual_models_reopened']==3 and r['rows']==2817131 and r['every_qualifying_prediction_bit_equal'] for r in replay['records'])):
        raise ValueError('Original actual all15 qualifying model/input/bag replay required')
    thirteen.own6.verify_bindings(bindings)
    label_path=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload=label_path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical fixed recovered blob differs')
    with gzip.open(label_path,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',dtype=np.int16,skiprows=1)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Complete canonical recovered domain required')
    pending=[]
    for study,wide,p,c,f,pre,a,paths in work:
        rows=c['families'];weights=np.asarray(c['simplex_weights'] if wide else c['weights'])
        independent=np.asarray(pre['independent_simplex_weights'] if wide else pre['independent_weights'])
        if (len(rows)!=(13 if wide else 7) or weights.shape!=(len(rows),) or weights.min()<0 or
            abs(weights.sum()-1)>1e-12 or np.max(abs(weights-independent))>1e-5 or
            a['primary']!=p['primary'] or p['primary']!=study.CANDIDATES[0 if wide else 1]):raise ValueError('Original declared primary and all-support weights required')
        known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows]
        full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
        if any(v.shape!=(1128164,) or not np.isfinite(v).all() for v in known) or any(v.shape!=(2817131,) or not np.isfinite(v).all() for v in full):
            raise ValueError('Every complete finite component required')
        tree=next(i for i,r in enumerate(rows) if r['id']=='tree154')
        if wide:
            ridge=np.r_[c['ridge_weights'],c['ridge_intercept']]
            if c['ridge_lambda']!=10. or pre['ridge_equations']!=14 or np.max(abs(ridge-np.asarray(pre['independent_ridge_coefficients'])))>1e-7:
                raise ValueError('Independently checked full released-default ridge required')
            specifications=[('known-simplex',known,weights,0.),('known-ridge10',known,ridge[:-1],ridge[-1]),
                (study.CANDIDATES[0],full,weights,0.),(study.CANDIDATES[1],full,ridge[:-1],ridge[-1])]
        else:
            specifications=[('known-fit',known,weights,0.),
                (study.CANDIDATES[0],[full[-1],full[tree]],[.5,.5],0.),(study.CANDIDATES[1],full,weights,0.)]
        if set(f['predictions'])!={r[0] for r in specifications}:raise ValueError('All declared frozen vectors required')
        results=[]
        for name,arrays,coefficients,intercept in specifications:
            path=study.RUN/(name+'.f32')
            if u.sha(path)!=f['predictions'][name]:raise ValueError('Original frozen vector changed')
            stored=np.fromfile(path,dtype='<f4')
            if stored.shape!=arrays[0].shape or not np.isfinite(stored).all():raise ValueError('Complete finite vector required')
            for lo in range(0,len(stored),16384):
                hi=min(lo+16384,len(stored));expected=np.full(hi-lo,float(intercept),dtype=np.float64)
                for array,coefficient in zip(arrays,coefficients):expected=expected+array[lo:hi].astype(np.float64)*float(coefficient)
                if wide:expected=np.clip(expected,1.05,4.95)
                if not np.array_equal(expected.astype('<f4').view('u4'),stored[lo:hi].view('u4')):raise ValueError('Every independent stored arithmetic bit must match')
            if name not in study.CANDIDATES:continue
            values=np.clip(stored.astype(np.float64),1.05,4.95);groups={}
            audited=next(r for r in a['results'] if r['candidate']==name)
            for group,count in [(0,1408789),(1,1408342)]:
                ids=np.flatnonzero(labels[:,1]==group)
                if len(ids)!=count:raise ValueError('Exact complete canonical group count required')
                partials=[]
                for start in range(0,count,16384):
                    selected=ids[start:start+16384];errors=values[selected]-labels[selected,0]
                    partials.append(math.fsum(float(e)*float(e) for e in errors))
                score=math.sqrt(math.fsum(partials)/count);key='source'+str(group)
                if abs(score-audited['groups'][key]['rmse'])>1e-12:raise ValueError('Separate compensated canonical score differs')
                groups[key]=dict(rows=count,rmse=score)
            results.append(dict(candidate=name,groups=groups,prediction_sha256=f['predictions'][name],
                both_groups_below_target=all(g['rmse']<.85725 for g in groups.values())))
        primary=next(r for r in results if r['candidate']==a['primary'])
        previous=prior['groups']
        record=dict(state=('complete_independent_complete13_calibration_and_frozen_group_scores' if wide else
            'complete_independent_own352_foldbag_calibration_and_frozen_group_scores'),observed=thirteen.own6.stamp(),
            primary=a['primary'],groups=primary['groups'],results=results,previous_groups=previous,
            both_groups_improved=all(primary['groups'][k]['rmse']<previous[k]['rmse'] for k in previous),
            both_groups_below_target=primary['both_groups_below_target'],weights=weights.tolist(),
            independent_simplex_weights=independent.tolist(),simplex_supports_examined=8191 if wide else 127,
            independently_solved_full_simplex=True,exact_known_and_full_stored_vectors_bitwise=True,
            all_original_development_and_full_family_replays_verified=True,all_original_source_and_fitted_model_files_rehashed=True,
            actual_own352_qualifying_models_reopened=15,known_score_is_nested_cross_validation=False,
            qualifying_labels_used_for_fitting_or_selection=False,label_git_blob=blob,prediction_sha256=primary['prediction_sha256'],
            source_sha256=c['source_sha256'],
            limitation='Adaptive community-recovered audit, not official or untouched Test; original A/B input realization caveat retained.')
        pending.append((study,record))
    thirteen.own6.verify_bindings(bindings)
    metadata={key:value for key,value in bindings.items() if Path(key).suffix in {'.py','.json','.pt','.npz','.txt','.rs','.toml'}}
    metadata[str(label_path.relative_to(u.ROOT))]=u.sha(label_path)
    for study,record in pending:
        record['bindings']=metadata
        u.write(study.REPORT/'independent-canonical-verification.json',record)
        print(record['primary'],record['groups'],flush=True)


if __name__=='__main__':main()
