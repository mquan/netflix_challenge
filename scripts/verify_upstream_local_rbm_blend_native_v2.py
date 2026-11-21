"""Independent full16 calibration replay, frozen audit and canonical scores."""
import math
from pathlib import Path
import numpy as np
import upstream_local_rbm_blend_native_v2 as study
import verify_upstream_own16_weekend_trajectory_calibration as arithmetic

rbm=study.rbm
inputs=study.inputs
STATE='complete_independent_local_rbm16_all65535_supports_ridge17_and_six_vectors_before_audit'
GROUP_COUNTS={0:1408789,1:1408342}


def known_rmse(values,labels):
    clipped=np.clip(values.astype(np.float64),1.05,4.95)
    return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(clipped,labels))/len(labels))


def evidence(guard,rehash=True):
    p,c,f=[rbm.read(study.REPORT/name) for name in ('plan.json','calibration.json','frozen.json')]
    if (p['state']!='declared_own16_local_rbm_known_only_simplex_ridge10_and_original_control'
            or p['primary']!=study.CANDIDATES[0] or p['candidates']!=list(study.CANDIDATES) or p['families']!=16
            or p['ridge_lambda']!=10. or p['qualifying_labels_read'] is not False
            or p['known_score_is_nested_cross_validation'] is not False
            or c['family_ids']!=study.BASE_IDS+[study.FAMILY_ID] or [r['id'] for r in c['families']]!=c['family_ids']
            or c['source_sha256']!=p['source_sha256'] or c['qualifying_labels_read'] is not False
            or c['known_score_is_nested_cross_validation'] is not False
            or f['state']!='complete_all_three_local_rbm16_candidates_frozen_before_recovered_audit'
            or f['primary']!=p['primary'] or f['plan_sha256']!=inputs.sha(study.REPORT/'plan.json')
            or f['calibration_sha256']!=inputs.sha(study.REPORT/'calibration.json')
            or f['known_completion_sha256']!=inputs.sha(study.REPORT/'known-completion.json')
            or f['source_sha256']!=c['source_sha256'] or f['qualifying_labels_read'] is not False
            or set(f['outputs'])!=set(study.CANDIDATES)|{'known-simplex','known-ridge10','known-control'}):
        raise ValueError('Exact all16 known-only fit and all six frozen candidates required')
    original_rows,original_bindings=study.source_declaration()
    if c['families']!=original_rows or c['source_sha256']!=original_bindings:
        raise ValueError('Every original source-bound family path/order and full inherited closure required')
    own={}
    for name,row in f['outputs'].items():
        path=study.ART/(name+'.f32')
        if Path(row['path'])!=path or path.stat().st_size!=row['bytes']:raise ValueError('Exact owned full frozen vector required')
        own[str(path.relative_to(study.ROOT))]=row['sha256']
    inputs.verify_files(study.ROOT,own,guard)
    records=study.verify_bindings(c['source_sha256'],guard) if rehash else c['source_file_records']
    return p,c,f,records


def require_preaudit(guard,rehash=True):
    p,c,f,records=evidence(guard,rehash);pre=rbm.read(study.REPORT/'independent-preaudit-verification.json')
    if (pre['state']!=STATE or pre['freeze_sha256']!=inputs.sha(study.REPORT/'frozen.json')
            or pre['calibration_sha256']!=inputs.sha(study.REPORT/'calibration.json')
            or pre['verifier_sha256']!=inputs.sha(Path(__file__)) or pre['simplex_supports_examined']!=65535
            or pre['ridge_equations']!=17 or pre['ridge_lambda']!=10. or pre['all_six_vectors_bitwise'] is not True
            or pre['source_sha256']!=f['source_sha256'] or pre['qualifying_labels_read'] is not False
            or pre['known_score_is_nested_cross_validation'] is not False):
        raise ValueError('Complete independent preaudit required before reading recovered labels')
    return p,c,f,records,pre


def verify(guard):
    p,c,f,records=evidence(guard);guard()
    label_path=study.preceding.KNOWN/'labels.npy'
    with study.columns(c['families'],'known',records,guard) as columns,study.admission.mapped(label_path,records[str(label_path.relative_to(study.ROOT))]) as labels:
        y=labels.astype(np.float64)
        if y.shape!=(study.COUNTS['known'],) or not np.isin(y,[1,2,3,4,5]).all():raise ValueError('Full original known labels required')
        reference=columns[-1].astype(np.float64)
        gram,rhs=arithmetic.products([v.astype(np.float64)-reference for v in columns],y-reference);gram/=len(y);rhs/=len(y);guard()
        independent,count=arithmetic.support_solution(gram,rhs);guard()
        actual=np.asarray(c['simplex_weights']);optimality=arithmetic.verify_simplex_weights(gram,rhs,actual,independent)
        if (np.max(abs(gram-np.asarray(c['simplex_gram'])))>1e-12 or np.max(abs(rhs-np.asarray(c['simplex_rhs'])))>1e-12):
            raise ValueError('All independently compensated simplex statistics required')
        raw,target=arithmetic.products([v.astype(np.float64) for v in columns]+[np.ones(len(y))],y);guard()
        penalty=np.eye(17)*10.;penalty[-1,-1]=0;ridge=np.r_[c['ridge_weights'],c['ridge_intercept']]
        independently_solved=np.linalg.solve(raw+penalty,target)
        if (c['ridge_lambda']!=10. or ridge.shape!=(17,) or not np.isfinite(ridge).all()
                or np.max(abs(raw-np.asarray(c['ridge_gram'])))/len(y)>1e-12
                or np.max(abs(target-np.asarray(c['ridge_rhs'])))/len(y)>1e-12
                or np.max(abs(ridge-independently_solved))>1e-7
                or np.max(abs((raw+penalty)@ridge-target))/len(y)>1e-12):
            raise ValueError('All original seventeen independent ridge equations required')
        fitted=[arithmetic.stored(study.ART/'known-simplex.f32',columns,actual),
                arithmetic.stored(study.ART/'known-ridge10.f32',columns,ridge[:-1],ridge[-1])]
        guard()
        known_control=study.preceding.RUN/'known-simplex.f32'
        if inputs.sha(study.ART/'known-control.f32')!=records[str(known_control.relative_to(study.ROOT))]['sha256']:
            raise ValueError('Original preceding own15 control must remain bitwise')
        control=np.fromfile(study.ART/'known-control.f32',dtype='<f4')
        if control.shape!=y.shape:raise ValueError('Complete known control required')
        scores=[known_rmse(v,y) for v in fitted+[control]]
    with study.columns(c['families'],'qualifying',records,guard) as columns:
        arithmetic.stored(study.ART/(study.CANDIDATES[0]+'.f32'),columns,actual);guard()
        arithmetic.stored(study.ART/(study.CANDIDATES[1]+'.f32'),columns,ridge[:-1],ridge[-1]);guard()
    control_path=study.preceding.RUN/(study.preceding.CANDIDATES[0]+'.f32')
    if inputs.sha(study.ART/(study.CANDIDATES[2]+'.f32'))!=records[str(control_path.relative_to(study.ROOT))]['sha256']:
        raise ValueError('Original preceding full qualifying control must remain bitwise')
    rbm.exclusive_json(study.REPORT/'independent-preaudit-verification.json',dict(state=STATE,observed=rbm.stamp(),
        freeze_sha256=inputs.sha(study.REPORT/'frozen.json'),calibration_sha256=inputs.sha(study.REPORT/'calibration.json'),
        verifier_sha256=inputs.sha(Path(__file__)),simplex_supports_examined=count,independent_simplex_weights=independent.tolist(),
        actual_simplex_optimality=optimality,ridge_equations=17,ridge_lambda=10.,independent_ridge_coefficients=independently_solved.tolist(),
        all_six_vectors_bitwise=True,known_apparent_fit_rmse=scores,known_score_is_nested_cross_validation=False,
        source_sha256=f['source_sha256'],qualifying_labels_read=False),guard)


def labels(guard):
    # This function is called only after require_preaudit has succeeded. The
    # immutable blob/domain checker is shared with the preceding canonical audit.
    import verify_upstream_own15_complete_trajectory_calibration as previous
    guard();value,blob=previous.labels();guard()
    if value.shape!=(study.COUNTS['qualifying'],2):raise ValueError('Full original recovered population required')
    return value,blob


def score_groups(values,y,guard,chunked=False):
    if values.shape!=(study.COUNTS['qualifying'],) or not np.isfinite(values).all():raise ValueError('Full finite frozen candidate required')
    if y.shape!=(len(values),2) or not np.isin(y[:,0],[1,2,3,4,5]).all() or not np.isin(y[:,1],[0,1]).all():
        raise ValueError('Canonical outcome domains required')
    groups={}
    for group,count in GROUP_COUNTS.items():
        guard();ids=np.flatnonzero(y[:,1]==group)
        if len(ids)!=count:raise ValueError('Exact recovered group population required')
        if chunked:
            partial=[]
            for lo in range(0,count,16384):
                guard();selected=ids[lo:lo+16384];errors=values[selected]-y[selected,0]
                partial.append(math.fsum(float(e)*float(e) for e in errors))
            total=math.fsum(partial)
        else:total=math.fsum((float(values[i])-float(y[i,0]))**2 for i in ids)
        groups['source'+str(group)]={'rows':count,'rmse':math.sqrt(total/count)}
    return groups


def audit(guard):
    p,c,f,records,pre=require_preaudit(guard);y,blob=labels(guard);results=[]
    for name in study.CANDIDATES:
        values=np.clip(np.fromfile(study.ART/(name+'.f32'),dtype='<f4').astype(np.float64),1.05,4.95)
        groups=score_groups(values,y,guard)
        results.append(dict(candidate=name,groups=groups,prediction_sha256=f['outputs'][name]['sha256'],
            both_groups_below_target=all(v['rmse']<.85725 for v in groups.values())))
    rbm.exclusive_json(study.REPORT/'independent-recovered-audit.json',dict(state='complete_all_three_fixed_local_rbm16_recovered_audits',
        observed=rbm.stamp(),primary=p['primary'],results=results,label_git_blob=blob,
        preaudit_sha256=inputs.sha(study.REPORT/'independent-preaudit-verification.json'),
        qualifying_labels_read=True,qualifying_labels_used_for_fitting_or_selection=False,official_or_untouched_test=False),guard)


def canonical(guard):
    p,c,f,records,pre=require_preaudit(guard);audit_path=study.REPORT/'independent-recovered-audit.json';aud=rbm.read(audit_path)
    if (aud['state']!='complete_all_three_fixed_local_rbm16_recovered_audits' or aud['primary']!=p['primary']
            or aud['preaudit_sha256']!=inputs.sha(study.REPORT/'independent-preaudit-verification.json')
            or aud['qualifying_labels_used_for_fitting_or_selection'] is not False
            or [r['candidate'] for r in aud['results']]!=list(study.CANDIDATES)):
        raise ValueError('Original fixed candidate audit required')
    y,blob=labels(guard);results=[]
    if aud['label_git_blob']!=blob:raise ValueError('Original canonical label blob required')
    with study.columns(c['families'],'qualifying',records,guard) as columns:
        for index,name in enumerate(study.CANDIDATES):
            if index==0:values=arithmetic.stored(study.ART/(name+'.f32'),columns,c['simplex_weights'])
            elif index==1:values=arithmetic.stored(study.ART/(name+'.f32'),columns,c['ridge_weights'],c['ridge_intercept'])
            else:
                path=study.preceding.RUN/(study.preceding.CANDIDATES[0]+'.f32')
                if inputs.sha(study.ART/(name+'.f32'))!=records[str(path.relative_to(study.ROOT))]['sha256']:
                    raise ValueError('Original preceding candidate required')
                values=np.fromfile(study.ART/(name+'.f32'),dtype='<f4')
            groups=score_groups(np.clip(values.astype(np.float64),1.05,4.95),y,guard,chunked=True)
            audited=aud['results'][index]
            if audited['prediction_sha256']!=f['outputs'][name]['sha256'] or any(abs(groups[k]['rmse']-audited['groups'][k]['rmse'])>1e-12 for k in groups):
                raise ValueError('Separate complete canonical scores must match')
            results.append(dict(candidate=name,groups=groups,prediction_sha256=f['outputs'][name]['sha256'],
                both_groups_below_target=all(v['rmse']<.85725 for v in groups.values())))
    primary=results[0]
    rbm.exclusive_json(study.REPORT/'independent-canonical-verification.json',dict(
        state='complete_independent_local_rbm16_full_source_models_coefficients_and_frozen_group_scores',
        observed=rbm.stamp(),primary=p['primary'],groups=primary['groups'],results=results,prediction_sha256=primary['prediction_sha256'],
        both_groups_below_target=primary['both_groups_below_target'],simplex_supports_examined=65535,ridge_equations=17,
        all_six_vectors_independently_verified=True,full_inherited_source_closure_rehashed=True,source_sha256=f['source_sha256'],
        label_git_blob=blob,qualifying_labels_used_for_fitting_or_selection=False,known_score_is_nested_cross_validation=False,
        official_or_untouched_test=False,bindings={str((study.REPORT/name).relative_to(study.ROOT)):inputs.sha(study.REPORT/name) for name in
            ('plan.json','calibration.json','known-completion.json','frozen.json','independent-preaudit-verification.json','independent-recovered-audit.json')}),guard)


def main(phase):
    import upstream_local_rbm_blend_queue_native_v2 as queue
    queue.authorize(phase);guard=queue.Guard(phase);guard()
    rbm.exclusive_json(study.REPORT/(phase+'-started.json'),dict(phase=phase,pid=study.os.getpid(),observed=rbm.stamp()),guard)
    {'preaudit':verify,'audit':audit,'canonical':canonical}[phase](guard)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('preaudit','audit','canonical'));main(parser.parse_args().phase)
