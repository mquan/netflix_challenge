"""Admit only the original known-only own15 primary and complete frozen audit."""
import datetime as dt
import math
from pathlib import Path
from complete_trajectory_publication import digest,read,require,validate as validate_neural

QUEUE_SHA='c74b884f1ed95b147c0cd7e6bdcf598a7c27c592ea845b0cd3dd1a812bb4b672'
BASE_SHA='4bee06e3d9fb81aa8b5148a401e706156030acab8c3d804d540084ea4da312bc'
NEURAL_SHA='aa9385d900f2c8a4844c1fc4d2413fddb418340c8e07c37472624ba4912f23b8'
REPORT='experiments/upstream-own15-complete-trajectory-calibration'
RUN='artifacts/upstream-own15-complete-trajectory-calibration'
BASE='experiments/upstream-own14-matched391-calibration'
NEURAL='experiments/upstream-complete-trajectory-h128-mps'
BAG='experiments/upstream-complete-trajectory-fold-bag'
CANDIDATES=['own15-simplex','own15-ridge10','complete-trajectory-fixed-half']
METADATA=['plan.json','calibration.json','known-completion.json','frozen.json',
          'independent-preaudit-verification.json','independent-recovered-audit.json']
KNOWN=['known-simplex','known-ridge10','known-fixed-half']


def validate_metadata(latest,parts,base_cal,hashes):
    """Semantic checks independent of file IO; these do not establish replay."""
    p,c,k,f,pre,a=[parts[name] for name in METADATA]
    families=base_cal['family_ids']+['nn-complete-trajectory-fold-bag']
    require(latest['state']=='complete_independent_own15_full_source_models_coefficients_and_frozen_group_scores' and
        all(v['primary']==CANDIDATES[0] for v in (latest,p,f,a)) and p['families']==15 and
        p['candidates']==CANDIDATES and c['family_ids']==families and len(families)==15 and
        [r['id'] for r in c['families']]==families and c['families'][:14]==base_cal['families'] and
        c['families'][-1]==dict(id=families[-1],family='nn',development='upstream-complete-trajectory-h128-mps',
            known_prediction='artifacts/upstream-complete-trajectory-h128-mps/nn.f32',
            qualifying_prediction='artifacts/upstream-complete-trajectory-fold-bag/nn-bag.f32'),
        'Original15 ordered families and predeclared simplex primary required')
    weights=c['simplex_weights'];ridge=c['ridge_weights'];independent=pre['independent_simplex_weights'];opt=pre['actual_simplex_optimality']
    require(len(weights)==len(independent)==len(ridge)==15 and
        all(math.isfinite(w) and w>=0 for w in weights) and abs(sum(weights)-1)<1e-12 and
        all(math.isfinite(w) for w in ridge+[c['ridge_intercept']]) and
        all(math.isfinite(w) and w>=-1e-9 for w in independent) and abs(sum(independent)-1)<1e-9 and
        all(v['ridge_lambda']==10. for v in (p,c,pre)) and pre['ridge_equations']==latest['ridge_equations']==16 and
        pre['simplex_supports_examined']==latest['simplex_supports_examined']==32767 and
        math.isfinite(opt['absolute_gap']) and 0<=opt['absolute_gap']<=1e-12 and
        all(math.isfinite(opt[key]) for key in ('objective','independent_optimum')) and
        opt['absolute_gap']==abs(opt['objective']-opt['independent_optimum']) and
        math.isfinite(opt['minimum_reduced_gradient']) and opt['minimum_reduced_gradient']>=-1e-9 and
        math.isfinite(opt['complementarity_error']) and 0<=opt['complementarity_error']<=1e-10 and
        opt['nonunique_coefficients_allowed'] is True,
        'Complete32767-support actual KKT optimum and all16 ridge equations required')
    require(k['state']=='complete_original15_known_only_coefficients_before_candidate_freeze' and
        f['state']=='complete_all_three_own15_candidates_frozen_before_any_new_audit' and
        pre['state']=='complete_independent_all32767_supports_ridge16_and_six_vectors_before_audit' and
        a['state']=='complete_all_three_original_own15_frozen_candidate_audits' and
        pre['all_six_candidate_vectors_bitwise'] is True and latest['all_six_vectors_independently_verified'] is True and
        latest['full_inherited_source_closure_rehashed'] is True and latest['actual_new_qualifying_models_reopened']==15 and
        latest['source_sha256']==pre['source_sha256']==f['source_sha256'] and
        p['source_sha256']==c['source_sha256'] and
        all(f['source_sha256'].get(name)==value for name,value in c['source_sha256'].items()) and
        all(v['qualifying_labels_read'] is False for v in (p,c,k,f,pre)) and
        all(v['known_score_is_nested_cross_validation'] is False for v in (p,c,pre,latest)) and
        all(v['qualifying_labels_used_for_fitting_or_selection'] is False for v in (latest,a)) and
        all(v['official_or_untouched_test'] is False for v in (latest,a)),
        'Complete label-free known fit/freeze/preaudit and separate adaptive audit required')
    require(f['plan_sha256']==k['plan_sha256']==hashes['plan.json'] and
        f['calibration_sha256']==k['calibration_sha256']==pre['calibration_sha256']==hashes['calibration.json'] and
        f['known_completion_sha256']==hashes['known-completion.json'] and
        pre['freeze_sha256']==hashes['frozen.json'] and pre['verifier_sha256']==p['verifier_sha256'] and
        a['preaudit_verification_sha256']==hashes['independent-preaudit-verification.json'] and
        set(f['predictions'])==set(KNOWN+CANDIDATES) and set(k['known_predictions'])==set(KNOWN) and
        all(k['known_predictions'][name]==f['predictions'][name] for name in KNOWN),
        'Exact six-vector freeze and complete preceding proof hashes required')
    dates=[dt.datetime.fromisoformat(value) for value in
        (p['created'],k['observed'],f['observed'],pre['observed'],a['observed'],latest['observed'])]
    require(all(value.utcoffset() is not None for value in dates) and all(x<y for x,y in zip(dates,dates[1:])),
        'Known fit and complete preaudit must precede all recovered outcomes')
    require([r['candidate'] for r in latest['results']]==[r['candidate'] for r in a['results']]==CANDIDATES and
        latest['groups']==latest['results'][0]['groups'] and
        latest['prediction_sha256']==f['predictions'][CANDIDATES[0]] and
        latest['both_groups_below_target']==latest['results'][0]['both_groups_below_target'] and
        latest['label_git_blob']==a['label_git_blob']=='47ed4788ffba11aad029854a8e60d4dee04fc899',
        'All three fixed candidates and the original primary group vector required')
    for result,audited in zip(latest['results'],a['results']):
        require(result['prediction_sha256']==audited['prediction_sha256']==f['predictions'][result['candidate']] and
            set(result['groups'])==set(audited['groups'])=={'source0','source1'},'Original frozen audit vector required')
        for group,count in [('source0',1408789),('source1',1408342)]:
            x,y=result['groups'][group],audited['groups'][group]
            require(x['rows']==y['rows']==count and math.isfinite(x['rmse']) and x['rmse']>0 and
                math.isfinite(y['rmse']) and abs(x['rmse']-y['rmse'])<=1e-12,
                'Both complete original groups and separate compensated scores required')
        require(result['both_groups_below_target']==all(g['rmse']<.85725 for g in result['groups'].values()) and
            audited['both_groups_below_target']==result['both_groups_below_target'],
            'Strict target must apply to both groups of the same predeclared candidate')


def verify_sources(root,bindings):
    groups={'code':0,'prior_neural':0,'new_neural':0,'new_bag':0}
    for name,expected in bindings.items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'External publication source refused')
        group=None
        if name.endswith('.py'):group='code'
        elif name.startswith(('artifacts/upstream-own391-paired-h128-mps/','artifacts/upstream-own391-fold-bag/',
                              'experiments/upstream-own391-paired-h128-mps/','experiments/upstream-own391-fold-bag/')):group='prior_neural'
        elif name.startswith(('artifacts/upstream-complete-trajectory-h128-mps/',NEURAL+'/')):group='new_neural'
        elif name.startswith(('artifacts/upstream-complete-trajectory-fold-bag/',BAG+'/')):group='new_bag'
        if group:
            require(digest(root/path)==expected,'Published source/model/scaler/query proof changed: '+name);groups[group]+=1
    require(all(groups.values()),'All code and old/new neural/bag source categories required')
    return groups


def validate(root,latest_path,latest,*,require_improvement=True):
    root=Path(root);report=root/REPORT;run=root/RUN
    require(latest_path==report/'independent-canonical-verification.json','Exact original own15 canonical path required')
    queue=root/'experiments/upstream-own15-complete-trajectory-queue/plan.json'
    require(digest(queue)==QUEUE_SHA,'Original own15 declaration before outcomes required')
    for name,value in read(queue)['source_sha256'].items():require(digest(root/name)==value,'Original own15 source changed: '+name)
    base_path=root/BASE/'independent-canonical-verification.json'
    require(digest(base_path)==BASE_SHA,'Original verified own14 benchmark required')
    base=read(base_path);base_cal_path=root/BASE/'calibration.json';base_cal=read(base_cal_path)
    require(digest(base_cal_path)==base['bindings'][str(base_cal_path.relative_to(root))],'Original14 family metadata changed')
    require(set(latest['bindings'])=={REPORT+'/'+name for name in METADATA},'All six original own15 metadata bindings required')
    hashes={name:digest(report/name) for name in METADATA}
    for name,value in latest['bindings'].items():require(digest(root/name)==value,'Frozen own15 metadata changed: '+name)
    parts={name:read(report/name) for name in METADATA};validate_metadata(latest,parts,base_cal,hashes)
    p,c,k,f,pre,a=[parts[name] for name in METADATA]
    for key,name in [('script_sha256','scripts/upstream_own15_complete_trajectory_calibration.py'),
        ('verifier_sha256','scripts/verify_upstream_own15_complete_trajectory_calibration.py'),
        ('declaration_sha256','OWN15_COMPLETE_TRAJECTORY_CALIBRATION_PLAN.md')]:
        require(p[key]==digest(root/name),'Original known-only calibration implementation changed')
    if require_improvement:
        require(all(latest['groups'][g]['rmse']<base['groups'][g]['rmse'] for g in ('source0','source1')),
            'Only a primary improving both previous groups may replace the qualifying benchmark')
    for name,value in f['predictions'].items():
        path=run/(name+'.f32');rows=1128164 if name in KNOWN else 2817131
        require(path.stat().st_size==4*rows and digest(path)==value,'Complete original frozen vector changed')
    verify_sources(root,f['source_sha256'])
    cp=root/NEURAL/'independent-canonical-verification.json';vp=root/NEURAL/'independent-verification.json'
    require(digest(cp)==NEURAL_SHA==f['source_sha256'].get(str(cp.relative_to(root))),'Original complete trajectory NN source required')
    canonical=read(cp);model=read(vp)
    record=dict(independent_verification_sha256=NEURAL_SHA,verified_development_report_sha256=NEURAL_SHA,
        best_development_rmse=canonical['fixed_half_tree154_rmse'])
    validate_neural(root,record,cp,canonical,vp,model,require_improvement=False)
    bp=root/BAG/'independent-verification.json';bag=read(bp)
    require(digest(bp)==f['source_sha256'].get(str(bp.relative_to(root))) and
        bag['state']=='complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag' and
        bag['actual_models_reopened']==bag['seed_slots']==15 and bag['rows']==2817131 and
        all(bag[key] is True for key in ('all_original_model_and_input_streams_replayed','uniform_five_fold_bag_bitwise_equal',
            'matched_B_raw_operators_on_both_queries','known_OOF_and_qualifying_bag_constructions_distinct')) and
        bag['qualifying_labels_read'] is False and bag['models_fitted']==0 and bag['qualifying_score_claimed'] is False,
        'Complete15-model/all-input qualifying replay required before calibration publication')
    bag_vector=root/'artifacts/upstream-complete-trajectory-fold-bag/nn-bag.f32'
    require(digest(bag_vector)==bag['prediction_sha256']==f['source_sha256'].get(str(bag_vector.relative_to(root))),
        'Exact full qualifying neural bag required')
    for row in c['families']:
        for key,rows in [('known_prediction',1128164),('qualifying_prediction',2817131)]:
            path=root/row[key];require(path.stat().st_size==4*rows and digest(path)==f['source_sha256'][row[key]],
                'Every original complete known/qualifying family vector must remain unchanged')
    return True,True
