"""Publish only the completed original local-RBM16 primary and actual source chain."""
import datetime as dt
import math
from pathlib import Path
from complete_trajectory_publication import digest,read,require

REPORT='experiments/upstream-local-rbm-blend-deterministic-metal'
QUEUE='experiments/upstream-local-rbm-blend-queue-deterministic-metal'
CANONICAL_SHA='d646e5b449052126fdfae87978578d741a4e58a85ce0a8ec524406a2caa9d1e0'
QUEUE_SHA='958db8f3131e3db4ad57cbd317d7fa8ae58c87c5879d747b3fc554ea5591bd57'
PAIR_SHA='3e8d2ecdbb40d0c953f98b24601c2e84a9353bb45ca23d74fea26797a814199a'
CANDIDATES=['local-rbm16-simplex','local-rbm16-ridge10','preceding-own15-simplex']
METADATA=['plan.json','calibration.json','known-completion.json','frozen.json',
          'independent-preaudit-verification.json','independent-recovered-audit.json']


def validate_metadata(latest,parts,hashes):
    p,c,k,f,pre,a=[parts[name] for name in METADATA]
    require(latest['state']=='complete_independent_local_rbm16_full_source_models_coefficients_and_frozen_group_scores'
        and all(v['primary']==CANDIDATES[0] for v in (latest,p,f,a)) and p['candidates']==CANDIDATES
        and p['families']==len(c['families'])==len(c['family_ids'])==16
        and [r['id'] for r in c['families']]==c['family_ids'], 'Original sixteen families and fixed primary required')
    w=c['simplex_weights'];opt=pre['actual_simplex_optimality']
    require(len(w)==len(c['ridge_weights'])==len(pre['independent_simplex_weights'])==16
        and all(math.isfinite(x) and x>=0 for x in w) and abs(sum(w)-1)<1e-12
        and all(math.isfinite(x) and x>=-1e-9 for x in pre['independent_simplex_weights'])
        and abs(sum(pre['independent_simplex_weights'])-1)<1e-9
        and all(math.isfinite(x) for x in c['ridge_weights']+[c['ridge_intercept']])
        and all(v['ridge_lambda']==10. for v in (p,c,pre))
        and pre['simplex_supports_examined']==latest['simplex_supports_examined']==65535
        and pre['ridge_equations']==latest['ridge_equations']==17
        and math.isfinite(opt['absolute_gap']) and 0<=opt['absolute_gap']<=1e-12
        and opt['absolute_gap']==abs(opt['objective']-opt['independent_optimum'])
        and math.isfinite(opt['minimum_reduced_gradient']) and opt['minimum_reduced_gradient']>=-1e-9
        and math.isfinite(opt['complementarity_error']) and 0<=opt['complementarity_error']<=1e-10,
        'All65535 supports, actual coefficient KKT optimum and17 ridge equations required')
    require(all(v['qualifying_labels_read'] is False for v in (p,c,k,f,pre))
        and all(v['known_score_is_nested_cross_validation'] is False for v in (p,c,pre,latest))
        and all(v['qualifying_labels_used_for_fitting_or_selection'] is False for v in (a,latest))
        and all(v['official_or_untouched_test'] is False for v in (a,latest))
        and pre['all_six_vectors_bitwise'] is True and latest['all_six_vectors_independently_verified'] is True
        and latest['full_inherited_source_closure_rehashed'] is True
        and p['source_sha256']==c['source_sha256']==f['source_sha256']==pre['source_sha256']==latest['source_sha256'],
        'Complete known-only fit/freeze/preaudit and adaptive full source verification required')
    require(k['plan_sha256']==f['plan_sha256']==hashes['plan.json']
        and k['calibration_sha256']==f['calibration_sha256']==pre['calibration_sha256']==hashes['calibration.json']
        and f['known_completion_sha256']==hashes['known-completion.json'] and pre['freeze_sha256']==hashes['frozen.json']
        and a['preaudit_sha256']==hashes['independent-preaudit-verification.json']
        and set(f['outputs'])==set(CANDIDATES+['known-simplex','known-ridge10','known-control'])
        and set(k['outputs'])=={'known-simplex','known-ridge10','known-control'}
        and all(k['outputs'][name]==f['outputs'][name] for name in k['outputs']),
        'All original six-vector metadata expectations required')
    require([r['candidate'] for r in latest['results']]==[r['candidate'] for r in a['results']]==CANDIDATES
        and latest['groups']==latest['results'][0]['groups']
        and latest['prediction_sha256']==f['outputs'][CANDIDATES[0]]['sha256']
        and latest['label_git_blob']==a['label_git_blob']=='47ed4788ffba11aad029854a8e60d4dee04fc899',
        'Original primary vector and all three fixed audits required')
    dates=[dt.datetime.fromisoformat(v['observed']) for v in (pre,a,latest)]
    require(all(d.utcoffset() is not None for d in dates) and dates[0]<dates[1]<dates[2],
        'Complete preaudit must precede recovered audits and canonical checks')
    for row,audited in zip(latest['results'],a['results']):
        require(set(row['groups'])==set(audited['groups'])=={'source0','source1'}
            and row['prediction_sha256']==audited['prediction_sha256']==f['outputs'][row['candidate']]['sha256'],
            'Each fixed candidate must retain its original vector')
        for group,count in [('source0',1408789),('source1',1408342)]:
            x,y=row['groups'][group],audited['groups'][group]
            require(x['rows']==y['rows']==count and math.isfinite(x['rmse']) and math.isfinite(y['rmse'])
                and x['rmse']>0 and abs(x['rmse']-y['rmse'])<=1e-12,'Both complete canonical groups required')
        require(row['both_groups_below_target']==audited['both_groups_below_target']==all(g['rmse']<.85725 for g in row['groups'].values()),
            'The strict target must hold for both groups of the same candidate')
    require(latest['both_groups_below_target']==latest['results'][0]['both_groups_below_target'],
        'Primary target status must match its complete group scores')


def validate(root,latest_path,latest):
    import upstream_local_rbm_blend_metal as study
    import upstream_local_rbm_blend_queue_metal as queue
    import verify_upstream_local_rbm_blend_metal as verification
    import own15_publication as previous
    root=Path(root);report=root/REPORT
    require(latest_path==report/'independent-canonical-verification.json' and digest(latest_path)==CANONICAL_SHA
        and latest==read(latest_path),'Exact independently completed local-RBM16 canonical proof required')
    require(root==study.ROOT and digest(root/QUEUE/'plan.json')==QUEUE_SHA,'Original local-RBM16 declaration required')
    queue.check_sources(read(root/QUEUE/'plan.json'))
    done=read(root/QUEUE/'completion.json')
    require(done['state']=='complete_local_rbm16_known_only_calibration_freeze_full_replay_and_independent_recovered_audit'
        and done['verification_sha256']==CANONICAL_SHA
        and done['qualifying_labels_used_for_fitting_or_selection'] is False,'Completed original owned queue required')
    # This rehashes the entire inherited source/model/vector closure and checks
    # the original full RBM source lifecycle; it does not read recovered labels,
    # refit coefficients, infer new predictions or invent a model replay.
    p,c,f,records,pre=verification.require_preaudit(lambda **kwargs:None,rehash=True)
    require(set(latest['bindings'])=={REPORT+'/'+name for name in METADATA},'Exact six metadata bindings required')
    hashes={name:digest(report/name) for name in METADATA}
    for name,expected in latest['bindings'].items():require(digest(root/name)==expected,'Original frozen metadata changed: '+name)
    parts={name:read(report/name) for name in METADATA};validate_metadata(latest,parts,hashes)
    require(pre['verifier_sha256']==digest(root/'scripts/verify_upstream_local_rbm_blend_metal.py'),
        'Original independent numerical verifier required')
    pair=root/'experiments/upstream-factored-rbm-gpu400-deterministic-metal/paired-verification.json'
    require(digest(pair)==PAIR_SHA==f['source_sha256'].get(str(pair.relative_to(root))),
        'Complete actual fifteen-plus-fifteen RBM and both full saved-model query replays required')
    prior=root/previous.REPORT/'independent-canonical-verification.json';base=read(prior)
    previous.validate(root,prior,base)
    require(c['families'][:15]==read(root/previous.REPORT/'calibration.json')['families']
        and all(latest['groups'][g]['rmse']<base['groups'][g]['rmse'] for g in ('source0','source1')),
        'All prior fifteen families and a primary improving both groups required')
    for name,row in f['outputs'].items():
        path=study.ART/(name+'.f32');rows=1128164 if name.startswith('known-') else 2817131
        require(Path(row['path'])==path and path.stat().st_size==rows*4 and digest(path)==row['sha256'],
            'Every complete original frozen float32 vector must survive publication')
    return True,True
