#!/usr/bin/env python3
"""Declare/freeze known-only own16 weights and three immutable query candidates."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import datetime as dt
import fcntl
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import upstream_repro as u
import upstream_weekend_trajectory_ops as ops
import upstream_own15_complete_trajectory_calibration as preceding

REPORT=u.ROOT/'experiments/upstream-own16-weekend-trajectory-calibration'
RUN=u.ROOT/'artifacts/upstream-own16-weekend-trajectory-calibration'
DECL=u.ROOT/'WEEKEND_COMPLETE_TRAJECTORY_PLAN.md'
KNOWN=u.ROOT/'artifacts/upstream-weekend-trajectory-h128-mps'
NEURAL=u.ROOT/'experiments/upstream-weekend-trajectory-h128-mps'
BAG=u.ROOT/'experiments/upstream-weekend-trajectory-fold-bag'
BAG_RUN=u.ROOT/'artifacts/upstream-weekend-trajectory-fold-bag'
COUNTS=dict(known=1128164,qual=2817131)
CANDIDATES=['own16-simplex','own16-ridge10','weekend-trajectory-fixed-half']
BASE_IDS=preceding.BASE_IDS+['nn-complete-trajectory-fold-bag']


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


sha=ops.sha


def verify_bindings(bindings):
    synthetic=u.read(u.BASE/'fresh-training-plan.json')['synthetic_qualifying_payload_sha256']
    allowed={'artifacts/'+name+'/source/data/qual_ratings/qual_ratings.csv.gz'
             for name in ['upstream-original1200','upstream-high-capacity-original']}
    for name,digest in bindings.items():
        ops.relative(name)
        if 'qual_ratings.csv' in name and (name not in allowed or digest!=synthetic):
            raise ValueError('Recovered outcomes cannot enter fitting closure')
        if sha(u.ROOT/name)!=digest:raise ValueError('Immutable original source changed: '+name)


def bind(bindings,path,expected=None):
    relative=str(Path(path).relative_to(u.ROOT));value=sha(path) if expected is None else expected
    if relative in bindings and bindings[relative]!=value:raise ValueError('Conflicting original expectation: '+relative)
    bindings[relative]=value


def bind_known_metadata(bindings,name):
    if name not in ['labels.npy','users.npy','groups.npy']:raise ValueError('Canonical metadata name required')
    original='artifacts/upstream-weekend-trajectory-data/'+name
    if original not in bindings:raise ValueError('Original admitted canonical metadata expectation required')
    bind(bindings,KNOWN/name,bindings[original])


def sources():
    old=preceding.REPORT
    p,c,f,pre=[u.read(old/name) for name in ['plan.json','calibration.json','frozen.json','independent-preaudit-verification.json']]
    if (p['families']!=15 or c['family_ids']!=BASE_IDS or [r['id'] for r in c['families']]!=BASE_IDS or
        c['qualifying_labels_read'] is not False or c['known_score_is_nested_cross_validation'] is not False or
        pre['state']!='complete_independent_all32767_supports_ridge16_and_six_vectors_before_audit' or
        pre['calibration_sha256']!=sha(old/'calibration.json') or pre['freeze_sha256']!=sha(old/'frozen.json') or
        pre['verifier_sha256']!=sha(u.ROOT/'scripts/verify_upstream_own15_complete_trajectory_calibration.py') or
        pre['simplex_supports_examined']!=32767 or pre['ridge_equations']!=16 or
        pre['qualifying_labels_read'] is not False or not pre['all_six_candidate_vectors_bitwise'] or
        f['plan_sha256']!=sha(old/'plan.json') or f['qualifying_labels_read'] is not False or
        f['source_sha256']!=pre['source_sha256'] or
        any(f['source_sha256'].get(k)!=v for k,v in c['source_sha256'].items())):
        raise ValueError('All15 original own families and full label-free preaudit required')
    bindings=dict(pre['source_sha256'])
    for name,digest in f['predictions'].items():bind(bindings,preceding.RUN/(name+'.f32'),digest)
    for name,digest in [('plan.json',f['plan_sha256']),('calibration.json',pre['calibration_sha256']),
        ('frozen.json',pre['freeze_sha256']),('known-completion.json',f['known_completion_sha256'])]:bind(bindings,old/name,digest)
    bind(bindings,old/'independent-preaudit-verification.json')
    mp=NEURAL/'manifest.json';vp=NEURAL/'independent-verification.json';cp=NEURAL/'independent-canonical-verification.json'
    m,v,canonical=map(u.read,[mp,vp,cp]);bp=BAG/'independent-verification.json';bag=u.read(bp)
    if (canonical['state']!='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores' or
        canonical['manifest_sha256']!=sha(mp) or canonical['complete_actual_model_replay_sha256']!=sha(vp) or
        canonical['source_sha256']!=m['source_sha256'] or canonical['qualifying_labels_read'] is not False or
        not canonical['every_nn_and_fixed_half_float32_bit_equal'] or not canonical['all_five_original_folds_verified'] or
        not canonical['neural_inputs_matched_B_on_both_queries'] or
        canonical['tree_control_retains_historical_A_B_realization_caveat'] is not True or
        v['state']!='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions' or
        v['manifest_sha256']!=sha(mp) or v['qualifying_labels_read'] is not False or
        not v['complete_original_statistics_bitwise_equal'] or not v['all_original_models_preserved_without_refitting'] or
        [r['fold'] for r in v['records']]!=list(range(5)) or
        not all(r['both_complete_eligible_scalers_independently_recomputed'] and r['all_normalized_input_bits_equal'] and
            r['actual_saved_pipeline_bitwise_equal'] for r in v['records']) or
        bag['state']!='complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag' or
        bag['actual_models_reopened']!=canonical['actual_models_rehashed'] or bag['seed_slots']!=15 or bag['rows']!=COUNTS['qual'] or
        not bag['all_original_model_and_input_streams_replayed'] or not bag['uniform_five_fold_bag_bitwise_equal'] or
        not bag['matched_B_raw_operators_on_both_queries'] or not bag['known_OOF_and_qualifying_bag_constructions_distinct'] or
        bag['qualifying_labels_read'] is not False or bag['models_fitted']!=0 or bag['qualifying_score_claimed'] is not False):
        raise ValueError('Full original new neural and qualifying-bag numerical proofs required')
    for closure in [m['source_sha256'],bag['source_sha256']]:ops.merge_bindings(bindings,closure)
    for key,digest in m['code_sha256'].items():bind(bindings,u.ROOT/key,digest)
    reopened=0
    for record in v['records']:
        path=u.ROOT/record['receipt_path'];receipt=u.read(path);counts=receipt['chosen_iterations'];folder=KNOWN/'nn'/str(record['fold'])
        if (sha(path)!=record['receipt_sha256'] or receipt['outputs']!=record['outputs'] or receipt['manifest_sha256']!=sha(mp) or
            len(counts)!=3 or any(type(count)!=int or not 0<=count<=40 for count in counts) or
            record['actual_models_reopened']!=sum(count>0 for count in counts)):
            raise ValueError('Every actual new selected model/zero-budget slot required')
        expected={'ridge.npz','scalers.npz','prediction.f32'}|{'nn-%s.pt'%seed for seed,count in zip(m['config']['seeds'],counts) if count}
        if set(receipt['outputs'])!=expected or {p.name for p in folder.iterdir()}!=expected:
            raise ValueError('Complete original pipeline inventory required')
        bind(bindings,path,record['receipt_sha256'])
        for name,digest in record['outputs'].items():bind(bindings,folder/name,digest)
        reopened+=record['actual_models_reopened']
    if reopened!=bag['actual_models_reopened']:raise ValueError('All actual new models must be accounted for')
    bind(bindings,KNOWN/'nn.f32',v['nn_sha256']);bind(bindings,KNOWN/'fixed-half-tree154.f32',canonical['prediction_sha256'])
    bind(bindings,BAG_RUN/'nn-bag.f32',bag['prediction_sha256'])
    rows=list(c['families'])+[dict(id='nn-weekend-trajectory-fold-bag',family='nn',development=KNOWN.name,
        known_prediction=str((KNOWN/'nn.f32').relative_to(u.ROOT)),qualifying_prediction=str((BAG_RUN/'nn-bag.f32').relative_to(u.ROOT)))]
    for name in ['labels.npy','users.npy','groups.npy']:
        path=KNOWN/name;bind_known_metadata(bindings,name)
        if not np.array_equal(np.load(path),np.load(u.ROOT/'artifacts/upstream-item154-tree'/name)):
            raise ValueError('All complete canonical labels/users/groups must match')
    for path in [mp,vp,cp,bp,Path(__file__),u.ROOT/'scripts/verify_upstream_own16_weekend_trajectory_calibration.py',DECL,Path(ops.__file__),
        Path(preceding.__file__),u.ROOT/'artifacts/upstream-repro/fresh/src/bin/ridge.rs']:bind(bindings,path)
    verify_bindings(bindings)
    return rows,bindings


def combine(arrays,weights,intercept=0.):
    output=np.empty(len(arrays[0]),dtype='<f4')
    for lo in range(0,len(output),32768):
        hi=min(lo+32768,len(output));total=np.full(hi-lo,float(intercept),dtype=np.float64)
        for array,weight in zip(arrays,weights):total+=array[lo:hi].astype(np.float64)*float(weight)
        output[lo:hi]=np.clip(total,1.05,4.95).astype('<f4')
    return output


def known():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing own16 study retained')
    rows,bindings=sources();REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(created=stamp(),primary=CANDIDATES[0],candidates=CANDIDATES,ridge_lambda=10.,families=16,
        source_sha256=bindings,script_sha256=sha(Path(__file__)),verifier_sha256=sha(u.ROOT/'scripts/verify_upstream_own16_weekend_trajectory_calibration.py'),
        declaration_sha256=sha(DECL),known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    arrays=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows];y=np.load(KNOWN/'labels.npy').astype(np.float64)
    if any(a.shape!=(COUNTS['known'],) or not np.isfinite(a).all() for a in arrays):raise ValueError('Complete finite known families required')
    x=np.column_stack(arrays).astype(np.float64);d=x-x[:,-1,None];residual=y-x[:,-1]
    gram=d.T@d/len(y);rhs=d.T@residual/len(y)
    opt=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(16,1/16),jac=lambda w:2*(gram@w-rhs),bounds=[(0.,1.)]*16,
        constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(16))],method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
    if not opt.success or opt.x.min()<-1e-12 or abs(opt.x.sum()-1)>1e-10:raise ValueError('Declared complete simplex did not converge')
    weights=np.maximum(opt.x,0);weights/=weights.sum();z=np.column_stack([x,np.ones(len(y))]);g=z.T@z;r=z.T@y
    penalty=np.eye(17)*10.;penalty[-1,-1]=0;ridge=np.linalg.solve(g+penalty,r)
    u.write(REPORT/'calibration.json',dict(families=rows,family_ids=[r['id'] for r in rows],simplex_weights=weights.tolist(),
        simplex_gram=gram.tolist(),simplex_rhs=rhs.tolist(),ridge_weights=ridge[:-1].tolist(),ridge_intercept=float(ridge[-1]),
        ridge_gram=g.tolist(),ridge_rhs=r.tolist(),ridge_lambda=10.,source_sha256=bindings,
        known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    for name,coefficients,intercept in [('known-simplex',weights,0.),('known-ridge10',ridge[:-1],ridge[-1])]:
        combine(arrays,coefficients,intercept).tofile(RUN/(name+'.f32'))
    half=combine([arrays[-1],arrays[5]],[.5,.5]);half.tofile(RUN/'known-fixed-half.f32')
    if sha(RUN/'known-fixed-half.f32')!=sha(KNOWN/'fixed-half-tree154.f32'):raise ValueError('Original known fixed half must match every bit')
    verify_bindings(bindings)
    u.write(REPORT/'known-completion.json',dict(state='complete_original16_known_only_coefficients_before_candidate_freeze',
        observed=stamp(),plan_sha256=sha(REPORT/'plan.json'),calibration_sha256=sha(REPORT/'calibration.json'),
        known_predictions={p.stem:sha(p) for p in RUN.glob('known-*.f32')},qualifying_labels_read=False))


def freeze():
    if (REPORT/'frozen.json').exists():raise ValueError('Original own15 freeze retained')
    done=u.read(REPORT/'known-completion.json');cal=u.read(REPORT/'calibration.json');plan=u.read(REPORT/'plan.json')
    if (done['state']!='complete_original16_known_only_coefficients_before_candidate_freeze' or
        done['calibration_sha256']!=sha(REPORT/'calibration.json') or done['plan_sha256']!=sha(REPORT/'plan.json') or done['qualifying_labels_read'] is not False or
        set(done['known_predictions'])!={'known-simplex','known-ridge10','known-fixed-half'} or
        any(sha(RUN/(name+'.f32'))!=digest for name,digest in done['known_predictions'].items())):
        raise ValueError('Original known-only coefficient freeze required')
    if {p.name for p in RUN.iterdir()}!={'known-simplex.f32','known-ridge10.f32','known-fixed-half.f32'}:
        raise ValueError('Preserve partial/extra candidate artifacts; no automatic freeze retry')
    cp=BAG/'independent-verification.json';proof=u.read(cp)
    if (proof['state']!='complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag' or
        proof['seed_slots']!=15 or proof['rows']!=COUNTS['qual'] or
        not proof['all_original_model_and_input_streams_replayed'] or not proof['uniform_five_fold_bag_bitwise_equal'] or
        not proof['matched_B_raw_operators_on_both_queries'] or not proof['known_OOF_and_qualifying_bag_constructions_distinct'] or
        proof['qualifying_labels_read'] is not False or proof['models_fitted']!=0 or proof['qualifying_score_claimed'] is not False):
        raise ValueError('Complete original15-model/all-stream qualifying replay required')
    bindings=dict(cal['source_sha256'])
    for rel,digest in proof['source_sha256'].items():bind(bindings,u.ROOT/rel,digest)
    bind(bindings,cp);bind(bindings,BAG_RUN/'nn-bag.f32',proof['prediction_sha256'])
    rows=cal['families'];arrays=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
    if len(rows)!=15 or any(a.shape!=(COUNTS['qual'],) or not np.isfinite(a).all() for a in arrays):raise ValueError('All complete finite qualifying families required')
    outputs=[combine(arrays,cal['simplex_weights']),combine(arrays,cal['ridge_weights'],cal['ridge_intercept']),combine([arrays[-1],arrays[5]],[.5,.5])]
    for name,values in zip(CANDIDATES,outputs):values.tofile(RUN/(name+'.f32'))
    verify_bindings(bindings)
    u.write(REPORT/'frozen.json',dict(state='complete_all_three_own16_candidates_frozen_before_any_new_audit',observed=stamp(),
        primary=CANDIDATES[0],plan_sha256=sha(REPORT/'plan.json'),calibration_sha256=sha(REPORT/'calibration.json'),
        known_completion_sha256=sha(REPORT/'known-completion.json'),source_sha256=bindings,
        predictions={p.stem:sha(p) for p in RUN.glob('*.f32')},qualifying_labels_read=False,
        independent_coefficient_and_arithmetic_verification_remaining=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['known','freeze']);args=parser.parse_args()
    known() if args.phase=='known' else freeze()
