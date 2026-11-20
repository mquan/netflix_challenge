#!/usr/bin/env python3
"""Declare/freeze known-only own14 weights and three immutable query candidates."""
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

REPORT=u.ROOT/'experiments/upstream-own14-matched391-calibration'
RUN=u.ROOT/'artifacts/upstream-own14-matched391-calibration'
DECL=u.ROOT/'OWN14_MATCHED391_CALIBRATION_PLAN.md'
BASE=u.ROOT/'experiments/upstream-complete13-calibration'
BASE_SHA='75b9d5e6861c1949eb30b62f06b9ea4d3b02ee0ff962e6b6545ac4b91f52d88c'
KNOWN=u.ROOT/'artifacts/upstream-own391-paired-h128-mps'
NEURAL=u.ROOT/'experiments/upstream-own391-paired-h128-mps'
BAG=u.ROOT/'experiments/upstream-own391-fold-bag'
BAG_RUN=u.ROOT/'artifacts/upstream-own391-fold-bag'
COUNTS=dict(known=1128164,qual=2817131)
CANDIDATES=['own14-simplex','own14-ridge10','matched391-fixed-half']
BASE_IDS=['tree50','nn50','nn90','nn109','nn154','tree154',
    'native:upstream-split3-virtual-mps-context','native:upstream-attention2-virtual-mps-context',
    'native:upstream-knn3pair2-compact-mps-context','native:upstream-split3eight-compact-mps-context',
    'native:upstream-cf96-prefix2030-compact-mps-context','nn330','nn352-fold-bag']


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    """Same full SHA256 bytes, descriptor-local cache hint only."""
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());fcntl.fcntl(stream.fileno(),48,1)
        for block in iter(lambda:stream.read(4*2**20),b''):h.update(block)
        after=os.fstat(stream.fileno());current=path.stat()
    def ident(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_mode,s.st_nlink,s.st_flags)
    if ident(before)!=ident(after) or ident(after)!=ident(current):raise ValueError('Source bytes/identity changed while hashing: '+str(path))
    return h.hexdigest()


def verify_bindings(bindings):
    for relative,expected in bindings.items():
        if 'qual_ratings.csv' in relative:raise ValueError('Recovered outcomes cannot enter fitting closure')
        if sha(u.ROOT/relative)!=expected:raise ValueError('Immutable original source changed: '+relative)


def bind(bindings,path,expected=None):
    relative=str(Path(path).relative_to(u.ROOT));value=sha(path) if expected is None else expected
    if relative in bindings and bindings[relative]!=value:raise ValueError('Conflicting original expectation: '+relative)
    bindings[relative]=value


def sources():
    canonical=BASE/'independent-canonical-verification.json'
    if sha(canonical)!=BASE_SHA:raise ValueError('Exact original full complete13 canonical proof required')
    baseproof=u.read(canonical);cal_path=BASE/'calibration.json';cal=u.read(cal_path)
    if (sha(cal_path)!=baseproof['bindings'][str(cal_path.relative_to(u.ROOT))] or
        cal['family_ids']!=BASE_IDS or [r['id'] for r in cal['families']]!=BASE_IDS or cal['qualifying_labels_read'] is not False):
        raise ValueError('All original13 ordered own families required')
    bindings=dict(cal['source_sha256']);bind(bindings,canonical,BASE_SHA);bind(bindings,cal_path)
    cp=NEURAL/'independent-canonical-verification.json';vp=NEURAL/'independent-verification.json';mp=NEURAL/'manifest.json'
    c,v,m=map(u.read,[cp,vp,mp])
    control_path=u.ROOT/'experiments/upstream-own391-paired-h512-mps/independent-canonical-verification.json';control=u.read(control_path)
    cvp=control_path.parent/'independent-verification.json';cmp=control_path.parent/'manifest.json'
    if (c['state']!='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores' or c['variant']!='h128' or
        c['complete_actual_model_replay_sha256']!=sha(vp) or c['manifest_sha256']!=sha(mp) or
        c['source_sha256']!=m['source_sha256'] or c['qualifying_labels_read'] is not False or not c['improves_previous'] or
        not c['neural_inputs_matched_B_on_both_queries'] or not c['all_five_original_folds_verified'] or
        c['tree_control_retains_historical_A_B_realization_caveat'] is not True or
        not c['every_nn_and_fixed_half_float32_bit_equal'] or c['actual_models_rehashed']!=15 or
        v['state']!='complete_independent_matched_B391_actual_neural_models_full_eligible_statistics_and_predictions' or
        v['qualifying_labels_read'] is not False or not v['complete_original_statistics_bitwise_equal'] or
        not v['all_original_models_preserved_without_refitting'] or not v['matched_B_known_and_qualifying'] or
        [r['fold'] for r in v['records']]!=list(range(5)) or
        not all(r['actual_models_reopened']==3 and r['both_complete_eligible_scalers_independently_recomputed'] and
            r['all_normalized_input_bits_equal'] and r['actual_saved_pipeline_bitwise_equal'] for r in v['records']) or
        control['state']!=c['state'] or control['variant']!='h512' or control['qualifying_labels_read'] is not False or
        control['complete_actual_model_replay_sha256']!=sha(cvp) or control['manifest_sha256']!=sha(cmp) or
        not control['all_five_original_folds_verified'] or not control['every_nn_and_fixed_half_float32_bit_equal'] or
        control['actual_models_rehashed']!=15 or
        c['fixed_half_tree154_rmse']>control['fixed_half_tree154_rmse']):
        raise ValueError('Complete original known-only architecture/model proofs and eligible H128 required')
    for rel,digest in m['source_sha256'].items():bind(bindings,u.ROOT/rel,digest)
    for row in v['records']:
        path=u.ROOT/row['receipt_path'];receipt=u.read(path);bind(bindings,path,row['receipt_sha256'])
        if receipt['outputs']!=row['outputs'] or receipt['manifest_sha256']!=sha(mp):raise ValueError('Original model receipt differs')
        for name,digest in row['outputs'].items():bind(bindings,KNOWN/'nn'/str(row['fold'])/name,digest)
    bind(bindings,KNOWN/'nn.f32',v['nn_sha256']);bind(bindings,KNOWN/'fixed-half-tree154.f32',c['prediction_sha256'])
    fixture=u.ROOT/'experiments/upstream-uncached-hash-fixture/verification.json';proof=u.read(fixture)
    header=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/fcntl.h')
    if (proof['state']!='complete_private_fixture_uncached_hash_bytes_metadata_and_prior_policy_roundtrip' or
        not proof['original_bytes_and_metadata_unchanged'] or proof['fixture_sha256']!=proof['per_descriptor_sha256'] or
        proof['header_sha256']!=sha(header) or proof['script_sha256']!=sha(u.ROOT/'scripts/test_uncached_hash_fixture.py')):
        raise ValueError('Actual descriptor-local SHA fixture and SDK proof required')
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_own14_matched391_calibration.py',DECL,cp,vp,mp,control_path,cvp,cmp,fixture,
        u.ROOT/'scripts/test_uncached_hash_fixture.py',u.ROOT/'artifacts/upstream-repro/fresh/src/bin/ridge.rs']:
        bind(bindings,path)
    rows=list(cal['families'])+[dict(id='nn391-matched-fold-bag',family='nn',development=KNOWN.name,
        known_prediction=str((KNOWN/'nn.f32').relative_to(u.ROOT)),qualifying_prediction=str((BAG_RUN/'nn-bag.f32').relative_to(u.ROOT)))]
    for key in ['labels.npy','users.npy','groups.npy']:
        expected=m['source_sha256'][str((u.ROOT/'artifacts/upstream-own391-paired-data'/key).relative_to(u.ROOT))]
        bind(bindings,KNOWN/key,expected)
        if not np.array_equal(np.load(KNOWN/key),np.load(u.ROOT/'artifacts/upstream-item154-tree'/key)):
            raise ValueError('Complete canonical known labels/users/groups must match')
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
    if REPORT.exists() or RUN.exists():raise ValueError('Existing own14 study retained')
    rows,bindings=sources();REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(created=stamp(),primary=CANDIDATES[0],candidates=CANDIDATES,ridge_lambda=10.,families=14,
        source_sha256=bindings,script_sha256=sha(Path(__file__)),verifier_sha256=sha(u.ROOT/'scripts/verify_upstream_own14_matched391_calibration.py'),
        declaration_sha256=sha(DECL),known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    arrays=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows];y=np.load(KNOWN/'labels.npy').astype(np.float64)
    if any(a.shape!=(COUNTS['known'],) or not np.isfinite(a).all() for a in arrays):raise ValueError('Complete finite known families required')
    x=np.column_stack(arrays).astype(np.float64);d=x-x[:,-1,None];residual=y-x[:,-1]
    gram=d.T@d/len(y);rhs=d.T@residual/len(y)
    opt=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(14,1/14),jac=lambda w:2*(gram@w-rhs),bounds=[(0.,1.)]*14,
        constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(14))],method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
    if not opt.success or opt.x.min()<-1e-12 or abs(opt.x.sum()-1)>1e-10:raise ValueError('Declared complete simplex did not converge')
    weights=np.maximum(opt.x,0);weights/=weights.sum();z=np.column_stack([x,np.ones(len(y))]);g=z.T@z;r=z.T@y
    penalty=np.eye(15)*10.;penalty[-1,-1]=0;ridge=np.linalg.solve(g+penalty,r)
    u.write(REPORT/'calibration.json',dict(families=rows,family_ids=[r['id'] for r in rows],simplex_weights=weights.tolist(),
        simplex_gram=gram.tolist(),simplex_rhs=rhs.tolist(),ridge_weights=ridge[:-1].tolist(),ridge_intercept=float(ridge[-1]),
        ridge_gram=g.tolist(),ridge_rhs=r.tolist(),ridge_lambda=10.,source_sha256=bindings,
        known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    for name,coefficients,intercept in [('known-simplex',weights,0.),('known-ridge10',ridge[:-1],ridge[-1])]:
        combine(arrays,coefficients,intercept).tofile(RUN/(name+'.f32'))
    half=combine([arrays[-1],arrays[5]],[.5,.5]);half.tofile(RUN/'known-fixed-half.f32')
    if sha(RUN/'known-fixed-half.f32')!=sha(KNOWN/'fixed-half-tree154.f32'):raise ValueError('Original known fixed half must match every bit')
    verify_bindings(bindings)
    u.write(REPORT/'known-completion.json',dict(state='complete_original14_known_only_coefficients_before_new_bag',
        observed=stamp(),plan_sha256=sha(REPORT/'plan.json'),calibration_sha256=sha(REPORT/'calibration.json'),
        known_predictions={p.stem:sha(p) for p in RUN.glob('known-*.f32')},qualifying_labels_read=False))


def freeze():
    if (REPORT/'frozen.json').exists():raise ValueError('Original own14 freeze retained')
    done=u.read(REPORT/'known-completion.json');cal=u.read(REPORT/'calibration.json');plan=u.read(REPORT/'plan.json')
    if (done['state']!='complete_original14_known_only_coefficients_before_new_bag' or
        done['calibration_sha256']!=sha(REPORT/'calibration.json') or done['plan_sha256']!=sha(REPORT/'plan.json') or done['qualifying_labels_read'] is not False or
        set(done['known_predictions'])!={'known-simplex','known-ridge10','known-fixed-half'} or
        any(sha(RUN/(name+'.f32'))!=digest for name,digest in done['known_predictions'].items())):
        raise ValueError('Original known-only coefficient freeze required')
    cp=BAG/'independent-verification.json';proof=u.read(cp)
    if (proof['state']!='complete_independent_matched391_actual_fold_models_all_query_streams_and_bag' or
        proof['actual_models_reopened']!=15 or proof['seed_slots']!=15 or proof['rows']!=COUNTS['qual'] or
        not proof['all_original_model_and_input_streams_replayed'] or not proof['uniform_five_fold_bag_bitwise_equal'] or
        not proof['matched_B_raw_operators_on_both_queries'] or not proof['known_OOF_and_qualifying_bag_constructions_distinct'] or
        proof['qualifying_labels_read'] is not False or proof['models_fitted']!=0 or proof['qualifying_score_claimed'] is not False):
        raise ValueError('Complete original15-model/all-stream qualifying replay required')
    bindings=dict(cal['source_sha256'])
    for rel,digest in proof['source_sha256'].items():bind(bindings,u.ROOT/rel,digest)
    bind(bindings,cp);bind(bindings,BAG_RUN/'nn-bag.f32',proof['prediction_sha256'])
    rows=cal['families'];arrays=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
    if len(rows)!=14 or any(a.shape!=(COUNTS['qual'],) or not np.isfinite(a).all() for a in arrays):raise ValueError('All complete finite qualifying families required')
    outputs=[combine(arrays,cal['simplex_weights']),combine(arrays,cal['ridge_weights'],cal['ridge_intercept']),combine([arrays[-1],arrays[5]],[.5,.5])]
    for name,values in zip(CANDIDATES,outputs):values.tofile(RUN/(name+'.f32'))
    verify_bindings(bindings)
    u.write(REPORT/'frozen.json',dict(state='complete_all_three_own14_candidates_frozen_before_any_new_audit',observed=stamp(),
        primary=CANDIDATES[0],plan_sha256=sha(REPORT/'plan.json'),calibration_sha256=sha(REPORT/'calibration.json'),
        known_completion_sha256=sha(REPORT/'known-completion.json'),source_sha256=bindings,
        predictions={p.stem:sha(p) for p in RUN.glob('*.f32')},qualifying_labels_read=False,
        independent_coefficient_and_arithmetic_verification_remaining=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['known','freeze']);args=parser.parse_args()
    known() if args.phase=='known' else freeze()
