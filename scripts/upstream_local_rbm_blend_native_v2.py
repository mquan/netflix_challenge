"""Known-only sixteen-family calibration adding the complete own local RBM.

Production dispatch is exclusively through the source-bound owned queue.
This module has no numerical or label-reading side effect on import.
"""
from contextlib import contextmanager,ExitStack
import os
from pathlib import Path
import numpy as np
import upstream_factored_rbm_execution_native_v2 as rbm
import upstream_factored_rbm_inputs as inputs
import upstream_factored_rbm_admission as admission
import upstream_factored_rbm_terminal as terminal
import upstream_own15_complete_trajectory_calibration as preceding

ROOT=rbm.ROOT
REPORT=ROOT/'experiments/upstream-local-rbm-blend-native-observer-v2'
ART=ROOT/'artifacts/upstream-local-rbm-blend-native-observer-v2'
DECL=ROOT/'OWN_LOCAL_RBM_NATIVE_OBSERVER_V2_BLEND_PLAN.md'
COUNTS={'known':1128164,'qualifying':2817131}
CANDIDATES=('local-rbm16-simplex','local-rbm16-ridge10','preceding-own15-simplex')
BASE_IDS=preceding.BASE_IDS+['nn-complete-trajectory-fold-bag']
FAMILY_ID='own-local-factored-bipolar-rbm400-full15'


def bind(bindings,path,digest=None):
    path=Path(path);relative=str(path.relative_to(ROOT))
    value=inputs.sha(path) if digest is None else digest
    if relative in bindings and bindings[relative]!=value:raise ValueError('Conflicting original source expectation')
    bindings[relative]=value


def verify_bindings(bindings,guard):
    # Native source trees contain only their previously declared synthetic
    # zero-target CSV. Recovered CSV bytes are excluded from fitting closure.
    synthetic=rbm.read(preceding.u.BASE/'fresh-training-plan.json')['synthetic_qualifying_payload_sha256']
    allowed={'artifacts/'+name+'/source/data/qual_ratings/qual_ratings.csv.gz'
             for name in ('upstream-original1200','upstream-high-capacity-original')}
    for name,digest in bindings.items():
        if 'qual_ratings.csv' in name and (name not in allowed or digest!=synthetic):
            raise ValueError('Recovered qualifying targets cannot enter fitting closure')
    return inputs.verify_files(ROOT,bindings,guard)


def gpu_sources(bindings):
    """Require actual full-budget producer/replay proofs, never CPU test flags."""
    rbm.check_sources(rbm.read(rbm.REPORT/'plan.json'))
    plan_sha=inputs.sha(rbm.REPORT/'plan.json');done=rbm.read(rbm.REPORT/'completion.json')
    paired_path=rbm.STUDY/'paired-verification.json';paired=rbm.read(paired_path)
    if (done['state']!='complete_full_own_gpu_rbm400_original15_15_and_independent_queries'
            or done['verification_sha256']!=inputs.sha(paired_path) or done['qualifying_labels_read'] is not False):
        raise ValueError('Complete actual full15+15 own RBM producer required')
    rbm.validate_result('assemble',paired,plan_sha)
    preflight_sha=inputs.sha(rbm.STUDY/'mps-preflight.json')
    if (paired['configuration']!=rbm.read(rbm.REPORT/'plan.json')['configuration'] or paired['models_fitted']!=2
            or paired['epochs_each']!=15 or paired['score_computed'] is not False or paired['preflight_sha256']!=preflight_sha
            or paired['known']['rows']!=COUNTS['known'] or paired['qualifying']['rows']!=COUNTS['qualifying']):
        raise ValueError('Full original paired model population/recipe required')
    plan=rbm.read(rbm.REPORT/'plan.json')
    for name,digest in plan['source_sha256'].items():bind(bindings,ROOT/name,digest)
    for phase in rbm.PHASES:
        path=rbm.STUDY/rbm.RESULTS[phase][0];record=rbm.read(path)
        rbm.validate_result(phase,record,plan_sha);bind(bindings,path)
    data=rbm.read(rbm.STUDY/'data-admission.json')
    if paired['data_admission_sha256']!=inputs.sha(rbm.STUDY/'data-admission.json'):
        raise ValueError('Original full63 data proof required')
    for name,record in data['bound_file_records'].items():bind(bindings,ROOT/name,record['sha256'])
    for phase in ('known','qualifying'):
        fit_path=rbm.STUDY/(phase+'-fit.json');replay_path=rbm.STUDY/(phase+'-replay.json')
        fit,replay=rbm.read(fit_path),rbm.read(replay_path)
        if (replay['fit_sha256']!=inputs.sha(fit_path) or replay['data_admission_sha256']!=paired['data_admission_sha256']
                or paired['replay_sha256'][str(replay_path.relative_to(ROOT))]!=inputs.sha(replay_path)
                or fit['configuration']!=plan['configuration'] or replay['configuration']!=plan['configuration'] or fit['epochs']!=15
                or fit['preflight_sha256']!=preflight_sha or fit['data_admission_sha256']!=paired['data_admission_sha256']
                or fit['native_rows']!=(1760494 if phase=='known' else COUNTS['qualifying']) or replay['native_rows']!=fit['native_rows']
                or replay['all_native_prediction_bits_match'] is not True
                or replay['epoch_sha256']!=fit['epoch_sha256']
                or set(fit['epoch_sha256'])!={str((rbm.STUDY/('%s-epoch-%02d.json'%(phase,n))).relative_to(ROOT)) for n in range(1,16)}
                or replay['actual_model_payloads']!=paired['actual_model_payloads'][phase]
                or (phase=='known' and replay['known_projection_bits_match'] is not True)):
            raise ValueError('Every complete actual model/epoch/native query replay required')
        for name,digest in fit['epoch_sha256'].items():bind(bindings,ROOT/name,digest)
        for name,record in replay['actual_model_payloads'].items():bind(bindings,ROOT/name,record['sha256'])
        for key in ('model','native_predictions','known_predictions'):
            row=fit.get(key)
            if row is not None and (str(Path(row['path']).relative_to(ROOT)) not in replay['actual_model_payloads']
                    or replay['actual_model_payloads'][str(Path(row['path']).relative_to(ROOT))]['sha256']!=row['sha256']):
                raise ValueError('Original complete learned predictor and all prediction payloads required')
    for name in ('known','qualifying'):
        expected=rbm.ART/('known.npy' if name=='known' else 'qualifying-native.npy')
        if Path(paired[name]['path'])!=expected:raise ValueError('Exact owned paired vector paths required')
        bind(bindings,expected,paired[name]['sha256'])
    for path in (rbm.REPORT/'plan.json',rbm.REPORT/'completion.json',paired_path):bind(bindings,path)
    return dict(id=FAMILY_ID,family='own-factored-bipolar-rbm',known_prediction=str((rbm.ART/'known.npy').relative_to(ROOT)),
                qualifying_prediction=str((rbm.ART/'qualifying-native.npy').relative_to(ROOT)),format='npy')


def source_declaration():
    old=preceding.REPORT
    p,c,f,pre=[rbm.read(old/name) for name in ('plan.json','calibration.json','frozen.json','independent-preaudit-verification.json')]
    if (p['families']!=15 or c['family_ids']!=BASE_IDS or [r['id'] for r in c['families']]!=BASE_IDS
            or c['qualifying_labels_read'] is not False or c['known_score_is_nested_cross_validation'] is not False
            or pre['state']!='complete_independent_all32767_supports_ridge16_and_six_vectors_before_audit'
            or pre['calibration_sha256']!=inputs.sha(old/'calibration.json') or pre['freeze_sha256']!=inputs.sha(old/'frozen.json')
            or pre['verifier_sha256']!=inputs.sha(ROOT/'scripts/verify_upstream_own15_complete_trajectory_calibration.py')
            or pre['simplex_supports_examined']!=32767 or pre['ridge_equations']!=16 or pre['qualifying_labels_read'] is not False
            or pre['all_six_candidate_vectors_bitwise'] is not True or f['source_sha256']!=pre['source_sha256']
            or f['plan_sha256']!=inputs.sha(old/'plan.json') or f['calibration_sha256']!=inputs.sha(old/'calibration.json')
            or f['qualifying_labels_read'] is not False or any(f['source_sha256'].get(k)!=v for k,v in c['source_sha256'].items())):
        raise ValueError('All15 own original families and full label-free preaudit required')
    bindings=dict(f['source_sha256'])
    for name,digest in f['predictions'].items():bind(bindings,preceding.RUN/(name+'.f32'),digest)
    for path in (old/name for name in ('plan.json','calibration.json','frozen.json','known-completion.json','independent-preaudit-verification.json')):bind(bindings,path)
    row=gpu_sources(bindings)
    labels=preceding.KNOWN/'labels.npy'
    if str(labels.relative_to(ROOT)) not in bindings:raise ValueError('Original admitted known labels required')
    from upstream_local_rbm_blend_queue_native_v2 import REPORT as queue_report
    for path in (Path(__file__),ROOT/'scripts/verify_upstream_local_rbm_blend_native_v2.py',ROOT/'scripts/upstream_local_rbm_blend_queue_native_v2.py',DECL,
                 ROOT/'scripts/upstream_own15_complete_trajectory_calibration.py',ROOT/'scripts/verify_upstream_own16_weekend_trajectory_calibration.py'):
        bind(bindings,path)
    for name in ('plan.json','monitor.json','first-scientific-baseline.json','prerequisites.json','known-gate.json','known-ownership.json'):
        bind(bindings,queue_report/name)
    return list(c['families'])+[row],bindings


def sources(guard):
    rows,bindings=source_declaration()
    return rows,bindings,verify_bindings(bindings,guard)


@contextmanager
def vector(relative,count,records,fmt='raw'):
    path=ROOT/relative;record=records[relative]
    if fmt=='npy':
        with admission.mapped(path,record) as array:
            if array.dtype!=np.dtype('<f4') or array.shape!=(count,) or path.stat().st_size!=array.offset+array.nbytes:
                raise ValueError('Exact complete native f32 vector required')
            if not np.isfinite(array).all():raise ValueError('Finite full prediction required')
            yield array
    elif fmt=='raw':
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno())
            if admission.file_identity(before)!=record['identity'] or before.st_size!=count*4:
                raise ValueError('Full originally hashed raw f32 population required')
            array=np.memmap(stream,dtype='<f4',mode='r',shape=(count,))
            try:
                if not np.isfinite(array).all():raise ValueError('Finite full prediction required')
                yield array
            finally:
                array._mmap.close()
                if (admission.file_identity(before)!=admission.file_identity(os.fstat(stream.fileno()))
                        or admission.file_identity(before)!=admission.file_identity(path.stat())):
                    raise ValueError('Original raw vector changed during mapped use')
    else:raise ValueError('Only owned raw or native f32 vectors permitted')


@contextmanager
def columns(rows,query,records,guard):
    if len(rows)!=16 or [r['id'] for r in rows]!=BASE_IDS+[FAMILY_ID]:raise ValueError('Every original16 family required')
    with ExitStack() as stack:
        arrays=[]
        for row in rows:
            guard();arrays.append(stack.enter_context(vector(row[query+'_prediction'],COUNTS[query],records,row.get('format','raw'))))
        yield arrays


def save_raw(name,values,guard):
    return terminal.durable_payload(ART/(name+'.f32'),lambda stream:stream.write(memoryview(values)),ROOT,guard)


def combine(arrays,weights,guard,intercept=0.):
    output=np.empty(len(arrays[0]),dtype='<f4')
    if len(arrays)!=len(weights):raise ValueError('Every original coefficient required')
    for lo in range(0,len(output),32768):
        guard();hi=min(lo+32768,len(output));values=np.full(hi-lo,float(intercept),dtype=np.float64)
        for column,weight in zip(arrays,weights):values+=column[lo:hi].astype(np.float64)*float(weight)
        output[lo:hi]=np.clip(values,1.05,4.95).astype('<f4')
    return output


def known(guard):
    from scipy.optimize import minimize
    rows,bindings,records=sources(guard)
    plan=dict(state='declared_own16_local_rbm_known_only_simplex_ridge10_and_original_control',primary=CANDIDATES[0],
        candidates=list(CANDIDATES),families=16,ridge_lambda=10.,source_sha256=bindings,
        qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
    rbm.exclusive_json(REPORT/'plan.json',plan,guard)
    with columns(rows,'known',records,guard) as arrays,admission.mapped(preceding.KNOWN/'labels.npy',records[str((preceding.KNOWN/'labels.npy').relative_to(ROOT))]) as labels:
        if labels.shape!=(COUNTS['known'],) or not np.isin(labels,[1,2,3,4,5]).all():raise ValueError('Original complete known labels required')
        y=labels.astype(np.float64);x=np.column_stack(arrays).astype(np.float64);guard()
        d=x-x[:,-1,None];residual=y-x[:,-1];gram=d.T@d/len(y);rhs=d.T@residual/len(y);guard()
        result=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(16,1/16),jac=lambda w:2*(gram@w-rhs),bounds=[(0.,1.)]*16,
            constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(16))],method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
        if not result.success or result.x.min()<-1e-12 or abs(result.x.sum()-1)>1e-10:raise ValueError('Full original simplex convergence required')
        weights=np.maximum(result.x,0);weights/=weights.sum();z=np.column_stack([x,np.ones(len(y))]);raw=z.T@z;target=z.T@y
        penalty=np.eye(17)*10.;penalty[-1,-1]=0;ridge=np.linalg.solve(raw+penalty,target);guard()
        cal=dict(families=rows,family_ids=[r['id'] for r in rows],simplex_weights=weights.tolist(),simplex_gram=gram.tolist(),simplex_rhs=rhs.tolist(),
            ridge_weights=ridge[:-1].tolist(),ridge_intercept=float(ridge[-1]),ridge_gram=raw.tolist(),ridge_rhs=target.tolist(),ridge_lambda=10.,
            source_sha256=bindings,source_file_records=records,qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
        rbm.exclusive_json(REPORT/'calibration.json',cal,guard)
        outputs={}
        for name,coefficients,bias in [('known-simplex',weights,0.),('known-ridge10',ridge[:-1],ridge[-1])]:
            outputs[name]=save_raw(name,combine(arrays,coefficients,guard,bias),guard)
    relative=str((preceding.RUN/'known-simplex.f32').relative_to(ROOT))
    with vector(relative,COUNTS['known'],records) as control:outputs['known-control']=save_raw('known-control',control,guard)
    rbm.exclusive_json(REPORT/'known-completion.json',dict(state='complete_original16_local_rbm_known_only_coefficients',
        plan_sha256=inputs.sha(REPORT/'plan.json'),calibration_sha256=inputs.sha(REPORT/'calibration.json'),outputs=outputs,qualifying_labels_read=False),guard)


def frozen(guard):
    p,c,done=[rbm.read(REPORT/name) for name in ('plan.json','calibration.json','known-completion.json')]
    if (done['state']!='complete_original16_local_rbm_known_only_coefficients' or done['plan_sha256']!=inputs.sha(REPORT/'plan.json')
            or done['calibration_sha256']!=inputs.sha(REPORT/'calibration.json') or done['qualifying_labels_read'] is not False
            or set(done['outputs'])!={'known-simplex','known-ridge10','known-control'}):
        raise ValueError('Original complete known-only fit required before freeze')
    for name,row in done['outputs'].items():
        if Path(row['path'])!=ART/(name+'.f32') or inputs.sha(ART/(name+'.f32'))!=row['sha256']:raise ValueError('Original known output changed')
    records=verify_bindings(c['source_sha256'],guard);outputs=dict(done['outputs'])
    with columns(c['families'],'qualifying',records,guard) as arrays:
        outputs[CANDIDATES[0]]=save_raw(CANDIDATES[0],combine(arrays,c['simplex_weights'],guard),guard)
        outputs[CANDIDATES[1]]=save_raw(CANDIDATES[1],combine(arrays,c['ridge_weights'],guard,c['ridge_intercept']),guard)
    relative=str((preceding.RUN/(preceding.CANDIDATES[0]+'.f32')).relative_to(ROOT))
    with vector(relative,COUNTS['qualifying'],records) as control:outputs[CANDIDATES[2]]=save_raw(CANDIDATES[2],control,guard)
    rbm.exclusive_json(REPORT/'frozen.json',dict(state='complete_all_three_local_rbm16_candidates_frozen_before_recovered_audit',
        primary=CANDIDATES[0],plan_sha256=inputs.sha(REPORT/'plan.json'),calibration_sha256=inputs.sha(REPORT/'calibration.json'),
        known_completion_sha256=inputs.sha(REPORT/'known-completion.json'),source_sha256=c['source_sha256'],outputs=outputs,
        qualifying_labels_read=False,independent_verification_remaining=True),guard)


def main(phase):
    import upstream_local_rbm_blend_queue_native_v2 as queue
    queue.authorize(phase);guard=queue.Guard(phase);guard()
    terminal.durable_directory(REPORT,ROOT,guard);terminal.durable_directory(ART,ROOT,guard)
    rbm.exclusive_json(REPORT/(phase+'-started.json'),dict(phase=phase,pid=os.getpid(),observed=rbm.stamp()),guard)
    if phase=='known':known(guard)
    elif phase=='freeze':frozen(guard)
    else:raise ValueError('Fixed known/freeze phase required')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('known','freeze'));main(parser.parse_args().phase)
