"""Known-only seventeen-family calibration adding the verified selected NN-only bag.

Production dispatch is exclusively through the source-bound owned queue.
This module has no numerical or label-reading side effect on import.
"""
from contextlib import contextmanager,ExitStack
import os
from pathlib import Path
import numpy as np
import upstream_factored_rbm_execution_metal as rbm
import upstream_factored_rbm_inputs as inputs
import upstream_factored_rbm_admission as admission
import upstream_factored_rbm_terminal_metal as terminal
import upstream_local_rbm_blend_metal as preceding
import upstream_rbm400_bag as bag
import verify_upstream_rbm400_bag as bag_replay
import upstream_rbm400_bag_execution as bag_owner

ROOT=rbm.ROOT
REPORT=ROOT/'experiments/upstream-rbm400-own17-calibration'
ART=ROOT/'artifacts/upstream-rbm400-own17-calibration'
DECL=ROOT/'OWN_RBM400_OWN17_CALIBRATION_PLAN.md'
COUNTS={'known':1128164,'qualifying':2817131}
CANDIDATES=('rbm400-own17-simplex','rbm400-own17-ridge10','selected-NN-tree154-fixed-half')
BASE_IDS=preceding.BASE_IDS+[preceding.FAMILY_ID]
FAMILY_ID='own-rbm400-selected-context-neural-fold-bag'
KNOWN=preceding.preceding.KNOWN


def bind(bindings,path,digest=None):
    path=Path(path);relative=str(path.relative_to(ROOT))
    value=inputs.sha(path) if digest is None else digest
    if relative in bindings and bindings[relative]!=value:raise ValueError('Conflicting original source expectation')
    bindings[relative]=value


def verify_bindings(bindings,guard):
    # Native source trees contain only their previously declared synthetic
    # zero-target CSV. Recovered CSV bytes are excluded from fitting closure.
    synthetic=rbm.read(preceding.preceding.u.BASE/'fresh-training-plan.json')['synthetic_qualifying_payload_sha256']
    allowed={'artifacts/'+name+'/source/data/qual_ratings/qual_ratings.csv.gz'
             for name in ('upstream-original1200','upstream-high-capacity-original')}
    for name,digest in bindings.items():
        if 'qual_ratings.csv' in name and (name not in allowed or digest!=synthetic):
            raise ValueError('Recovered qualifying targets cannot enter fitting closure')
    return inputs.verify_files(ROOT,bindings,guard)


def source_declaration(guard):
    source=bag.sources(guard)
    proof_path=bag.REPORT/'independent-verification.json'
    receipt_path=bag_owner.REPORT/'completion.json'
    proof,receipt=[rbm.read(p) for p in (proof_path,receipt_path)]
    if (receipt['state']!='complete_selected_RBM400_qualifying_bag_and_independent_full_query_replay'
            or receipt['verification_sha256']!=inputs.sha(proof_path) or receipt['qualifying_labels_read'] is not False
            or proof['state']!=bag_replay.STATE or proof['rows']!=COUNTS['qualifying']
            or proof['source_sha256']!=source['source_sha256'] or proof['qualifying_labels_read'] is not False
            or proof['models_fitted']!=0 or proof['actual_MPS_predictions_replayed'] is not True
            or proof['all_five_folds_and_three_seed_slots'] is not True
            or proof['provider_is_NN_only_on_both_queries'] is not True
            or proof['every_full_query_operator_raw_normalized_seed_fold_and_output_bit_verified'] is not True
            or proof['known_NN_provider']!=source['roles']['known_NN_provider']
            or proof['known_fixed_half_control']!=source['roles']['known_fixed_half_control']
            or proof['actual_models_reopened']!=source['actual_models']):
        raise ValueError('Complete actual selected NN-only paired source/model/query replay required')
    cal_path=preceding.REPORT/'calibration.json';cal=rbm.read(cal_path)
    if cal['family_ids']!=BASE_IDS or [r['id'] for r in cal['families']]!=BASE_IDS:
        raise ValueError('All sixteen preceding own families in original order required')
    bindings=dict(source['source_sha256'])
    if bindings.get(str(cal_path.relative_to(ROOT)))!=inputs.sha(cal_path):
        raise ValueError('Original independently completed own16 calibration expectation required')
    for name in ('plan.json','inference-completion.json'):
        path=bag.REPORT/name;expected=proof['plan_sha256' if name=='plan.json' else 'inference_sha256']
        bind(bindings,path,expected)
    known=proof['known_NN_provider'];qual=proof['qualifying_NN_provider']
    if Path(qual['path'])!=bag.ART/'NN-bag.npy':raise ValueError('Exact full NN-only query bag required')
    for row in (known,qual,proof['known_fixed_half_control'],proof['qualifying_fixed_half_control']):
        bind(bindings,Path(row['path']),row['sha256'])
    row=dict(id=FAMILY_ID,family='own-selected-context-neural',known_prediction=str(Path(known['path']).relative_to(ROOT)),
        qualifying_prediction=str(Path(qual['path']).relative_to(ROOT)),format='npy',selected_arm=source['arm'])
    import upstream_rbm400_own17_execution as queue
    queue_report=queue.REPORT
    own_plan=queue.read(queue_report/'plan.json');queue.check_sources(own_plan)
    for name,digest in own_plan['source_sha256'].items():bind(bindings,ROOT/name,digest)
    for path in (proof_path,receipt_path,Path(__file__),DECL,ROOT/'scripts/verify_upstream_rbm400_own17_calibration.py',
                 ROOT/'scripts/upstream_rbm400_own17_execution.py',ROOT/'experiments/upstream-rbm400-own17-execution-protocol/verification.json'):
        bind(bindings,path)
    for name in ('plan.json','monitor.json','first-scientific-baseline.json','prerequisites.json','known-gate.json','known-ownership.json'):
        bind(bindings,queue_report/name)
    if str((KNOWN/'labels.npy').relative_to(ROOT)) not in bindings:raise ValueError('Original admitted known outcomes required')
    return list(cal['families'])+[row],bindings


def controls():
    proof=rbm.read(bag.REPORT/'independent-verification.json')
    return {query:proof[key] for query,key in [('known','known_fixed_half_control'),('qualifying','qualifying_fixed_half_control')]}


def sources(guard):
    rows,bindings=source_declaration(guard)
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
    if len(rows)!=17 or [r['id'] for r in rows]!=BASE_IDS+[FAMILY_ID]:raise ValueError('Every original17 family required')
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


def polish_simplex(gram,rhs,start,guard):
    """Solve active faces of the same known-only QP to full KKT precision.

    SLSQP can declare success with tiny objective error but inadequate reduced
    gradients. This changes numerical solution precision, never the objective,
    families, constraints or independent exhaustive verification tolerances.
    """
    if gram.shape!=(17,17) or rhs.shape!=(17,) or start.shape!=(17,):
        raise ValueError('All17 original simplex coordinates required')
    weights=np.maximum(start,0.);weights[weights<1e-12]=0.;weights/=weights.sum()
    active=set(np.flatnonzero(weights))
    for iteration in range(1000):
        guard();ids=sorted(active);count=len(ids)
        system=np.zeros((count+1,count+1));system[:count,:count]=gram[np.ix_(ids,ids)]
        system[:count,-1]=1.;system[-1,:count]=1.;target=np.r_[rhs[ids],1.]
        try:solution=np.linalg.solve(system,target)
        except np.linalg.LinAlgError:solution=np.linalg.lstsq(system,target,rcond=None)[0]
        if not np.isfinite(solution).all() or np.max(abs(system@solution-target))>1e-10:
            raise ValueError('Original active simplex equality system must solve')
        face=np.zeros(17);face[ids]=solution[:-1]
        if face.min()<-1e-14:
            falling=[i for i in ids if face[i]<0.]
            alpha=min(weights[i]/(weights[i]-face[i]) for i in falling)
            weights+=alpha*(face-weights);weights[weights<1e-14]=0.;weights/=weights.sum()
            active={i for i in active if weights[i]>0.};continue
        weights=np.maximum(face,0.);weights/=weights.sum()
        gradient=gram@weights-rhs;reduced=gradient-float(weights@gradient)
        inactive=[i for i in range(17) if i not in active]
        if inactive:
            entering=min(inactive,key=lambda i:reduced[i])
            if reduced[entering]<-1e-12:active.add(entering);continue
        if np.min(reduced)<-1e-9 or np.max(abs(weights*reduced))>1e-10:
            raise ValueError('Full actual known-only simplex KKT precision required')
        return weights,dict(method='SLSQP_then_feasible_active_face_KKT_polish',active_face_solves=iteration+1,
            objective=float(weights@gram@weights-2*rhs@weights),minimum_reduced_gradient=float(reduced.min()),
            complementarity_error=float(np.max(abs(weights*reduced))),qualifying_labels_read=False)
    raise ValueError('Original simplex objective did not converge; no restart or relaxed tolerance')


def known(guard):
    from scipy.optimize import minimize
    rows,bindings,records=sources(guard)
    plan=dict(state='declared_own17_selected_RBM400_NN_known_only_simplex_ridge10_and_fixed_half_control',primary=CANDIDATES[0],
        candidates=list(CANDIDATES),families=17,ridge_lambda=10.,source_sha256=bindings,
        qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
    rbm.exclusive_json(REPORT/'plan.json',plan,guard)
    with columns(rows,'known',records,guard) as arrays,admission.mapped(KNOWN/'labels.npy',records[str((KNOWN/'labels.npy').relative_to(ROOT))]) as labels:
        if labels.shape!=(COUNTS['known'],) or not np.isin(labels,[1,2,3,4,5]).all():raise ValueError('Original complete known labels required')
        y=labels.astype(np.float64);x=np.column_stack(arrays).astype(np.float64);guard()
        d=x-x[:,-1,None];residual=y-x[:,-1];gram=d.T@d/len(y);rhs=d.T@residual/len(y);guard()
        result=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(17,1/17),jac=lambda w:2*(gram@w-rhs),bounds=[(0.,1.)]*17,
            constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(17))],method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
        if not result.success or result.x.min()<-1e-12 or abs(result.x.sum()-1)>1e-10:raise ValueError('Full original simplex convergence required')
        weights,solver_proof=polish_simplex(gram,rhs,result.x,guard)
        z=np.column_stack([x,np.ones(len(y))]);raw=z.T@z;target=z.T@y
        penalty=np.eye(18)*10.;penalty[-1,-1]=0;ridge=np.linalg.solve(raw+penalty,target);guard()
        cal=dict(families=rows,family_ids=[r['id'] for r in rows],simplex_weights=weights.tolist(),simplex_gram=gram.tolist(),simplex_rhs=rhs.tolist(),
            ridge_weights=ridge[:-1].tolist(),ridge_intercept=float(ridge[-1]),ridge_gram=raw.tolist(),ridge_rhs=target.tolist(),ridge_lambda=10.,
            source_sha256=bindings,source_file_records=records,simplex_solver=solver_proof,
            qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
        rbm.exclusive_json(REPORT/'calibration.json',cal,guard)
        outputs={}
        for name,coefficients,bias in [('known-simplex',weights,0.),('known-ridge10',ridge[:-1],ridge[-1])]:
            outputs[name]=save_raw(name,combine(arrays,coefficients,guard,bias),guard)
    relative=str(Path(controls()['known']['path']).relative_to(ROOT))
    with vector(relative,COUNTS['known'],records,'npy') as control:outputs['known-control']=save_raw('known-control',control,guard)
    rbm.exclusive_json(REPORT/'known-completion.json',dict(state='complete_original17_selected_RBM400_NN_known_only_coefficients',
        plan_sha256=inputs.sha(REPORT/'plan.json'),calibration_sha256=inputs.sha(REPORT/'calibration.json'),outputs=outputs,qualifying_labels_read=False),guard)


def frozen(guard):
    p,c,done=[rbm.read(REPORT/name) for name in ('plan.json','calibration.json','known-completion.json')]
    if (done['state']!='complete_original17_selected_RBM400_NN_known_only_coefficients' or done['plan_sha256']!=inputs.sha(REPORT/'plan.json')
            or done['calibration_sha256']!=inputs.sha(REPORT/'calibration.json') or done['qualifying_labels_read'] is not False
            or set(done['outputs'])!={'known-simplex','known-ridge10','known-control'}):
        raise ValueError('Original complete known-only fit required before freeze')
    for name,row in done['outputs'].items():
        if Path(row['path'])!=ART/(name+'.f32') or inputs.sha(ART/(name+'.f32'))!=row['sha256']:raise ValueError('Original known output changed')
    records=verify_bindings(c['source_sha256'],guard);outputs=dict(done['outputs'])
    with columns(c['families'],'qualifying',records,guard) as arrays:
        outputs[CANDIDATES[0]]=save_raw(CANDIDATES[0],combine(arrays,c['simplex_weights'],guard),guard)
        outputs[CANDIDATES[1]]=save_raw(CANDIDATES[1],combine(arrays,c['ridge_weights'],guard,c['ridge_intercept']),guard)
    relative=str(Path(controls()['qualifying']['path']).relative_to(ROOT))
    with vector(relative,COUNTS['qualifying'],records,'npy') as control:outputs[CANDIDATES[2]]=save_raw(CANDIDATES[2],control,guard)
    rbm.exclusive_json(REPORT/'frozen.json',dict(state='complete_all_three_selected_RBM400_own17_candidates_frozen_before_recovered_audit',
        primary=CANDIDATES[0],plan_sha256=inputs.sha(REPORT/'plan.json'),calibration_sha256=inputs.sha(REPORT/'calibration.json'),
        known_completion_sha256=inputs.sha(REPORT/'known-completion.json'),source_sha256=c['source_sha256'],outputs=outputs,
        qualifying_labels_read=False,independent_verification_remaining=True),guard)


def main(phase):
    import upstream_rbm400_own17_execution as queue
    queue.authorize(phase);guard=queue.WorkerGuard(phase);guard()
    terminal.durable_directory(REPORT,ROOT,guard);terminal.durable_directory(ART,ROOT,guard)
    rbm.exclusive_json(REPORT/(phase+'-started.json'),dict(phase=phase,pid=os.getpid(),observed=rbm.stamp()),guard)
    if phase=='known':known(guard)
    elif phase=='freeze':frozen(guard)
    else:raise ValueError('Fixed known/freeze phase required')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('known','freeze'));main(parser.parse_args().phase)
