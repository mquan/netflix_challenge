"""Known-only eighteen-family calibration adding the full selected tree pair.

Component only: no CLI or standalone numerical launch. The future reviewed
owned controller must supply source/phase authorization and resource guards.
"""
from contextlib import contextmanager,ExitStack
import os
from pathlib import Path
import numpy as np
import upstream_tree_own18_sources_v1 as provider

previous=provider.previous
rbm=previous.rbm
inputs=previous.inputs
admission=previous.admission
terminal=previous.terminal
ROOT=rbm.ROOT
REPORT=ROOT/'experiments/upstream-tree-own18-calibration-v1'
ART=ROOT/'artifacts/upstream-tree-own18-calibration-v1'
DECL=ROOT/'OWN_TREE_OWN18_CALIBRATION_PLAN_V1.md'
COUNTS=dict(provider.COUNTS)
CANDIDATES=provider.CANDIDATES
BASE_IDS=list(provider.BASE_IDS)
FAMILY_ID=provider.FAMILY_ID
KNOWN=previous.KNOWN


def verify_bindings(bindings,guard):
    return previous.verify_bindings(bindings,guard)


def source_declaration(guard):
    rows,bindings,control=provider.declaration(guard)
    import upstream_tree_own18_execution_v1 as queue
    plan=queue.read(queue.REPORT/'plan.json');queue.check_sources(plan)
    for name,digest in plan['source_sha256'].items():bind(bindings,ROOT/name,digest)
    for path in (Path(__file__),DECL,ROOT/'scripts/verify_upstream_tree_own18_calibration_v1.py',
            ROOT/'scripts/verify_upstream_tree_own18_arithmetic_v1.py',
            ROOT/'experiments/upstream-tree-own18-owned-protocol-v1/verification.json'):
        bind(bindings,path)
    for name in ('plan.json','monitor.json','first-scientific-baseline.json','prerequisites.json','known-gate.json','known-ownership.json'):
        bind(bindings,queue.REPORT/name)
    if str((KNOWN/'labels.npy').relative_to(ROOT)) not in bindings:
        raise ValueError('Original complete admitted known outcomes required')
    return rows,bindings


def controls():
    value=rbm.read(provider.tree_owner.SOURCE_REPORT/'pair-admission.json')
    return {query:dict(value['outputs'][query],format='raw') for query in ('known','qualifying')}


def bind(bindings,path,digest=None):
    path=Path(path);relative=str(path.relative_to(ROOT))
    value=inputs.sha(path) if digest is None else digest
    if relative in bindings and bindings[relative]!=value:raise ValueError('Conflicting original source expectation')
    bindings[relative]=value


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
    if len(rows)!=18 or [r['id'] for r in rows]!=BASE_IDS+[FAMILY_ID]:raise ValueError('Every original18 family required')
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
    if gram.shape!=(18,18) or rhs.shape!=(18,) or start.shape!=(18,):
        raise ValueError('All18 original simplex coordinates required')
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
        face=np.zeros(18);face[ids]=solution[:-1]
        if face.min()<-1e-14:
            falling=[i for i in ids if face[i]<0.]
            alpha=min(weights[i]/(weights[i]-face[i]) for i in falling)
            weights+=alpha*(face-weights);weights[weights<1e-14]=0.;weights/=weights.sum()
            active={i for i in active if weights[i]>0.};continue
        weights=np.maximum(face,0.);weights/=weights.sum()
        gradient=gram@weights-rhs;reduced=gradient-float(weights@gradient)
        inactive=[i for i in range(18) if i not in active]
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
    plan=dict(state='declared_own18_selected_full_context_tree_known_only_simplex_ridge10_and_standalone_tree_control',primary=CANDIDATES[0],
        candidates=list(CANDIDATES),families=18,ridge_lambda=10.,source_sha256=bindings,
        qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
    rbm.exclusive_json(REPORT/'plan.json',plan,guard)
    with columns(rows,'known',records,guard) as arrays,admission.mapped(KNOWN/'labels.npy',records[str((KNOWN/'labels.npy').relative_to(ROOT))]) as labels:
        if labels.shape!=(COUNTS['known'],) or not np.isin(labels,[1,2,3,4,5]).all():raise ValueError('Original complete known labels required')
        y=labels.astype(np.float64);x=np.column_stack(arrays).astype(np.float64);guard()
        d=x-x[:,-1,None];residual=y-x[:,-1];gram=d.T@d/len(y);rhs=d.T@residual/len(y);guard()
        result=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(18,1/18),jac=lambda w:2*(gram@w-rhs),bounds=[(0.,1.)]*18,
            constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(18))],method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
        if not result.success or result.x.min()<-1e-12 or abs(result.x.sum()-1)>1e-10:raise ValueError('Full original simplex convergence required')
        weights,solver_proof=polish_simplex(gram,rhs,result.x,guard)
        z=np.column_stack([x,np.ones(len(y))]);raw=z.T@z;target=z.T@y
        penalty=np.eye(19)*10.;penalty[-1,-1]=0;ridge=np.linalg.solve(raw+penalty,target);guard()
        cal=dict(families=rows,family_ids=[r['id'] for r in rows],simplex_weights=weights.tolist(),simplex_gram=gram.tolist(),simplex_rhs=rhs.tolist(),
            ridge_weights=ridge[:-1].tolist(),ridge_intercept=float(ridge[-1]),ridge_gram=raw.tolist(),ridge_rhs=target.tolist(),ridge_lambda=10.,
            source_sha256=bindings,source_file_records=records,simplex_solver=solver_proof,
            qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
        rbm.exclusive_json(REPORT/'calibration.json',cal,guard)
        outputs={}
        for name,coefficients,bias in [('known-simplex',weights,0.),('known-ridge10',ridge[:-1],ridge[-1])]:
            outputs[name]=save_raw(name,combine(arrays,coefficients,guard,bias),guard)
    relative=str(Path(controls()['known']['path']).relative_to(ROOT))
    with vector(relative,COUNTS['known'],records,'raw') as control:outputs['known-control']=save_raw('known-control',control,guard)
    rbm.exclusive_json(REPORT/'known-completion.json',dict(state='complete_original18_selected_full_context_tree_known_only_coefficients',
        plan_sha256=inputs.sha(REPORT/'plan.json'),calibration_sha256=inputs.sha(REPORT/'calibration.json'),outputs=outputs,qualifying_labels_read=False),guard)


def frozen(guard):
    p,c,done=[rbm.read(REPORT/name) for name in ('plan.json','calibration.json','known-completion.json')]
    if (done['state']!='complete_original18_selected_full_context_tree_known_only_coefficients' or done['plan_sha256']!=inputs.sha(REPORT/'plan.json')
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
    with vector(relative,COUNTS['qualifying'],records,'raw') as control:outputs[CANDIDATES[2]]=save_raw(CANDIDATES[2],control,guard)
    rbm.exclusive_json(REPORT/'frozen.json',dict(state='complete_all_three_selected_full_context_tree_own18_candidates_frozen_before_recovered_audit',
        primary=CANDIDATES[0],plan_sha256=inputs.sha(REPORT/'plan.json'),calibration_sha256=inputs.sha(REPORT/'calibration.json'),
        known_completion_sha256=inputs.sha(REPORT/'known-completion.json'),source_sha256=c['source_sha256'],outputs=outputs,
        qualifying_labels_read=False,independent_verification_remaining=True),guard)

