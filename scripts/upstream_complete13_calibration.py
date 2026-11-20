#!/usr/bin/env python3
"""Known-only simplex and released-default ridge across all13 declared own families."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import upstream_own352_fold_bag_assembly as current
import upstream_own330_assembly as preceding
import upstream_nn_pool6_assembly as pool

u,own6=current.u,current.own6
REPORT=u.ROOT/'experiments/upstream-complete13-calibration'
RUN=u.ROOT/'artifacts/upstream-complete13-calibration'
CANDIDATES=['complete13-simplex','complete13-ridge10']


def checked_sources():
    rows,new,_,sources=pool.checked_sources()
    previous,previous_sources=preceding.checked_sources()
    latest,latest_sources=current.checked_sources()
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));value=u.sha(path) if expected is None else expected
        if key in sources and sources[key]!=value:raise ValueError('Original component expectation conflicts: '+key)
        sources[key]=value
    for collection in [previous_sources,latest_sources]:
        for key,value in collection.items():bind(u.ROOT/key,value)
    aliases=[]
    for entry in new:
        name=entry['development'];known=u.ROOT/'artifacts'/name/'nn.f32';qual=u.ROOT/entry['qualifying_prediction']
        if str(known.relative_to(u.ROOT)) not in sources or str(qual.relative_to(u.ROOT)) not in sources:
            raise ValueError('Complete native member known/full vectors must retain original expected hashes')
        existing=next((r for r in rows if r['development']==name and r['family']=='nn'),None)
        if existing:
            if (u.sha(known)!=u.sha(u.ROOT/existing['known_prediction']) or
                    u.sha(qual)!=u.sha(u.ROOT/existing['qualifying_prediction'])):
                raise ValueError('Repeated original base must be the same complete numerical pair')
            aliases.append(dict(development=name,alias_of=existing['id']))
        else:
            rows.append(dict(id='native:'+name,family='nn',development=name,
                production='upstream-nn-pool6-full-refit/'+name,
                known_prediction=str(known.relative_to(u.ROOT)),qualifying_prediction=str(qual.relative_to(u.ROOT))))
    rows.append(next(r for r in previous if r['id']=='nn330'))
    rows.append(next(r for r in latest if r['id']=='nn352-fold-bag'))
    if len(rows)!=13 or len(aliases)!=1 or len({r['id'] for r in rows})!=13:
        raise ValueError('Exactly all13 declared distinct original families and one base alias required')
    canonical=u.ROOT/'artifacts/upstream-item154-tree'
    for row in rows:
        origin=u.ROOT/'artifacts'/row['development']
        for key in ['labels.npy','users.npy','groups.npy']:
            if not np.array_equal(np.load(origin/key),np.load(canonical/key)):
                raise ValueError('Every component must retain complete canonical rows and user folds')
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_complete13_calibration.py',
        u.ROOT/'ALL_COMPLETE13_CALIBRATION_PLAN.md',u.ROOT/'artifacts/upstream-repro/fresh/src/bin/ridge.rs']:
        bind(path)
    own6.verify_bindings(sources)
    return rows,sources,aliases


def combined(arrays,weights,intercept=0.):
    result=np.empty(len(arrays[0]),dtype='<f4')
    for lo in range(0,len(result),65536):
        hi=min(lo+65536,len(result));values=np.full(hi-lo,float(intercept),dtype=np.float64)
        for array,weight in zip(arrays,weights):values+=array[lo:hi].astype(np.float64)*float(weight)
        result[lo:hi]=np.clip(values,1.05,4.95).astype('<f4')
    return result


def freeze():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing complete13 evidence retained')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(created=own6.stamp(),candidates=CANDIDATES,primary=CANDIDATES[0],
        declaration_sha256=u.sha(u.ROOT/'ALL_COMPLETE13_CALIBRATION_PLAN.md'),
        script_sha256=u.sha(Path(__file__)),verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_complete13_calibration.py'),
        rule='All13 declared completed families; one exact base alias. Complete known OOF only. Primary simplex uniform SLSQP ftol1e-17/maxiter1000; control signed ridge lambda10 with unpenalized intercept. Independently solve8191 supports and full14dim normal equations. Both studies freeze/preverify before any new recovered audit.',
        ridge_lambda=10.,known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    rows,sources,aliases=checked_sources()
    known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows]
    y=np.load(own6.LABELS).astype(np.float64)
    if any(a.shape!=(1128164,) or not np.isfinite(a).all() for a in known):raise ValueError('Complete finite known components required')
    x=np.column_stack(known).astype(np.float64);d=x-x[:,-1,None];residual=y-x[:,-1]
    gram=d.T@d/len(y);rhs=d.T@residual/len(y);size=13
    opt=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(size,1/size),jac=lambda w:2*(gram@w-rhs),
        bounds=[(0.,1.)]*size,constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(size))],
        method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
    if not opt.success or opt.x.min()<-1e-12 or abs(opt.x.sum()-1)>1e-10:raise ValueError('Complete known simplex did not converge')
    weights=np.maximum(opt.x,0);weights/=weights.sum()
    z=np.column_stack([x,np.ones(len(y))]);g=z.T@z;r=z.T@y
    penalty=np.eye(14)*10;penalty[-1,-1]=0
    ridge=np.linalg.solve(g+penalty,r)
    u.write(REPORT/'calibration.json',dict(families=rows,family_ids=[r['id'] for r in rows],aliases=aliases,
        simplex_weights=weights.tolist(),simplex_gram=gram.tolist(),simplex_rhs=rhs.tolist(),
        ridge_weights=ridge[:-1].tolist(),ridge_intercept=float(ridge[-1]),ridge_gram=g.tolist(),ridge_rhs=r.tolist(),
        ridge_lambda=10.,source_sha256=sources,known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    combined(known,weights).tofile(RUN/'known-simplex.f32')
    combined(known,ridge[:-1],ridge[-1]).tofile(RUN/'known-ridge10.f32')
    del x,d,z,residual
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
    if any(a.shape!=(2817131,) or not np.isfinite(a).all() for a in full):raise ValueError('Complete finite full components required')
    combined(full,weights).tofile(RUN/(CANDIDATES[0]+'.f32'))
    combined(full,ridge[:-1],ridge[-1]).tofile(RUN/(CANDIDATES[1]+'.f32'))
    own6.verify_bindings(sources)
    u.write(REPORT/'frozen.json',dict(state='complete_both_complete13_candidates_frozen_before_any_new_audit',
        observed=own6.stamp(),plan_sha256=u.sha(REPORT/'plan.json'),calibration_sha256=u.sha(REPORT/'calibration.json'),
        predictions={p.stem:u.sha(p) for p in RUN.glob('*.f32')},qualifying_labels_read=False,
        independent_coefficient_and_arithmetic_verification_remaining=True))


if __name__=='__main__':freeze()
