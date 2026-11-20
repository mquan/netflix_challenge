#!/usr/bin/env python3
"""Freeze the predeclared own NN pool, fixed ratio and known-only own7 fit."""
import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='2'
import datetime as dt
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import upstream_repro as u
import upstream_own6_tight_calibration as own6
import upstream_native_nn_pool6 as pool

RUN=u.ROOT/'artifacts/upstream-nn-pool6-assembly'
REPORT=u.ROOT/'experiments/upstream-nn-pool6-assembly'
FULL=u.ROOT/'experiments/upstream-nn-pool6-full-refit'
FULL_RUN=u.ROOT/'artifacts/upstream-nn-pool6-full-refit'
DECL=u.ROOT/'experiments/upstream-native-nn-pool6-production/declaration.json'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def checked_sources():
    records,sources,_=own6.checked_sources()
    def bind(p,digest=None):
        key=str(p.relative_to(u.ROOT));actual=u.sha(p)
        if digest is not None and actual!=digest:raise ValueError('Expected source differs: '+key)
        if key in sources and sources[key]!=actual:raise ValueError('Conflicting original source: '+key)
        sources[key]=actual
    admission_path=u.ROOT/'experiments/upstream-nn-pool6-admission-queue/completion.json'
    if u.read(admission_path)['state']!='complete_all_five_independent_full_pipeline_admissions':
        raise ValueError('Completed final admission queue required')
    execution_path=u.ROOT/'experiments/upstream-nn-pool6-full-refit-queue/plan.json'
    execution=u.read(execution_path)
    for key,p in [('queue_sha256',u.ROOT/'scripts/upstream_nn_pool6_full_refit_queue.py'),
        ('worker_sha256',u.ROOT/'scripts/upstream_nn_pool6_full_refit.py'),('declaration_sha256',DECL)]:
        bind(p,execution[key])
    for p in [admission_path,execution_path]:bind(p)
    bankgate=u.ROOT/'experiments/upstream-nn-pool6-original-bank-admission/verification.json'
    gate=u.read(bankgate)
    if gate['state']!='complete_original_bank_expectations_for_all_five_full_refits':raise ValueError('Original bank admission required')
    for collection in [gate['original_bank_expected_sha256'],gate['bindings']]:
        for name,digest in collection.items():bind(u.ROOT/name,digest)
    bind(bankgate)
    declared=u.read(DECL);bind(DECL)
    new=[]
    for entry in declared['families']:
        name=entry['development']
        if entry['reuse_existing_full_base']:
            base=next(r for r in records if r['id']=='nn154')
            new.append(dict(development=name,qualifying_prediction=base['qualifying_prediction']))
            continue
        report=FULL/name;root=FULL_RUN/name
        pp=report/'plan.json';vp=report/'verification.json';ip=report/'independent-verification.json'
        plan=u.read(pp);v=u.read(vp);ind=u.read(ip)
        gate_row=next(r for r in gate['records'] if r['development']==name)
        if gate_row['plan_sha256']!=u.sha(pp):raise ValueError('Original bank gate binds another full plan')
        if (ind['state']!='complete_independent_serialized_pipeline_and_original_lineage_admission' or
            ind['development']!=name or [r['query'] for r in ind['records']]!=['probex','qual'] or
            not all(r['bitwise_equal'] for r in ind['records']) or
            not ind['actual_saved_ridge_scaler_and_all_models_reopened'] or
            ind['qualifying_labels_read'] is not False):raise ValueError('Both complete independent pipeline replays required')
        for collection in [ind['bindings'],plan['source_sha256']]:
            for key,digest in collection.items():bind(u.ROOT/key,digest)
        if (v['plan_sha256']!=u.sha(pp) or plan['config']!=entry['config'] or
            plan['iterations']!=entry['full_iterations']):raise ValueError('Original declared full recipe differs')
        for filename,digest in v['outputs'].items():bind(root/filename,digest)
        for row in ind['records']:
            bind(root/(row['query']+'.f32'),row['prediction_sha256'])
        for p in [pp,vp,ip]:bind(p)
        new.append(dict(development=name,qualifying_prediction=str((root/'qual.f32').relative_to(u.ROOT))))
    if [r['development'] for r in new]!=pool.ORIGINS:raise ValueError('Exact fixed six-NN family order required')
    pr=u.ROOT/'experiments/upstream-native-nn-pool6'
    original=u.read(pr/'verification.json');canonical=u.read(pr/'independent-canonical-verification.json')
    if (canonical['state']!='complete_independent_canonical_fixed_six_native_nn_pool' or
        canonical['verification_sha256']!=u.sha(pr/'verification.json')):raise ValueError('Original fixed pool proof differs')
    for key,digest in original['source_sha256'].items():bind(u.ROOT/key,digest)
    known_pool=u.ROOT/'artifacts/upstream-native-nn-pool6/nn-pool.f32'
    digest=next(r['prediction_sha256'] for r in canonical['records'] if r['arm']=='nn-pool')
    bind(known_pool,digest)
    for p in [pr/'verification.json',pr/'independent-canonical-verification.json',Path(__file__),
              u.ROOT/'scripts/verify_upstream_nn_pool6_assembly.py']:bind(p)
    return records,new,known_pool,sources

def fit():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing frozen assembly preserved')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(created=stamp(),declaration_sha256=u.sha(DECL),
        primary='own7-known-calibration',controls=['equal-six-nn','fixed-tree25-pool75'],
        rule='Fixed equal six NN full outputs; fixed25%tree154/75%pool. One own7 nonnegative sum-one known-OOF least-squares fit, uniform start,ftol1e-17,maxiter1000. No recovered outcomes in fitting/stopping/selection. Freeze all three before any audit; all127 simplex supports independently checked before recovered labels.',
        known_calibration_is_nested_cv=False,qualifying_labels_read=False,
        input_realization='Full refits use verified paired production B; development and calibration OOF vectors belong to recorded original A recipes. No matched B-trained CV claim.',
        script_sha256=u.sha(Path(__file__)),verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_nn_pool6_assembly.py')))
    rows,new,kpool,sources=checked_sources()
    known=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in rows]
    known.append(np.fromfile(kpool,dtype='<f4'))
    y=np.load(own6.LABELS).astype(np.float64)
    if any(a.shape!=(1128164,) or not np.isfinite(a).all() for a in known):raise ValueError('Known vectors malformed')
    x=np.column_stack(known).astype(np.float64);d=x-x[:,-1,None];residual=y-x[:,-1]
    gram=d.T@d/len(y);rhs=d.T@residual/len(y);size=7
    opt=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(size,1/size),jac=lambda w:2*(gram@w-rhs),
        bounds=[(0.,1.)]*size,constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(size))],
        method='SLSQP',options=dict(ftol=1e-17,maxiter=1000))
    if not opt.success or opt.x.min()<-1e-12 or abs(opt.x.sum()-1)>1e-10:raise ValueError('Known-only convex fit failed')
    weights=np.maximum(opt.x,0);weights/=weights.sum()
    own6.combined(known,weights).tofile(RUN/'known-fit.f32')
    u.write(REPORT/'calibration.json',dict(weights=weights.tolist(),family_ids=[r['id'] for r in rows]+['equal-six-nn'],
        families=rows,new_neural_full_sources=new,known_pool=str(kpool.relative_to(u.ROOT)),
        gram=gram.tolist(),rhs=rhs.tolist(),optimizer_success=True,source_sha256=sources,
        known_score_is_nested_cross_validation=False,qualifying_labels_read=False))
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in rows]
    neural=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in new]
    if any(a.shape!=(2817131,) or not np.isfinite(a).all() for a in full+neural):raise ValueError('Full qualifying vectors malformed')
    pooled=(sum(a.astype(np.float64) for a in neural)/6).astype('<f4')
    tree=full[next(i for i,r in enumerate(rows) if r['id']=='tree154')]
    fixed=(tree.astype(np.float64)*.25+pooled.astype(np.float64)*.75).astype('<f4')
    calibrated=own6.combined([*full,pooled],weights)
    for name,p in [('equal-six-nn',pooled),('fixed-tree25-pool75',fixed),('own7-known-calibration',calibrated)]:p.tofile(RUN/(name+'.f32'))
    own6.verify_bindings(sources)
    u.write(REPORT/'frozen.json',dict(state='complete_three_predeclared_own_candidates_frozen_before_audit',observed=stamp(),
        plan_sha256=u.sha(REPORT/'plan.json'),calibration_sha256=u.sha(REPORT/'calibration.json'),
        predictions={p.stem:u.sha(p) for p in RUN.glob('*.f32')},qualifying_labels_read=False,
        known_calibration_is_nested_cv=False,independent_coefficient_and_arithmetic_verification_remaining=True))
    print('Three own candidates frozen; independent simplex check required before audit',flush=True)

if __name__=='__main__':fit()
