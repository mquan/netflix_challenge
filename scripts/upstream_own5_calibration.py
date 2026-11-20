#!/usr/bin/env python3
"""One known-only convex fit of five fully replayed own production families."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['VECLIB_MAXIMUM_THREADS']='2'
from pathlib import Path
import datetime as dt
import math
import numpy as np
from scipy.optimize import minimize
import upstream_repro as u
RUN=u.ROOT/'artifacts/upstream-own5-calibration'
REPORT=u.ROOT/'experiments/upstream-own5-calibration'
FAMILIES=[('tree50','tree','upstream-rbm50-tree','upstream-rbm50-tree-latent90-production'),
 ('nn50','nn','upstream-rbm50-context','upstream-rbm50-production'),
 ('nn90','nn','upstream-latent90-mps-context','upstream-rbm50-tree-latent90-production'),
 ('nn109','nn','upstream-autorec109-mps-context','upstream-autorec-consensus-stream-production'),
 ('nn154','nn','upstream-item154-mps-context','upstream-item154-stream-production')]
LABELS=u.FRESH/'data/probex/ratings.npy'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def verify_bindings(bindings):
    cache={};signatures={}
    for name,digest in bindings.items():
        p=u.ROOT/name;s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in cache:cache[key]=u.sha(p)
        if cache[key]!=digest:raise ValueError('Own model/source binding changed: '+name)
        signatures[name]=key
    return signatures

def checked_sources():
    sources={};records=[];identities=None
    def bind(p,digest=None):
        key=str(p.relative_to(u.ROOT));actual=u.sha(p) if digest is None else digest
        if key in sources and sources[key]!=actual:raise ValueError('Conflicting own lineage: '+key)
        sources[key]=actual
    canonical={'labels':np.load(LABELS),'users':np.load(u.FRESH/'data/probex/user_idxs.npy')}
    for ident,role,dev,full in FAMILIES:
        dr,da=u.ROOT/'experiments'/dev,u.ROOT/'artifacts'/dev
        pr,pa=u.ROOT/'experiments'/full,u.ROOT/'artifacts'/full
        m=u.read(dr/'manifest.json');e=u.read(dr/'evaluation.json');proof=u.read(dr/'independent-verification.json');replay=u.read(dr/(role+'-replay.json'))
        if proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(dr/'evaluation.json') or not proof['fixed_average_bitwise_equal']:
            raise ValueError('Original independent own development proof required')
        if [r['fold'] for r in replay['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in replay['records']):raise ValueError('All actual development model replays required')
        ids={k:np.load(da/(k+'.npy')) for k in ['labels','users','groups']}
        if any(not np.array_equal(ids[k],canonical[k]) for k in canonical):raise ValueError('Canonical known rows differ')
        if identities is None:identities=ids
        elif any(not np.array_equal(ids[k],identities[k]) for k in ids):raise ValueError('Family folds/row identities differ')
        for collection in [m['code_sha256'],m['source_manifests_sha256']]:
            for name,digest in collection.items():bind(u.ROOT/name,digest)
        for name,digest in m['prepared_sha256'].items():bind(da/name,digest)
        budgets=[]
        for fold in range(5):
            rp=dr/role/f'{fold}.json';r=u.read(rp)
            if r['manifest_sha256']!=u.sha(dr/'manifest.json') or r['family']!=role or r['fold']!=fold:raise ValueError('Actual originating fold differs')
            budgets.append(r['chosen_iterations']);bind(rp)
            for name,digest in r['outputs'].items():bind(da/role/str(fold)/name,digest)
        row=next(r for r in e['records'] if r['arm']==role);checked=next(r for r in proof['records'] if r['arm']==role)
        pred=da/(role+'.f32')
        if row['prediction_sha256']!=checked['prediction_sha256']:raise ValueError('Actual own OOF vector differs')
        bind(pred,row['prediction_sha256'])
        p=u.read(pr/'plan.json');receipt=u.read(pr/(role+'.json'));actual=u.read(pr/(role+'-replay.json'))
        pd=u.sha(pr/'plan.json')
        if (p['qualifying_labels_read'] is not False or receipt['plan_sha256']!=pd or actual['plan_sha256']!=pd or
            not actual['bitwise_equal'] or actual['family_receipt_sha256']!=u.sha(pr/(role+'.json'))):raise ValueError('Actual own full-family replay required')
        expected=[int(np.median(np.asarray(budgets)[:,i])) for i in range(3)]
        if p['iterations'][role]!=expected or p['seeds']!=m['config']['seeds']:raise ValueError('Original known-selected full budgets differ')
        keys=list(m['config']) if role=='nn' else ['seeds','threads','ridge_lambda']+[k for k in m['config'] if k.startswith('tree_')]
        if any(p['config'][k]!=m['config'][k] for k in keys):raise ValueError('Original full model recipe differs')
        for collection in [p['source_sha256'],p['code_sha256']]:
            for name,digest in collection.items():
                if 'recovered-audit/' in name or 'qual_ratings' in name:continue
                bind(u.ROOT/name,digest)
        for name,digest in receipt['outputs'].items():bind(pa/role/name,digest)
        qual=pa/role/'qualifying.f32'
        if actual['prediction_sha256']!=receipt['outputs']['qualifying.f32']:raise ValueError('Actual full query vector differs')
        for path in [dr/'manifest.json',dr/'evaluation.json',dr/'independent-verification.json',dr/(role+'-replay.json'),pr/'plan.json',pr/(role+'.json'),pr/(role+'-replay.json')]:bind(path)
        records.append(dict(id=ident,family=role,development=dev,production=full,known_prediction=str(pred.relative_to(u.ROOT)),qualifying_prediction=str(qual.relative_to(u.ROOT)),original_full_iterations=expected))
    for path in [LABELS,u.FRESH/'data/probex/user_idxs.npy',Path(__file__),u.ROOT/'scripts/verify_upstream_own5_calibration.py']:bind(path)
    signatures=verify_bindings(sources)
    return records,sources,signatures

def combined(arrays,weights):
    values=np.zeros(len(arrays[0]),dtype=np.float64)
    for a,w in zip(arrays,weights):values=np.add(values,np.multiply(a.astype(np.float64),float(w)))
    return values.astype('<f4')

def fit():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing five-family evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=stamp(),families=FAMILIES,
        eligibility='All four fully produced neural branches of the current own107 lineage (50/90/item154 context and own109 raw extension), plus its actually replayed full tree50. Fixed complete list, not chosen using recovered group scores.',
        rule='One nonnegative sum-one least-squares calibration on complete own known OOF vectors. SLSQP convex quadratic fit with analytic Jacobian, uniform initialization,ftol1e-14,maxiter1000. No coefficient/family/hyperparameter search using qualifying outcomes. Sequential float64 multiply/add in declared family order, then one float32 store. Freeze before recovered audit.',
        known_score_is_nested_cross_validation=False,qualifying_labels_used_for_fitting_or_selection=False,qualifying_label_payload_read=False,
        source_families_fitted_again=False,script_sha256=u.sha(Path(__file__))))
    records,sources,signatures=checked_sources();y=np.load(LABELS).astype(np.float64)
    arrays=[np.fromfile(u.ROOT/r['known_prediction'],dtype='<f4') for r in records]
    if any(a.shape!=(1128164,) or not np.isfinite(a).all() for a in arrays):raise ValueError('Complete finite own known vectors required')
    x=np.column_stack(arrays).astype(np.float64);d=x-x[:,-1,None];residual=y-x[:,-1]
    gram=d.T@d/len(y);rhs=d.T@residual/len(y)
    opt=minimize(lambda w:float(w@gram@w-2*rhs@w),np.full(5,.2),jac=lambda w:2*(gram@w-rhs),
        bounds=[(0.,1.)]*5,constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1),jac=lambda w:np.ones(5))],
        method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
    if not opt.success or abs(opt.x.sum()-1)>1e-10 or opt.x.min()<-1e-12:raise ValueError('Convex calibration did not converge')
    weights=np.maximum(opt.x,0.);weights/=weights.sum();gradient=gram@weights-rhs;active=weights>1e-7;lagrange=float(np.mean(gradient[active]))
    if np.max(np.abs(gradient[active]-lagrange))>1e-7 or np.any(gradient[~active]<lagrange-1e-7):raise ValueError('Convex stationarity failed')
    RUN.mkdir();known=combined(arrays,weights);known.tofile(RUN/'known-fit.f32')
    score=math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(known,y))/len(y))
    u.write(REPORT/'calibration.json',dict(state='complete_own_five_family_known_calibration',observed=stamp(),families=records,weights=weights.tolist(),known_fit_rmse=score,
        gram=gram.tolist(),rhs=rhs.tolist(),optimizer_success=True,source_sha256=sources,source_signature_before=signatures,
        known_prediction_sha256=u.sha(RUN/'known-fit.f32'),known_score_is_nested_cross_validation=False,qualifying_label_payload_read=False,qualifying_labels_used_for_fitting_or_selection=False))
    full=[np.fromfile(u.ROOT/r['qualifying_prediction'],dtype='<f4') for r in records]
    if any(a.shape!=(2817131,) or not np.isfinite(a).all() for a in full):raise ValueError('Complete finite own full-query vectors required')
    combined(full,weights).tofile(RUN/'qualifying.f32')
    for name,before in signatures.items():
        s=(u.ROOT/name).stat()
        if tuple(before)!=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns):raise ValueError('Verified source changed during freezing')
    u.write(REPORT/'frozen.json',dict(state='complete_own_five_family_prediction_frozen_before_audit',observed=stamp(),plan_sha256=u.sha(REPORT/'plan.json'),
        calibration_sha256=u.sha(REPORT/'calibration.json'),known_prediction_sha256=u.sha(RUN/'known-fit.f32'),prediction_sha256=u.sha(RUN/'qualifying.f32'),
        weights=weights.tolist(),qualifying_labels_read=False,qualifying_labels_used_for_fitting_or_selection=False,known_score_is_nested_cross_validation=False))
    print(dict(weights=dict(zip([r['id'] for r in records],weights.tolist())),known_fit_rmse=score,qualifying_frozen=True),flush=True)
if __name__=='__main__':fit()
