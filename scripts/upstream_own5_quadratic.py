#!/usr/bin/env python3
"""Fixed quadratic residual calibration of the five fully replayed own families."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['VECLIB_MAXIMUM_THREADS']='2'
from pathlib import Path
import numpy as np
import upstream_own5_calibration as parent
u=parent.u
RUN=u.ROOT/'artifacts/upstream-own5-quadratic'
REPORT=u.ROOT/'experiments/upstream-own5-quadratic'
CHUNK=16384
PENALTY=1000.
TERMS=[(i,j) for i in range(5) for j in range(i,5)]

def raw(arrays,lo,hi):
    a=[np.asarray(x[lo:hi],dtype=np.float64) for x in arrays]
    q=[a[i]-a[4] for i in range(4)]+[a[4]-3.6]
    return np.column_stack(q+[q[i]*q[j] for i,j in TERMS])

def design(values,center,scale):
    return np.column_stack([np.ones(len(values)),(values-center)/scale])

def predict(arrays,base,center,scale,beta):
    out=np.empty(len(base),dtype='<f4')
    for lo in range(0,len(base),CHUNK):
        hi=min(lo+CHUNK,len(base));z=design(raw(arrays,lo,hi),center,scale)
        delta=np.zeros(hi-lo,dtype=np.float64)
        for j in range(21):delta=delta+z[:,j]*float(beta[j])
        out[lo:hi]=np.clip(base[lo:hi].astype(np.float64)+delta,1.05,4.95).astype('<f4')
    return out

def run():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing quadratic evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=parent.stamp(),families=parent.FAMILIES,penalty=PENALTY,
        rule='One fixed degree-two residual ridge,21 coefficients including an unpenalized intercept. Five coordinates: first four family predictions minus NN154, then NN154 minus3.6. Linear coordinates followed by upper-triangle products in lexicographic order. Center and population-scale20 columns on complete known rows. Fixed ridge1000 on20 standardized coefficients, no tuning. Offset is the preceding frozen known-only convex five-family prediction. Clip1.05/4.95 and storefloat32. Freeze before recovered-label access.',
        known_score_is_nested_cross_validation=False,qualifying_labels_used_for_fitting_or_selection=False,
        code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),u.ROOT/'scripts/verify_upstream_own5_quadratic.py']}))
    cal=u.read(parent.REPORT/'calibration.json');freeze=u.read(parent.REPORT/'frozen.json')
    sources=dict(cal['source_sha256'])
    for p in [parent.REPORT/'calibration.json',parent.REPORT/'frozen.json',parent.RUN/'known-fit.f32',parent.RUN/'qualifying.f32']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    if (freeze['calibration_sha256']!=u.sha(parent.REPORT/'calibration.json') or
        freeze['qualifying_labels_read'] is not False or freeze['weights']!=cal['weights'] or
        freeze['prediction_sha256']!=u.sha(parent.RUN/'qualifying.f32') or
        cal['known_prediction_sha256']!=u.sha(parent.RUN/'known-fit.f32')):raise ValueError('Original known-only offset freeze required')
    signatures=parent.verify_bindings(sources)
    rows=cal['families'];known=[np.memmap(u.ROOT/r['known_prediction'],dtype='<f4',mode='r') for r in rows]
    base=np.memmap(parent.RUN/'known-fit.f32',dtype='<f4',mode='r');y=np.load(parent.LABELS).astype(np.float64)
    if len(base)!=1128164 or any(len(a)!=len(base) for a in known):raise ValueError('Complete known rows required')
    sums=np.zeros(20);squares=np.zeros(20)
    for lo in range(0,len(y),CHUNK):
        z=raw(known,lo,min(lo+CHUNK,len(y)));sums+=z.sum(axis=0);squares+=(z*z).sum(axis=0)
    center=sums/len(y);scale=np.sqrt(np.maximum(squares/len(y)-center*center,0))
    if not np.isfinite(scale).all() or np.min(scale)<1e-12:raise ValueError('Nondegenerate finite fixed basis required')
    gram=np.zeros((21,21));rhs=np.zeros(21)
    for lo in range(0,len(y),CHUNK):
        hi=min(lo+CHUNK,len(y));z=design(raw(known,lo,hi),center,scale)
        gram+=z.T@z;rhs+=z.T@(y[lo:hi]-base[lo:hi].astype(np.float64))
    penalty=np.diag([0.]+[PENALTY]*20);beta=np.linalg.solve(gram+penalty,rhs)
    RUN.mkdir();kp=predict(known,base,center,scale,beta);kp.tofile(RUN/'known-fit.f32')
    score=float(np.sqrt(np.mean((kp.astype(np.float64)-y)**2)))
    u.write(REPORT/'calibration.json',dict(state='complete_fixed_known_only_quadratic_fit',observed=parent.stamp(),
        plan_sha256=u.sha(REPORT/'plan.json'),families=rows,center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),
        gram=gram.tolist(),rhs=rhs.tolist(),penalty=PENALTY,source_sha256=sources,known_fit_rmse=score,
        known_score_is_nested_cross_validation=False,qualifying_labels_used_for_fitting_or_selection=False,
        known_prediction_sha256=u.sha(RUN/'known-fit.f32')))
    full=[np.memmap(u.ROOT/r['qualifying_prediction'],dtype='<f4',mode='r') for r in rows]
    offset=np.memmap(parent.RUN/'qualifying.f32',dtype='<f4',mode='r')
    if len(offset)!=2817131 or any(len(a)!=len(offset) for a in full):raise ValueError('Complete full rows required')
    predict(full,offset,center,scale,beta).tofile(RUN/'qualifying.f32')
    for name,key in signatures.items():
        s=(u.ROOT/name).stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=key:raise ValueError('Source changed while fitting')
    u.write(REPORT/'frozen.json',dict(state='complete_quadratic_vector_frozen_before_audit',observed=parent.stamp(),
        plan_sha256=u.sha(REPORT/'plan.json'),calibration_sha256=u.sha(REPORT/'calibration.json'),
        known_prediction_sha256=u.sha(RUN/'known-fit.f32'),prediction_sha256=u.sha(RUN/'qualifying.f32'),
        qualifying_labels_read=False,qualifying_labels_used_for_fitting_or_selection=False,known_score_is_nested_cross_validation=False))
    print(dict(known_fit_rmse=score,qualifying_vector_frozen=True),flush=True)

if __name__=='__main__':run()
