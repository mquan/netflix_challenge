#!/usr/bin/env python3
"""Released FWLS interactions on13 freshly trained OOF families, known only."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import hashlib
import shutil
import subprocess
import tomli
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import upstream_complete13_calibration as prior

u,own6=prior.u,prior.own6
REPORT=u.ROOT/'experiments/upstream-own13-fwls'
RUN=u.ROOT/'artifacts/upstream-own13-fwls'
DECL=u.ROOT/'OWN13_FEATURE_WEIGHTED_STACKING_PLAN.md'
SIZE,GATES,DIMENSION=13,14,183
LAMBDA=30.


def check_disk():
    if shutil.disk_usage(u.ROOT).free<18.75*2**30:raise ValueError('Original disk reserve must remain available')


def checked_sources():
    report=prior.REPORT;proof=u.read(report/'independent-preaudit-verification.json')
    frozen=u.read(report/'frozen.json');cal=u.read(report/'calibration.json')
    if (proof['state']!='complete_independent8191_support_and_ridge10_all_frozen_arithmetic_before_audit' or
        proof['simplex_supports_examined']!=8191 or proof['ridge_equations']!=14 or
        proof['qualifying_labels_read'] is not False or not proof['all_stored_candidate_arithmetic_bitwise'] or
        proof['freeze_sha256']!=u.sha(report/'frozen.json') or
        proof['calibration_sha256']!=u.sha(report/'calibration.json') or
        frozen['plan_sha256']!=u.sha(report/'plan.json') or
        frozen['calibration_sha256']!=u.sha(report/'calibration.json') or
        proof['source_sha256']!=cal['source_sha256']):raise ValueError('Complete13 original component eligibility and preaudit proof required')
    sources=dict(proof['source_sha256'])
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));digest=u.sha(path) if expected is None else expected
        if key in sources and sources[key]!=digest:raise ValueError('Original own source expectation conflicts: '+key)
        sources[key]=digest
    receipt_path=u.BASE/'fresh-receipts/vfeat1.json'
    if str(receipt_path.relative_to(u.ROOT)) not in sources:raise ValueError('Original fresh context receipt must already be source-bound')
    if u.sha(receipt_path)!=sources[str(receipt_path.relative_to(u.ROOT))]:raise ValueError('Original fresh receipt changed')
    receipt=u.read(receipt_path)
    if receipt['source_commit']!=u.PIN or receipt['job']!='vfeat1':raise ValueError('Original fresh context generation required')
    for relative in ['voting-3.toml','pipeline-3.toml','src/bin/fwls.rs','src/blend.rs']:
        original=subprocess.check_output(['git','-C',str(u.FRESH),'show',u.PIN+':'+relative])
        bind(u.FRESH/relative,hashlib.sha256(original).hexdigest())
        if u.sha(u.FRESH/relative)!=hashlib.sha256(original).hexdigest():raise ValueError('Pinned original FWLS source/configuration changed')
    config=u.FRESH/'voting-3.toml'
    native=tomli.loads((u.FRESH/'pipeline-3.toml').read_text())['jobs']['fwls15k4']
    if native['voting']!=['fwls15'] or native['extra_args'][:2]!=['--lambda','30']:
        raise ValueError('Original declared fwls15 lambda30 required')
    names=list(dict.fromkeys(tomli.loads(config.read_text())['fwls15']))
    if len(names)!=GATES or names[0]!='vf/vf000_constant':raise ValueError('Original first-occurrence14 gates required')
    for name in names:
        for query,count in [('probex',1128164),('qual',2817131)]:
            relative='preds_new/'+name+'.'+query+'.npy';path=u.FRESH/relative
            if relative not in receipt['outputs']:raise ValueError('Missing original complete context counterpart')
            bind(path,receipt['outputs'][relative])
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.shape!=(count,) or a.dtype!=np.dtype('<f4') or not np.isfinite(a).all():raise ValueError('Complete original finite float32 contexts required')
            if name=='vf/vf000_constant' and not np.all(a==1):raise ValueError('Original constant gate differs')
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_own13_fwls.py',DECL,config,
                 u.FRESH/'src/bin/fwls.rs',u.FRESH/'src/blend.rs',report/'plan.json',report/'frozen.json',
                 report/'calibration.json',report/'independent-preaudit-verification.json']:
        bind(path)
    if len(cal['families'])!=SIZE or len(set(r['id'] for r in cal['families']))!=SIZE:raise ValueError('Complete13 fixed list required')
    own6.verify_bindings(sources)
    return cal['families'],names,sources


def inputs(rows,names,query):
    count=1128164 if query=='probex' else 2817131
    key='known_prediction' if query=='probex' else 'qualifying_prediction'
    predictions=[np.memmap(u.ROOT/r[key],mode='r',dtype='<f4',shape=(count,)) for r in rows]
    contexts=[np.load(u.FRESH/('preds_new/'+name+'.'+query+'.npy'),mmap_mode='r') for name in names]
    return predictions,contexts


def design(predictions,contexts,lo,hi):
    x=np.column_stack([a[lo:hi] for a in predictions]).astype(np.float64)
    f=np.column_stack([a[lo:hi] for a in contexts]).astype(np.float64)
    result=np.empty((hi-lo,DIMENSION),dtype=np.float64)
    result[:,:-1]=(x[:,:,None]*f[:,None,:]).reshape(hi-lo,-1);result[:,-1]=1.
    if not np.isfinite(result).all():raise ValueError('Finite complete interaction operators required')
    return result


def predict(predictions,contexts,weights,lo,hi):
    values=np.full(hi-lo,float(weights[-1]),dtype=np.float64)
    for i,a in enumerate(predictions):
        gate=np.zeros(hi-lo,dtype=np.float64)
        for j,f in enumerate(contexts):gate+=float(weights[i*GATES+j])*f[lo:hi].astype(np.float64)
        values+=a[lo:hi].astype(np.float64)*gate
    return np.clip(values,1.05,4.95)


def run():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing complete or partial FWLS evidence retained')
    rows,names,sources=checked_sources();check_disk();REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(created=own6.stamp(),source_sha256=sources,families=rows,context_names=names,
        declaration_sha256=u.sha(DECL),script_sha256=u.sha(Path(__file__)),
        verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_own13_fwls.py'),
        lambda_=LAMBDA,regularized_coefficients=182,unpenalized_intercept=True,
        outer_user_folds=5,groups=10,dimension=DIMENSION,qualifying_labels_read=False,
        score_scope='Cross-fitted meta-model conditional on original OOF features; not fully nested end-to-end CV or official Test.',
        production='Uniform original five-fold model bag; each clipped in float64, ordered sum, divide once, float32 store.'))
    labels=np.load(own6.LABELS).astype(np.float64)
    groups=np.load(u.ROOT/'artifacts/upstream-item154-tree/groups.npy')
    if labels.shape!=(1128164,) or groups.shape!=labels.shape or set(np.unique(groups))!=set(range(10)):
        raise ValueError('All original complete known labels and user groups required')
    arrays,contexts=inputs(rows,names,'probex');gram=np.zeros((10,DIMENSION,DIMENSION));rhs=np.zeros((10,DIMENSION))
    digest=hashlib.sha256()
    for lo in range(0,len(labels),8192):
        hi=min(lo+8192,len(labels));z=design(arrays,contexts,lo,hi);digest.update(z.tobytes(order='C'))
        for group in range(10):
            ids=groups[lo:hi]==group;block=z[ids]
            gram[group]+=block.T@block;rhs[group]+=block.T@labels[lo:hi][ids]
        check_disk()
    np.savez(RUN/'grouped-statistics.npz',gram=gram,rhs=rhs)
    penalty=np.eye(DIMENSION)*LAMBDA;penalty[-1,-1]=0
    weights=[];known=np.empty(len(labels),dtype='<f4');records=[]
    for fold in range(5):
        eligible=[g for g in range(10) if g//2!=fold]
        system=gram[eligible].sum(0)+penalty;target=rhs[eligible].sum(0)
        w=cho_solve(cho_factor(system,lower=True,check_finite=True),target);weights.append(w)
        np.savez(RUN/('fold-%s.npz'%fold),weights=w,eligible_groups=np.array(eligible),lambda_=LAMBDA)
        for lo in range(0,len(labels),8192):
            hi=min(lo+8192,len(labels));mask=groups[lo:hi]//2==fold
            known[lo:hi][mask]=predict(arrays,contexts,w,lo,hi)[mask].astype('<f4')
        records.append(dict(fold=fold,fit_rows=int(np.isin(groups,eligible).sum()),held_rows=int((groups//2==fold).sum())))
    known.tofile(RUN/'known.f32')
    del arrays,contexts
    arrays,contexts=inputs(rows,names,'qual');count=2817131
    output=np.memmap(RUN/'qualifying.f32',mode='w+',dtype='<f4',shape=(count,))
    for lo in range(0,count,8192):
        hi=min(lo+8192,count);values=np.zeros(hi-lo,dtype=np.float64)
        for w in weights:values+=predict(arrays,contexts,w,lo,hi)
        output[lo:hi]=(values/5).astype('<f4');check_disk()
    output.flush();del output
    own6.verify_bindings(sources)
    u.write(REPORT/'frozen.json',dict(state='complete_own13_fwls_all_five_userfold_models_and_uniform_qualifying_bag_before_audit',
        observed=own6.stamp(),plan_sha256=u.sha(REPORT/'plan.json'),source_sha256=sources,
        known_interaction_c_order_sha256=digest.hexdigest(),records=records,
        outputs={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,independent_replay_remaining=True))
    print('Complete all five FWLS models and known/Q vectors; independent replay remains',flush=True)


if __name__=='__main__':run()
