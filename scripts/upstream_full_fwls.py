#!/usr/bin/env python3
"""Full original own bank, fixed published context interactions and lambda30."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.linalg.blas import dsyrk
import upstream_full_fwls_inputs as data

u,checks=data.u,data.checks
REPORT=u.ROOT/'experiments/upstream-full-fwls'
RUN=u.ROOT/'artifacts/upstream-full-fwls'
FIT_BLOCK=1024


def matrix_sha(matrix):
    if not matrix.flags.f_contiguous:raise ValueError('Original column-major system required')
    return hashlib.sha256(memoryview(matrix.T).cast('B')).hexdigest()


def sources():
    mp=data.REPORT/'manifest.json';vp=data.REPORT/'independent-verification.json'
    m=u.read(mp);v=u.read(vp)
    if (v['state']!='complete_independent_all413_original_input_streams_and_exact_complete_paired_aliases' or
        v['manifest_sha256']!=u.sha(mp) or v['plan_sha256']!=u.sha(data.REPORT/'plan.json') or
        not v['all_known_and_qualifying_operator_bits_equal'] or v['all_original_entries']!=413 or
        v['qualifying_labels_read'] is not False or v['models_fitted']!=0 or
        v['source_sha256']!=m['source_sha256'] or v['distinct_signals']!=m['distinct_signals'] or
        v['interaction_dimension']!=m['interaction_dimension']):raise ValueError('Complete independent full bank and all aliases required')
    sources=dict(v['source_sha256'])
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_full_fwls.py',data.DECL,mp,vp,data.REPORT/'plan.json']:
        key=str(path.relative_to(u.ROOT));digest=u.sha(path)
        if key in sources and sources[key]!=digest:raise ValueError('Original full bank source expectation conflicts')
        sources[key]=digest
    checks.verify_bindings(sources)
    return m,sources


def contexts(names,query):
    result=np.column_stack([np.load(u.FRESH/('preds_new/'+name+'.'+query+'.npy')) for name in names]).astype(np.float64)
    if result.shape!=(data.COUNTS[query],14) or not np.isfinite(result).all():raise ValueError('Every original complete public gate required')
    return result


def interactions(x,f):
    result=np.ones((len(x),x.shape[1]*14+1),dtype=np.float64)
    for i in range(x.shape[1]):result[:,i*14:(i+1)*14]=x[:,i,None]*f
    return result


def prediction(x,f,weights):
    result=np.full(len(x),float(weights[-1]),dtype=np.float64)
    for i in range(x.shape[1]):
        gate=np.zeros(len(x))
        for j in range(14):gate+=float(weights[i*14+j])*f[:,j]
        result+=x[:,i]*gate
    return np.clip(result,1.05,4.95)


def run():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing full-model fit or partial evidence retained')
    m,sources_=sources();data.context.check_disk();REPORT.mkdir();RUN.mkdir()
    dimension=m['interaction_dimension'];indices=m['selected_indices']
    if len(indices)*14+1!=dimension or len(m['gates'])!=14:raise ValueError('Full original interaction dimension required')
    u.write(REPORT/'plan.json',dict(created=checks.stamp(),source_sha256=sources_,input_manifest_sha256=u.sha(data.REPORT/'manifest.json'),
        input_verification_sha256=u.sha(data.REPORT/'independent-verification.json'),dimension=dimension,
        signals=len(indices),selected_indices=indices,gates=m['gates'],lambda_=30.,fit_block=FIT_BLOCK,
        original_folds=5,unpenalized_intercept=True,worker_sha256=u.sha(Path(__file__)),
        verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_full_fwls.py'),qualifying_labels_read=False,
        rule='Complete five held-user models; all distinct paired own-bank signals,14 original gates,lambda30,unpenalized bias. Uniform clipped fivefold qualifying bag. Conditional metaCV, not fully nested end-to-end or official Test.'))
    labels=np.load(checks.LABELS).astype(np.float64);groups=np.load(u.ROOT/'artifacts/upstream-item154-tree/groups.npy')
    if labels.shape!=(1128164,) or groups.shape!=labels.shape or set(np.unique(groups))!=set(range(10)):
        raise ValueError('Complete canonical labels and user groups required')
    folds=groups//2;grams=[np.zeros((dimension,dimension),order='F') for _ in range(5)];rhs=np.zeros((5,dimension))
    reader=data.Inputs('probex',m['entries']);f=contexts(m['gates'],'probex');operator=hashlib.sha256()
    try:
        for lo in range(0,len(labels),FIT_BLOCK):
            hi=min(lo+FIT_BLOCK,len(labels));x=reader.block(lo,hi)[:,indices];z=interactions(x,f[lo:hi]);operator.update(z.tobytes())
            for fold in range(5):
                mask=folds[lo:hi]==fold;block=z[mask]
                grams[fold]=dsyrk(1.,np.asfortranarray(block.T),beta=1.,c=grams[fold],lower=1,overwrite_c=1)
                rhs[fold]+=np.einsum('ij,i->j',block,labels[lo:hi][mask],optimize=False)
            data.context.check_disk()
            if lo%(FIT_BLOCK*32)==0:
                u.write(REPORT/'status.json',dict(state='complete_feature_group_statistics',rows=hi,total_rows=len(labels)))
    finally:reader.close()
    statistics=[dict(fold=i,lower_column_major_sha256=matrix_sha(g),rhs_sha256=hashlib.sha256(rhs[i].tobytes()).hexdigest()) for i,g in enumerate(grams)]
    weights=[]
    for fold in range(5):
        system=np.zeros((dimension,dimension),order='F');target=np.zeros(dimension)
        for other in range(5):
            if other!=fold:system+=grams[other];target+=rhs[other]
        diagonal=np.diag_indices(dimension);system[diagonal]+=30.;system[-1,-1]-=30.
        w=cho_solve(cho_factor(system,lower=True,overwrite_a=True),target)
        if not np.isfinite(w).all():raise ValueError('Finite complete original full model required')
        np.savez(RUN/('fold-%s.npz'%fold),weights=w,eligible_folds=np.array([i for i in range(5) if i!=fold]),lambda_=30.)
        weights.append(w);print('Complete full FWLS model',fold,'of5; signals',len(indices),flush=True)
    del grams,system
    known=np.empty(len(labels),dtype='<f4');reader=data.Inputs('probex',m['entries'])
    try:
        for lo in range(0,len(labels),8192):
            hi=min(lo+8192,len(labels));x=reader.block(lo,hi)[:,indices]
            for fold,w in enumerate(weights):
                mask=folds[lo:hi]==fold
                known[lo:hi][mask]=prediction(x[mask],f[lo:hi][mask],w).astype('<f4')
    finally:reader.close()
    known.tofile(RUN/'known.f32');del f
    reader=data.Inputs('qual',m['entries']);f=contexts(m['gates'],'qual')
    output=np.memmap(RUN/'qualifying.f32',mode='w+',dtype='<f4',shape=(2817131,))
    try:
        for lo in range(0,2817131,8192):
            hi=min(lo+8192,2817131);x=reader.block(lo,hi)[:,indices];values=np.zeros(hi-lo)
            for w in weights:values+=prediction(x,f[lo:hi],w)
            output[lo:hi]=(values/5).astype('<f4');data.context.check_disk()
            u.write(REPORT/'status.json',dict(state='complete_five_model_qualifying_bag',rows=hi,total_rows=2817131))
    finally:reader.close()
    output.flush();del output
    checks.verify_bindings(sources_)
    u.write(REPORT/'frozen.json',dict(state='complete_original_full_own_bank_fwls_all_five_models_and_uniform_bag_before_audit',
        observed=checks.stamp(),plan_sha256=u.sha(REPORT/'plan.json'),source_sha256=sources_,statistics=statistics,
        known_interaction_c_order_sha256=operator.hexdigest(),outputs={p.name:u.sha(p) for p in RUN.iterdir()},
        qualifying_labels_read=False,independent_actual_model_replay_remaining=True))


if __name__=='__main__':run()
