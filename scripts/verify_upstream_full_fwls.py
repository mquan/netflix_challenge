#!/usr/bin/env python3
"""Rebuild complete full-bank systems and all actual saved model predictions."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.linalg.blas import dsyrk
import upstream_full_fwls as experiment
import verify_upstream_full_fwls_inputs as operators

u=experiment.u
STATE='complete_independent_full_own_bank_interactions_all_systems_actual_five_models_and_all_predictions'


def gates(names,query):
    count=experiment.data.COUNTS[query];result=np.empty((count,14),dtype=np.float64)
    for column,name in enumerate(names):result[:,column]=np.load(u.FRESH/('preds_new/'+name+'.'+query+'.npy'))
    return result


def prediction(x,f,weights):
    result=np.repeat(np.float64(weights[-1]),len(x))
    for column in range(x.shape[1]):
        weighted=np.zeros(len(x))
        for gate in range(14):weighted=weighted+f[:,gate]*float(weights[column*14+gate])
        result=result+weighted*x[:,column]
    return np.clip(result,1.05,4.95)


def verify():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-preaudit-verification.json'
    if output.exists():raise ValueError('Existing complete full-model proof retained')
    plan=u.read(report/'plan.json');frozen=u.read(report/'frozen.json')
    if (frozen['state']!='complete_original_full_own_bank_fwls_all_five_models_and_uniform_bag_before_audit' or
        frozen['plan_sha256']!=u.sha(report/'plan.json') or frozen['qualifying_labels_read'] is not False or
        plan['worker_sha256']!=u.sha(Path(experiment.__file__)) or plan['verifier_sha256']!=u.sha(Path(__file__)) or
        plan['lambda_']!=30. or plan['original_folds']!=5 or plan['fit_block']!=1024 or
        plan['input_manifest_sha256']!=u.sha(experiment.data.REPORT/'manifest.json') or
        plan['input_verification_sha256']!=u.sha(experiment.data.REPORT/'independent-verification.json')):
        raise ValueError('Complete original full-bank fitting/freeze declaration required')
    m,sources=experiment.sources();dimension=m['interaction_dimension'];indices=m['selected_indices']
    if (sources!=plan['source_sha256'] or sources!=frozen['source_sha256'] or indices!=plan['selected_indices'] or
        m['gates']!=plan['gates'] or dimension!=plan['dimension'] or len(indices)!=plan['signals']):raise ValueError('Original full fit source/recipe differs')
    for filename,digest in frozen['outputs'].items():
        if u.sha(root/filename)!=digest:raise ValueError('Actual frozen full-model artifact changed')
    labels=np.load(experiment.checks.LABELS).astype(np.float64)
    groups=np.load(u.ROOT/'artifacts/upstream-item154-tree/groups.npy');folds=groups//2
    users=np.load(u.ROOT/'artifacts/upstream-item154-tree/users.npy');order=np.argsort(users,kind='stable')
    if (labels.shape!=(1128164,) or groups.shape!=labels.shape or set(np.unique(groups))!=set(range(10)) or
        np.any((users[order][1:]==users[order][:-1]) & (folds[order][1:]!=folds[order][:-1]))):raise ValueError('All complete original held-user folds required')
    matrices=[np.zeros((dimension,dimension),order='F') for _ in range(5)];targets=np.zeros((5,dimension))
    reader=operators.parent('probex');f=gates(m['gates'],'probex');digest=hashlib.sha256()
    try:
        for lo in range(0,len(labels),1024):
            hi=min(lo+1024,len(labels));x=operators.block(reader,m['entries'],'probex',lo,hi)[:,indices]
            z=np.ones((hi-lo,dimension))
            for i in range(len(indices)):
                for j in range(14):z[:,i*14+j]=x[:,i]*f[lo:hi,j]
            digest.update(z.tobytes())
            for fold in range(5):
                mask=folds[lo:hi]==fold;block=z[mask]
                matrices[fold]=dsyrk(1.,np.asfortranarray(block.T),beta=1.,c=matrices[fold],lower=1,overwrite_c=1)
                targets[fold]+=np.einsum('ij,i->j',block,labels[lo:hi][mask],optimize=False)
            experiment.data.context.check_disk()
            if lo%(1024*32)==0:u.write(report/'replay-status.json',dict(state='independent_full_statistics',rows=hi,total_rows=len(labels)))
    finally:reader.close()
    if digest.hexdigest()!=frozen['known_interaction_c_order_sha256']:raise ValueError('Every complete known interaction bit must match')
    for i,matrix in enumerate(matrices):
        record=frozen['statistics'][i]
        actual=hashlib.sha256(memoryview(matrix.T).cast('B')).hexdigest()
        if (record['fold']!=i or record['lower_column_major_sha256']!=actual or
            record['rhs_sha256']!=hashlib.sha256(targets[i].tobytes()).hexdigest()):raise ValueError('Every original complete system bit must match')
    weights=[];records=[]
    for fold in range(5):
        actual=np.load(root/('fold-%s.npz'%fold));w=actual['weights']
        eligible=[i for i in range(5) if i!=fold];system=np.zeros((dimension,dimension),order='F');target=np.zeros(dimension)
        for other in eligible:system+=matrices[other];target+=targets[other]
        diagonal=np.diag_indices(dimension);system[diagonal]+=30.;system[-1,-1]-=30.
        independent=cho_solve(cho_factor(system,lower=True,overwrite_a=True),target)
        if (actual['lambda_']!=30. or actual['eligible_folds'].tolist()!=eligible or w.shape!=(dimension,) or
            not np.isfinite(w).all() or not np.array_equal(w.view('u8'),independent.view('u8'))):
            raise ValueError('Actual full eligible saved Cholesky coefficient bits must replay')
        weights.append(w);records.append(dict(fold=fold,all_eligible_system_bits_equal=True,actual_saved_model_reopened=True,
            every_independently_solved_coefficient_bit_equal=True,held_rows=int((folds==fold).sum())))
    del matrices,system
    known=np.fromfile(root/'known.f32',dtype='<f4')
    if known.shape!=(1128164,) or not np.isfinite(known).all():raise ValueError('Complete finite held prediction vector required')
    reader=operators.parent('probex')
    try:
        for lo in range(0,len(labels),6144):
            hi=min(lo+6144,len(labels));x=operators.block(reader,m['entries'],'probex',lo,hi)[:,indices]
            for fold,w in enumerate(weights):
                mask=folds[lo:hi]==fold;expected=prediction(x[mask],f[lo:hi][mask],w).astype('<f4')
                if not np.array_equal(expected.view('u4'),known[lo:hi][mask].view('u4')):raise ValueError('Every actual held prediction must replay')
    finally:reader.close()
    del f
    actual=np.fromfile(root/'qualifying.f32',dtype='<f4');f=gates(m['gates'],'qual');reader=operators.parent('qual')
    if actual.shape!=(2817131,) or not np.isfinite(actual).all():raise ValueError('Complete finite qualifying bag required')
    try:
        for lo in range(0,len(actual),6144):
            hi=min(lo+6144,len(actual));x=operators.block(reader,m['entries'],'qual',lo,hi)[:,indices];expected=np.zeros(hi-lo)
            for w in weights:expected=expected+prediction(x,f[lo:hi],w)
            expected=(expected/5).astype('<f4')
            if not np.array_equal(expected.view('u4'),actual[lo:hi].view('u4')):raise ValueError('Every qualifying model and uniform bag bit must replay')
    finally:reader.close()
    experiment.checks.verify_bindings(sources)
    score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(known.astype(np.float64),1.05,4.95),labels))/len(labels))
    u.write(output,dict(state=STATE,observed=experiment.checks.stamp(),plan_sha256=u.sha(report/'plan.json'),
        freeze_sha256=u.sha(report/'frozen.json'),verifier_sha256=u.sha(Path(__file__)),source_sha256=sources,
        all_original_413_signals_admitted=True,distinct_signals=len(indices),interaction_dimension=dimension,
        all_complete_interaction_and_system_bits_equal=True,records=records,
        all_actual_held_and_qualifying_prediction_bits_equal=True,uniform_fivefold_bag_bitwise_equal=True,
        known_meta_crossfit_rmse=score,fully_nested_end_to_end_cv=False,prediction_sha256=u.sha(root/'qualifying.f32'),
        qualifying_labels_read=False))
    print('Complete independently replayed full-bank FWLS; conditional metaCV',score,flush=True)


def audit():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-recovered-audit.json'
    if output.exists():raise ValueError('Existing complete full-model recovered audit retained')
    proof=u.read(report/'independent-preaudit-verification.json');frozen=u.read(report/'frozen.json')
    if (proof['state']!=STATE or proof['qualifying_labels_read'] is not False or
        proof['freeze_sha256']!=u.sha(report/'frozen.json') or proof['plan_sha256']!=u.sha(report/'plan.json') or
        proof['verifier_sha256']!=u.sha(Path(__file__)) or frozen['plan_sha256']!=u.sha(report/'plan.json') or
        not proof['all_complete_interaction_and_system_bits_equal'] or
        not proof['all_actual_held_and_qualifying_prediction_bits_equal'] or not proof['uniform_fivefold_bag_bitwise_equal'] or
        [r['fold'] for r in proof['records']]!=list(range(5)) or
        not all(r['actual_saved_model_reopened'] and r['every_independently_solved_coefficient_bit_equal'] for r in proof['records'])):
        raise ValueError('Complete actual independently rebuilt full models and vector proof required before outcomes')
    experiment.checks.verify_bindings(proof['source_sha256'])
    for filename,digest in frozen['outputs'].items():
        if u.sha(root/filename)!=digest:raise ValueError('Every frozen actual model and output must remain unchanged')
    path=root/'qualifying.f32'
    if u.sha(path)!=proof['prediction_sha256']:raise ValueError('Original independently replayed full qualifying vector required')
    prediction=np.fromfile(path,dtype='<f4')
    if prediction.shape!=(2817131,) or not np.isfinite(prediction).all():raise ValueError('Complete finite canonical candidate required')
    path=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered label payload differs')
    with gzip.open(path,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Complete recovered canonical domains required')
    prediction=np.clip(prediction.astype(np.float64),1.05,4.95);groups={}
    for group,count in [(0,1408789),(1,1408342)]:
        ids=np.flatnonzero(labels[:,1]==group)
        if len(ids)!=count:raise ValueError('Complete canonical recovered group count differs')
        score=math.sqrt(math.fsum((float(prediction[i])-float(labels[i,0]))**2 for i in ids)/count)
        groups['source'+str(group)]=dict(rows=count,rmse=score,below_target=score<.85725)
    experiment.checks.verify_bindings(proof['source_sha256'])
    u.write(output,dict(state='complete_independent_frozen_full_own_bank_fwls_canonical_recovered_audit',
        observed=experiment.checks.stamp(),groups=groups,both_groups_below_target=all(v['below_target'] for v in groups.values()),
        preaudit_verification_sha256=u.sha(report/'independent-preaudit-verification.json'),freeze_sha256=u.sha(report/'frozen.json'),
        prediction_sha256=proof['prediction_sha256'],label_git_blob=blob,
        qualifying_outcomes_used_for_fitting_or_selection=False,
        limitation='Adaptive community-recovered groups, not official or untouched Test. Original A/B input realization caveat retained.'))
    print(groups,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    audit() if args.audit else verify()
