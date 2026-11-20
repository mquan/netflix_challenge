#!/usr/bin/env python3
"""Rebuild every FWLS interaction/group system, reopen all five models and replay."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
from scipy.linalg.blas import dsyrk
import upstream_own13_fwls as experiment

u=experiment.u
STATE='complete_independent_all_own13_fwls_interactions_ten_group_systems_five_models_and_full_vectors'


def basis(arrays,contexts,lo,hi):
    values=np.ones((hi-lo,183),dtype=np.float64)
    for i,a in enumerate(arrays):
        for j,c in enumerate(contexts):
            values[:,i*14+j]=a[lo:hi].astype(np.float64)*c[lo:hi].astype(np.float64)
    return values


def replay(arrays,contexts,weights,lo,hi):
    result=np.repeat(np.float64(weights[-1]),hi-lo)
    for i in range(13):
        terms=np.zeros(hi-lo)
        for j in range(14):terms=terms+contexts[j][lo:hi].astype(np.float64)*float(weights[i*14+j])
        result=result+terms*arrays[i][lo:hi].astype(np.float64)
    return np.clip(result,1.05,4.95)


def verify():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-preaudit-verification.json'
    if output.exists():raise ValueError('Existing independent FWLS evidence retained')
    plan=u.read(report/'plan.json');frozen=u.read(report/'frozen.json')
    if (frozen['state']!='complete_own13_fwls_all_five_userfold_models_and_uniform_qualifying_bag_before_audit' or
        frozen['plan_sha256']!=u.sha(report/'plan.json') or frozen['qualifying_labels_read'] is not False or
        plan['script_sha256']!=u.sha(Path(experiment.__file__)) or plan['verifier_sha256']!=u.sha(Path(__file__)) or
        plan['declaration_sha256']!=u.sha(experiment.DECL) or plan['lambda_']!=30. or
        plan['outer_user_folds']!=5 or plan['dimension']!=183):raise ValueError('Original complete FWLS freeze required')
    rows,names,sources=experiment.checked_sources()
    if rows!=plan['families'] or names!=plan['context_names'] or sources!=plan['source_sha256'] or sources!=frozen['source_sha256']:
        raise ValueError('Original13-family/14-gate source expectations differ')
    for filename,digest in frozen['outputs'].items():
        if u.sha(root/filename)!=digest:raise ValueError('Actual saved FWLS artifact changed')
    labels=np.load(experiment.own6.LABELS).astype(np.float64)
    groups=np.load(u.ROOT/'artifacts/upstream-item154-tree/groups.npy')
    users=np.load(u.ROOT/'artifacts/upstream-item154-tree/users.npy')
    order=np.argsort(users,kind='stable')
    if np.any((users[order][1:]==users[order][:-1]) & (groups[order][1:]//2!=groups[order][:-1]//2)):
        raise ValueError('Original held-user groups overlap')
    arrays,contexts=experiment.inputs(rows,names,'probex')
    gram=np.zeros((10,183,183));rhs=np.zeros((10,183));digest=hashlib.sha256()
    for lo in range(0,len(labels),3072):
        hi=min(lo+3072,len(labels));z=basis(arrays,contexts,lo,hi);digest.update(z.tobytes(order='C'))
        for group in range(10):
            selected=groups[lo:hi]==group;block=z[selected]
            lower=dsyrk(1.,np.asfortranarray(block.T),lower=1)
            gram[group]+=np.tril(lower)+np.tril(lower,-1).T
            rhs[group]+=np.einsum('ij,i->j',block,labels[lo:hi][selected],optimize=False)
        experiment.check_disk()
    saved=np.load(root/'grouped-statistics.npz')
    if (digest.hexdigest()!=frozen['known_interaction_c_order_sha256'] or
        not np.allclose(gram,saved['gram'],rtol=2e-12,atol=1e-7) or
        not np.allclose(rhs,saved['rhs'],rtol=2e-12,atol=1e-7)):
        raise ValueError('Complete independent interactions/group statistics differ')
    known=np.fromfile(root/'known.f32',dtype='<f4')
    if known.shape!=(1128164,) or not np.isfinite(known).all():raise ValueError('Complete finite held vector required')
    penalty=np.eye(183)*30.;penalty[-1,-1]=0;weights=[];records=[]
    for fold in range(5):
        eligible=[g for g in range(10) if g//2!=fold]
        actual=np.load(root/('fold-%s.npz'%fold));w=actual['weights']
        system=sum(gram[g] for g in eligible)+penalty;target=sum(rhs[g] for g in eligible)
        independent=np.linalg.solve(system,target)
        if (w.shape!=(183,) or not np.isfinite(w).all() or actual['lambda_']!=30. or
            actual['eligible_groups'].tolist()!=eligible or not np.allclose(w,independent,atol=1e-6,rtol=1e-6) or
            np.max(abs(system@w-target))/len(labels)>1e-8):raise ValueError('Actual saved full FWLS normal-equation solution differs')
        for lo in range(0,len(labels),3072):
            hi=min(lo+3072,len(labels));held=groups[lo:hi]//2==fold
            expected=replay(arrays,contexts,w,lo,hi).astype('<f4')
            if not np.array_equal(expected[held].view('u4'),known[lo:hi][held].view('u4')):raise ValueError('Every held prediction must replay exactly')
        weights.append(w);records.append(dict(fold=fold,complete_eligible_group_system_rebuilt=True,
            actual_saved_model_reopened=True,held_prediction_bitwise_equal=True,
            independent_coefficient_max_error=float(np.max(abs(w-independent)))))
    arrays,contexts=experiment.inputs(rows,names,'qual');actual=np.fromfile(root/'qualifying.f32',dtype='<f4')
    if actual.shape!=(2817131,) or not np.isfinite(actual).all():raise ValueError('Complete finite qualifying bag required')
    for lo in range(0,len(actual),3072):
        hi=min(lo+3072,len(actual));expected=np.zeros(hi-lo)
        for w in weights:expected=expected+replay(arrays,contexts,w,lo,hi)
        expected=(expected/5).astype('<f4')
        if not np.array_equal(expected.view('u4'),actual[lo:hi].view('u4')):raise ValueError('All five qualifying model replays and uniform bag must match exactly')
    experiment.own6.verify_bindings(sources)
    clipped=np.clip(known.astype(np.float64),1.05,4.95)
    score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(clipped,labels))/len(labels))
    u.write(output,dict(state=STATE,observed=experiment.own6.stamp(),plan_sha256=u.sha(report/'plan.json'),
        freeze_sha256=u.sha(report/'frozen.json'),verifier_sha256=u.sha(Path(__file__)),source_sha256=sources,
        all_interaction_bits_reconstructed=True,all_ten_complete_group_systems_rebuilt=True,records=records,
        all_five_qualifying_models_and_uniform_bag_bitwise_equal=True,known_rows=1128164,qualifying_rows=2817131,
        known_meta_crossfit_rmse=score,fully_nested_end_to_end_cv=False,qualifying_labels_read=False,
        prediction_sha256=u.sha(root/'qualifying.f32')))
    print('All13 FWLS independently verified; meta-crossfit RMSE',score,flush=True)


def audit():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-recovered-audit.json'
    if output.exists():raise ValueError('Existing recovered FWLS audit retained')
    proof=u.read(report/'independent-preaudit-verification.json');frozen=u.read(report/'frozen.json')
    if (proof['state']!=STATE or proof['qualifying_labels_read'] is not False or
        proof['freeze_sha256']!=u.sha(report/'frozen.json') or proof['plan_sha256']!=u.sha(report/'plan.json') or
        proof['verifier_sha256']!=u.sha(Path(__file__)) or frozen['plan_sha256']!=u.sha(report/'plan.json') or
        len(proof['records'])!=5 or not proof['all_five_qualifying_models_and_uniform_bag_bitwise_equal']):raise ValueError('Actual complete independent FWLS proof required before outcomes')
    experiment.own6.verify_bindings(proof['source_sha256'])
    path=root/'qualifying.f32'
    if u.sha(path)!=proof['prediction_sha256'] or u.sha(path)!=frozen['outputs']['qualifying.f32']:
        raise ValueError('Frozen independently replayed qualifying vector differs')
    prediction=np.fromfile(path,dtype='<f4')
    if prediction.shape!=(2817131,) or not np.isfinite(prediction).all():raise ValueError('Complete finite qualifying vector required')
    path=u.ROOT/'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob!='47ed4788ffba11aad029854a8e60d4dee04fc899':raise ValueError('Canonical recovered blob differs')
    with gzip.open(path,'rt') as stream:labels=np.loadtxt(stream,delimiter=',',skiprows=1,dtype=np.int16)
    if labels.shape!=(2817131,2) or not np.isin(labels[:,0],[1,2,3,4,5]).all() or not np.isin(labels[:,1],[0,1]).all():raise ValueError('Canonical complete domains required')
    prediction=np.clip(prediction.astype(np.float64),1.05,4.95);groups={}
    for group,count in [(0,1408789),(1,1408342)]:
        ids=np.flatnonzero(labels[:,1]==group)
        if len(ids)!=count:raise ValueError('Canonical group count differs')
        score=math.sqrt(math.fsum((float(prediction[i])-float(labels[i,0]))**2 for i in ids)/count)
        groups['source'+str(group)]=dict(rows=count,rmse=score,below_target=score<.85725)
    experiment.own6.verify_bindings(proof['source_sha256'])
    u.write(output,dict(state='complete_independent_frozen_own13_fwls_canonical_recovered_audit',
        observed=experiment.own6.stamp(),groups=groups,both_groups_below_target=all(v['below_target'] for v in groups.values()),
        preaudit_verification_sha256=u.sha(report/'independent-preaudit-verification.json'),prediction_sha256=proof['prediction_sha256'],
        label_git_blob=blob,qualifying_outcomes_used_for_fitting_or_selection=False,
        limitation='Adaptive community-recovered audit, not official or untouched Test.'))
    print(groups,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    audit() if args.audit else verify()
