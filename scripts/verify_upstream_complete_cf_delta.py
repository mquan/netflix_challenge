#!/usr/bin/env python3
"""Separate complete scalar/operator/statistic and held-prediction replay."""
import argparse
import hashlib
import math
from pathlib import Path
import upstream_complete_cf_delta as study
import numpy as np
import verify_upstream_own391_paired_data as original

u=study.u


class Independent:
    def __init__(self,query,m):
        self.query,self.m=query,m
        self.parent=study.parent.old.base.Inputs(query,m['original_rows'])

    def block(self,lo,hi):
        return original.block(self.parent,self.query,self.m['temporal_rows'],self.m['new_rows'],lo,hi)

    def close(self):self.parent.close()


def scalar(row,query,lo,hi):
    array=np.load(u.ROOT/row['paired_files'][query],mmap_mode='r',allow_pickle=False)
    if array.shape!=(study.COUNTS[query],) or array.dtype!=np.dtype('<f4'):
        raise ValueError('Complete finite original query required')
    values=np.maximum(np.minimum(array[lo:hi],np.float32(6)),np.float32(0)).astype('<f4')
    if not np.isfinite(values).all():raise ValueError('Invalid scalar input')
    return values


def assemble(z,rows,query,lo,hi):
    result=np.ones((hi-lo,467+len(rows)),dtype=np.float64)
    result[:,:466]=z[:,:466]
    for i,row in enumerate(rows,466):result[:,i]=scalar(row,query,lo,hi)
    result[:,-1]=z[:,-1]
    return result


def novelty():
    output=study.REPORT/'input-verification.json'
    if output.exists():raise ValueError('Existing full novelty proof retained')
    guard=study.Guard();m,added,bindings=study.sources()
    im=u.read(study.REPORT/'input-manifest.json');plan=u.read(study.REPORT/'plan.json')
    if (im['state']!='complete_all488_scalar_pairs_pending_independent_replay' or
        im['plan_sha256']!=u.sha(study.REPORT/'plan.json') or
        im['source_sha256']!=bindings or plan['source_sha256']!=bindings or plan['candidates']!=added or
        plan['old_scalar_columns']!=list(range(62))+list(range(137,466)) or
        im['qualifying_labels_read'] is not False or im['models_fitted']!=0):
        raise ValueError('Original complete frozen input study required')
    hashes={}
    for query,count in study.COUNTS.items():
        streams=[hashlib.sha256() for _ in range(488)];ops=[hashlib.sha256(),hashlib.sha256()]
        reader=Independent(query,m)
        try:
            for block,lo in enumerate(range(0,count,8192)):
                hi=min(lo+8192,count);z,f=reader.block(lo,hi)
                ops[0].update(z.tobytes());ops[1].update(f.tobytes())
                old=z[:,list(range(62))+list(range(137,466))].astype('<f4')
                old=np.maximum(np.minimum(old,np.float32(6)),np.float32(0))
                for column in range(391):streams[column].update(old[:,column].tobytes())
                for i,row in enumerate(added,391):streams[i].update(scalar(row,query,lo,hi).tobytes())
                if block%20==0:guard.check()
        finally:reader.close()
        if [h.hexdigest() for h in ops]!=m['virtual_c_order_sha256'][query]:
            raise ValueError('Original complete independent391 operators differ')
        hashes[query]=[h.hexdigest() for h in streams]
        print('Independent complete scalar novelty:',query,flush=True)
    if hashes!=im['clipped_sha256']:raise ValueError('Complete scalar payload hashes differ')
    seen={}
    for i in range(391):seen.setdefault((hashes['probex'][i],hashes['qual'][i]),'original391:%03d'%i)
    rows,aliases=[],[]
    for i,row in enumerate(added,391):
        key=(hashes['probex'][i],hashes['qual'][i])
        if key in seen:aliases.append(dict(id=row['id'],alias_of=seen[key],clipped_pair_sha256=list(key)))
        else:seen[key]=row['id'];rows.append(row)
    if (rows!=im['retained'] or aliases!=im['aliases'] or im['raw_predictions']!=391+len(rows) or
        im['design_columns']!=467+len(rows)):
        raise ValueError('Complete independent retained/alias partition differs')
    study.parent.check(bindings);guard.check()
    u.write(output,dict(state='complete_independent_all488_scalar_pairs_and_novelty',
        input_manifest_sha256=u.sha(study.REPORT/'input-manifest.json'),source_sha256=bindings,
        retained_additions=len(rows),exact_paired_aliases=len(aliases),original391_retained=True,
        both_complete_original_operator_streams_equal=True,qualifying_labels_read=False,models_fitted=0))
    print('Independent genuine additions:',len(rows),flush=True)


def fit():
    output=study.REPORT/'independent-verification.json'
    if output.exists():raise ValueError('Existing full ridge proof retained')
    guard=study.Guard();m,added,bindings=study.sources()
    im=u.read(study.REPORT/'input-manifest.json');fm=u.read(study.REPORT/'fit-manifest.json')
    if (fm['state']!='complete_fixed_lambda_delta_ridge_pending_independent_replay' or
        fm['input_manifest_sha256']!=u.sha(study.REPORT/'input-manifest.json') or
        fm['input_verification_sha256']!=u.sha(study.REPORT/'input-verification.json') or
        fm['source_sha256']!=bindings or fm['qualifying_labels_read'] is not False):
        raise ValueError('Complete original fixed-lambda study required')
    for name,d in fm['prepared_sha256'].items():
        if u.sha(study.RUN/name)!=d:raise ValueError('Actual stored fit artifact changed')
    y=np.load(study.RUN/'labels.npy');groups=np.load(study.RUN/'groups.npy');users=np.load(study.RUN/'users.npy')
    order=np.argsort(users,kind='stable')
    if (y.shape!=(1128164,) or groups.shape!=y.shape or set(np.unique(groups))!=set(range(10)) or
        np.any((users[order][1:]==users[order][:-1])&(groups[order][1:]//2!=groups[order][:-1]//2))):
        raise ValueError('Canonical known rows and user-held folds required')
    d=467+len(im['retained']);gram=np.zeros((10,d,d));rhs=np.zeros((10,d));stream=hashlib.sha256()
    reader=Independent('probex',m)
    try:
        for block,lo in enumerate(range(0,len(y),8192)):
            hi=min(lo+8192,len(y));z,_=reader.block(lo,hi);a=assemble(z,im['retained'],'probex',lo,hi)
            stream.update(a.tobytes())
            for group in range(10):
                keep=groups[lo:hi]==group;b=a[keep];gram[group]+=b.T@b;rhs[group]+=b.T@y[lo:hi][keep]
            if block%10==0:guard.check()
    finally:reader.close()
    saved=np.load(study.RUN/'stats.npz')
    if (stream.hexdigest()!=fm['known_design_sha256'] or
        not np.array_equal(gram.view('u8'),saved['gram'].view('u8')) or
        not np.array_equal(rhs.view('u8'),saved['rhs'].view('u8'))):
        raise ValueError('Every complete expanded design/statistic bit must match')
    penalty=np.eye(d)*1000.;penalty[-1,-1]=0
    weights=np.stack([np.linalg.solve(gram[np.arange(10)//2!=fold].sum(0)+penalty,
        rhs[np.arange(10)//2!=fold].sum(0)) for fold in range(5)])
    actual=np.load(study.RUN/'weights.npy')
    if actual.shape!=weights.shape or not np.array_equal(actual.view('u8'),weights.view('u8')):
        raise ValueError('Every actual fold coefficient must match')
    pred=np.fromfile(study.RUN/'ridge.f32',dtype='<f4')
    if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('Complete finite held predictions required')
    reader=Independent('probex',m)
    try:
        for block,lo in enumerate(range(0,len(y),8192)):
            hi=min(lo+8192,len(y));z,_=reader.block(lo,hi);a=assemble(z,im['retained'],'probex',lo,hi)
            for fold in range(5):
                held=groups[lo:hi]//2==fold
                expected=np.clip(a[held]@weights[fold],1.05,4.95).astype('<f4')
                if not np.array_equal(expected.view('u4'),pred[lo:hi][held].view('u4')):
                    raise ValueError('Every actual held prediction bit must match')
            if block%10==0:guard.check()
    finally:reader.close()
    canonical=np.clip(pred.astype(np.float64),1.05,4.95)
    rmse=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(canonical,y))/len(y))
    base=np.fromfile(study.parent.RUN/'ridge.f32',dtype='<f4')
    if base.shape!=y.shape or not np.isfinite(base).all():raise ValueError('Complete canonical baseline required')
    control=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(base.astype(np.float64),1.05,4.95),y))/len(y))
    if abs(rmse-fm['ridge_rmse'])>1e-12 or abs(control-fm['parent391_rmse'])>1e-12:
        raise ValueError('Canonical full compensated known scores differ')
    study.parent.check(bindings);guard.check()
    u.write(output,dict(state='complete_independent_expanded_CF95_TX50n_known_ridge_and_canonical_scores',
        fit_manifest_sha256=u.sha(study.REPORT/'fit-manifest.json'),source_sha256=bindings,
        raw_predictions=im['raw_predictions'],design_columns=d,all_operator_statistic_weight_prediction_bits_equal=True,
        parent391_rmse=control,ridge_rmse=rmse,improvement=control-rmse,
        qualifying_labels_read=False,qualifying_predictions_generated=False))
    print('Independent matched391/expanded ridge:',control,rmse,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['novelty','fit']);args=parser.parse_args()
    novelty() if args.action=='novelty' else fit()
