#!/usr/bin/env python3
"""Shrink user/day residual means learned from available Probe labels only.

Screening leaves out each rating, while allowing other Probe labels for the
same user. The neural reference excludes the entire user. This availability
protocol differs from holding out whole users, and is reported explicitly.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from streaming_blend import digest


def statistics(users,days,residual,nusers):
    keys,groups=np.unique(users.astype('u8')*65536+days,return_inverse=True)
    return dict(user_count=np.bincount(users,minlength=nusers),
        user_sum=np.bincount(users,weights=residual,minlength=nusers),
        day_keys=keys,day_count=np.bincount(groups),day_sum=np.bincount(groups,weights=residual)),groups


def leave_one_out(users,residual,groups,stats,user_ridge,day_ridge):
    n=stats['user_count'][users]-1;s=stats['user_sum'][users]-residual
    user=s/(n+user_ridge)
    dn=stats['day_count'][groups]-1;ds=stats['day_sum'][groups]-residual
    return user+(ds-dn*user)/(dn+day_ridge)


def corrections(users,days,stats,user_ridge,day_ridge):
    user=stats['user_sum'][users]/(stats['user_count'][users]+user_ridge)
    keys=users.astype('u8')*65536+days;position=np.searchsorted(stats['day_keys'],keys)
    valid=position<len(stats['day_keys']);safe=np.minimum(position,len(stats['day_keys'])-1)
    valid &= stats['day_keys'][safe]==keys
    dn=np.where(valid,stats['day_count'][safe],0);ds=np.where(valid,stats['day_sum'][safe],0)
    return user+(ds-dn*user)/(dn+day_ridge)


def fit(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    d=load(a.data);y=d['probe']['rating'].astype('f8');u=d['probe']['user'].copy();day=d['probe']['day'].copy()
    base=np.clip(np.fromfile(a.reference,'<f4').astype('f8'),1,5)
    if base.shape!=y.shape or not np.isfinite(base).all():raise ValueError('invalid reference')
    ref=json.loads(Path(a.reference_report).read_text())
    bank=ref['config']['bank'];b=json.loads(Path(bank+'.inputs.json').read_text())
    if b['data_sha256']!=digest(a.data) or ref['bank_input_sha256']!=digest(bank+'.inputs.json'):
        raise ValueError('reference development data differs')
    residual=y-base;stats,groups=statistics(u,day,residual,d['users']);scores=[]
    for ur in [10,20,30,50,100,300,1e30]:
        for dr in [5,10,20,30,50,100,1e30]:
            pred=np.clip(base+leave_one_out(u,residual,groups,stats,ur,dr),1,5)
            scores.append(dict(user_ridge=ur,day_ridge=dr,rmse=float(np.sqrt(np.mean((pred-y)**2)))))
    best=min(scores,key=lambda row:row['rmse']);np.savez(root/'statistics.npz',**stats)
    report=dict(state='models_ready',kind='user_day_residual_calibration',selected=best,
        reference_rmse=float(np.sqrt(np.mean((base-y)**2))),candidates=scores,
        reference_sha256=digest(a.reference),reference_report_sha256=digest(a.reference_report),
        reference_bank_sha256=digest(bank+'.inputs.json'),data_sha256=digest(a.data),
        users=d['users'],development_rows=len(y),statistics_sha256=digest(root/'statistics.npz'),
        qualifying_labels_read=False,training_labels='Original Probe labels, available in production known ratings',
        validation_protocol='Leave one Probe rating out, allowing other Probe ratings of that user; NN reference excludes entire user; this differs from original user-fold validation',
        predeclared_variants=['selected','double_penalty'])
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def apply(a):
    from audit_qualifying import frozen_predictions
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    model=Path(a.models_dir);m=json.loads((model/'report.json').read_text())
    if m['state']!='models_ready' or m['qualifying_labels_read'] is not False:raise ValueError('invalid calibration model')
    if digest(model/'statistics.npz')!=m['statistics_sha256']:raise ValueError('calibration statistics changed')
    base,parent=frozen_predictions(a.base_model_dir)
    manifest=json.loads(Path(a.dataset_manifest).read_text());path=Path(a.dataset_manifest).parent/'train-probe-qualifying.bin'
    if digest(path)!=manifest['outputs'][path.name]:raise ValueError('query data changed')
    d=load(path)
    if d['users']!=m['users'] or not np.all(d['probe']['rating']==0) or len(d['probe'])!=len(base):
        raise ValueError('query identities/order or unknown labels differ')
    stats=dict(np.load(model/'statistics.npz'));selected=m['selected']
    correction=corrections(d['probe']['user'],d['probe']['day'],stats,
                          selected['user_ridge']*a.penalty_multiplier,selected['day_ridge']*a.penalty_multiplier)
    final=np.clip(base.astype('f8')+correction,1,5).astype('<f4');final.tofile(root/'qualifying-final.f32')
    report=dict(state='complete',kind='user_day_calibrated_prediction',qualifying_labels_read=False,
        metric=None,qualifying_rows=len(final),frozen_prediction_sha256=digest(root/'qualifying-final.f32'),
        prediction_min=float(final.min()),prediction_max=float(final.max()),
        calibration_report_sha256=digest(model/'report.json'),statistics_sha256=m['statistics_sha256'],
        base_model_dir=a.base_model_dir,base_model_report_sha256=digest(Path(a.base_model_dir)/'report.json'),
        base_prediction_sha256=parent['frozen_prediction_sha256'],
        reference_bank_sha256=m['reference_bank_sha256'],base_bank_sha256=parent.get('bank_sha256'),
        calibration_transferred_to_different_blend=parent.get('bank_sha256')!=m['reference_bank_sha256'],
        penalty_multiplier=a.penalty_multiplier,selected=selected,
        dataset_manifest_sha256=digest(a.dataset_manifest),prediction_correction_rms=float(np.sqrt(np.mean(correction**2))))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit')
    for name in ['data','reference','reference-report','output']:f.add_argument('--'+name,required=True)
    p=sub.add_parser('apply')
    for name in ['models-dir','base-model-dir','dataset-manifest','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--penalty-multiplier',type=float,default=1.)
    a=ap.parse_args()
    if a.command=='fit':fit(a)
    else:
        if a.penalty_multiplier<=0:raise ValueError('penalty multiplier must be positive')
        apply(a)


if __name__=='__main__':main()
