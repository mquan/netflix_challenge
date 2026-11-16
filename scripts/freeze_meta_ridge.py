#!/usr/bin/env python3
"""Select/freeze stacking ridge weights on known development ratings only."""
import argparse
import json
from pathlib import Path
import numpy as np
from ridge_selection import choose,refit
from streaming_blend import digest
from qual_matrix import verified_matrix,predict_matrix_ridge


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    b=json.loads(Path(a.bank+'.inputs.json').read_text());q=json.loads(Path(a.qual+'.json').read_text())
    if q['state']!='complete' or q['qualifying_labels_read'] is not False or q['bank_sha256']!=digest(a.bank+'.inputs.json'):
        raise ValueError('unverified query schema')
    s=np.load(a.bank+'.stats.npz');stats=tuple(s[k] for k in ['gram','rhs','count'])
    groups=np.arange(10);recipe=choose(stats,groups%2==0,groups%2==1)
    w=refit(stats,np.ones(10,dtype=bool),recipe)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);np.save(root/'weights.npy',w,allow_pickle=False)
    report=dict(state='weights_frozen',config=vars(a),production_recipe=recipe,
        bank_sha256=digest(a.bank+'.inputs.json'),stats_sha256=digest(a.bank+'.stats.npz'),
        weights_sha256=digest(root/'weights.npy'),qualifying_labels_read=False,
        qualifying_feature_values_used_for_fitting=False,qualifying_rows=q['rows'],
        development_limitation=b.get('pretrained_blender_policy',{}),
        metric=None,qual_manifest_sha256=digest(a.qual+'.json'))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    raw=verified_matrix(a.qual,q)
    if digest(a.qual+'.context.npy')!=q['context_sha256']:raise ValueError('context changed')
    c=np.load(a.qual+'.context.npy',mmap_mode='r')
    p=np.clip(predict_matrix_ridge(raw,c,b['columns'],w),1,5).astype('<f4')
    if p.shape!=(q['rows'],) or not np.isfinite(p).all():raise ValueError('invalid predictions')
    p.tofile(root/'qualifying-final.f32')
    if digest(root/'weights.npy')!=report['weights_sha256']:raise ValueError('weights changed during export')
    report.update(state='complete',frozen_prediction_sha256=digest(root/'qualifying-final.f32'))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
