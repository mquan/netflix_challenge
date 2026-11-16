#!/usr/bin/env python3
"""Replay frozen ridge inference and independently rescore both recovered groups."""
import argparse
import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from qual_matrix import verified_matrix,predict_matrix_ridge
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',required=True);ap.add_argument('--labels',required=True)
    ap.add_argument('--audit-report',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.model_dir);m=json.loads((root/'report.json').read_text())
    audit=json.loads(Path(a.audit_report).read_text())
    if m['state']!='complete' or m['qualifying_labels_read'] is not False or m['qualifying_feature_values_used_for_fitting'] is not False:
        raise ValueError('weights were not frozen before query inference')
    predpath=root/'qualifying-final.f32'
    if digest(predpath)!=m['frozen_prediction_sha256'] or digest(root/'weights.npy')!=m['weights_sha256']:
        raise ValueError('winning artifact changed')
    bank=m['config']['bank'];qual=m['config']['qual']
    if digest(bank+'.inputs.json')!=m['bank_sha256'] or digest(bank+'.stats.npz')!=m['stats_sha256'] or digest(qual+'.json')!=m['qual_manifest_sha256']:
        raise ValueError('frozen model inputs changed')
    b=json.loads(Path(bank+'.inputs.json').read_text());q=json.loads(Path(qual+'.json').read_text())
    if digest(qual+'.context.npy')!=q['context_sha256']:raise ValueError('query context changed')
    replay=np.clip(predict_matrix_ridge(verified_matrix(qual,q),np.load(qual+'.context.npy',mmap_mode='r'),
        b['columns'],np.load(root/'weights.npy')),1,5).astype('<f4')
    expected=np.fromfile(predpath,dtype='<f4')
    if not np.array_equal(replay,expected):raise ValueError('frozen inference replay is not bitwise identical')
    # Only after verifying/replaying frozen inference, read the saved audit
    # label blob. The scoring implementation below uses stdlib CSV and
    # compensated summation, separately from audit_qualifying.py / NumPy.
    blob=Path(a.labels).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=audit['label_sha256'] or hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()!=audit['label_git_blob_sha1']:
        raise ValueError('pinned audit labels changed')
    if sys.byteorder!='little':raise ValueError('expected little-endian artifact host')
    values=array.array('f');values.frombytes(predpath.read_bytes())
    totals=[0.,0.];compensation=[0.,0.];counts=[0,0];n=0
    with gzip.open(a.labels,'rt') as stream:
        reader=csv.reader(stream)
        if next(reader)!=['rating','is_test']:raise ValueError('label header changed')
        for n,row in enumerate(reader,1):
            rating,group=map(int,row)
            if group not in [0,1] or not 1<=rating<=5 or n>len(values):raise ValueError('invalid audit row')
            loss=(float(values[n-1])-rating)**2
            corrected=loss-compensation[group];updated=totals[group]+corrected
            compensation[group]=(updated-totals[group])-corrected;totals[group]=updated;counts[group]+=1
    if n!=len(values) or n!=2817131 or counts!=[1408789,1408342]:raise ValueError('row counts differ')
    scores=[math.sqrt(t/c) for t,c in zip(totals,counts)]
    for i,name in enumerate(['quiz','test']):
        if abs(scores[i]-audit[name]['rmse'])>1e-12:raise ValueError('independent score differs')
    if not all(v<=.85725 for v in scores):raise ValueError('target unmet')
    report=dict(state='verified_target_met',prediction_sha256=digest(predpath),weights_sha256=m['weights_sha256'],
        frozen_model_report_sha256=digest(root/'report.json'),inference_replay_bitwise_identical=True,
        independent_scoring='stdlib CSV, float32 artifact, float64 errors, compensated summation',
        groups={str(i):dict(rows=counts[i],rmse=scores[i]) for i in [0,1]},
        target_rmse=.85725,baseline_rmse=.9525,improvement_percent=100*(1-scores[1]/.9525),
        both_source_groups_below_target=True,margin_worst_group=.85725-max(scores),
        label_sha256=audit['label_sha256'],source_count_naming_warning=audit['split_provenance_warning'],
        limitation='Adaptive community recovered-label audit, not official Netflix scoring or an untouched private test',
        predictions_or_weights_modified=False)
    path=Path(a.output)
    with path.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
