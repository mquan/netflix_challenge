#!/usr/bin/env python3
"""Independently verify public query order and fitting data from metadata.

Probe/query label values are never read. Every statistic is computed from
physically permitted training arrays only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from streaming_blend import digest


def expected_metadata(d):
    n=np.diff(d['train_offsets']).astype(np.int64)
    offsets=d['train_offsets'].astype(np.int64)
    users=d['probe']['user']; qd=d['probe']['day'].astype(np.int64)
    days=d['train_days']; ratings=d['train_ratings']
    count=np.log(np.maximum(n[users],1)).astype(np.float32)
    day=np.log(np.maximum(qd-314+1,1)).astype(np.float32)
    boundary=np.r_[True,days[1:]!=days[:-1]]
    boundary[offsets[:-1][n>0]]=True
    starts=np.flatnonzero(boundary)
    owner=np.searchsorted(offsets,starts,side='right')-1
    key=owner*65536+days[starts].astype(np.int64)
    counts=np.diff(np.r_[starts,len(days)])
    sums=np.add.reduceat(ratings,starts)
    totals=np.zeros(d['users'],dtype=np.float64)
    active=np.flatnonzero(n)
    totals[active]=np.add.reduceat(ratings,offsets[active])
    user_mean=(totals/np.maximum(n,1)).astype(np.float32)
    qkey=users.astype(np.int64)*65536+qd
    where=np.searchsorted(key,qkey)
    exists=where<len(key)
    exists[exists]&=key[where[exists]]==qkey[exists]
    mean=np.zeros(len(users),dtype=np.float32)
    take=where[exists]
    mean[exists]=sums[take].astype(np.float32)/counts[take].astype(np.float32)-user_mean[users[exists]]
    return dict(vf004_log_user_cnt=count,vf050_log_day=day,vf084_same_day_mean_rating=mean)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--mapping',required=True)
    ap.add_argument('--checks',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--metric-label',required=True)
    a=ap.parse_args()
    d=load(a.data); mapping=np.fromfile(a.mapping,dtype='<u4')
    if not np.array_equal(np.sort(mapping),np.arange(len(d['probe']))): raise ValueError('mapping is not bijective')
    expected=expected_metadata(d); errors={}
    for name,col in expected.items():
        values=np.load(Path(a.checks)/(name+'.npy'),allow_pickle=False)
        if values.shape!=(len(mapping),): raise ValueError('source row count differs: '+name)
        error=float(np.max(np.abs(values[mapping]-col)))
        errors[name]=error
        print(name,'max_error',error,flush=True)
        if error>5e-7: raise ValueError('independent metadata mismatch: '+name)
    report=dict(rows=len(mapping),metric=a.metric_label,metadata_max_errors=errors,
                data_sha256=digest(a.data),mapping_sha256=digest(a.mapping),query_rating_values_used=False,
                source_checks={name:digest(Path(a.checks)/(name+'.npy')) for name in expected})
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
