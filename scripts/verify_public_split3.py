#!/usr/bin/env python3
"""Check reconstructed split_3 against independently published metadata.

Uses no Probe rating values. Checks every original Probe query against three
published columns: training-history count, query date, and same-day training
rating mean relative to the training user mean. The latter verifies the exact
random removal set, not merely plausible ordering or standalone model scores.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--mapping',required=True)
    ap.add_argument('--checks',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    d=load(a.data); mapping=np.fromfile(a.mapping+'.u32',dtype='<u4')
    moved=np.fromfile(a.mapping+'.moved.u32',dtype='<u4')
    if len(mapping)!=len(d['probe']) or len(np.unique(mapping))!=len(mapping): raise ValueError('invalid mapping')
    n=np.diff(d['train_offsets']).astype(np.int64)
    offsets=d['train_offsets'].astype(np.int64)
    moved_user=np.searchsorted(offsets,moved,side='right')-1
    removed=np.bincount(moved_user,minlength=d['users'])
    remain=n-removed
    qu=d['probe']['user']; qd=d['probe']['day'].astype(np.int64)
    expected_count=np.log(np.maximum(remain[qu],1)).astype(np.float32)
    # Source day zero is 1999-11-11, our day zero is 1999-01-01.
    expected_day=np.log(np.maximum(qd-314+1,1)).astype(np.float32)
    days=d['train_days']; ratings=d['train_ratings']
    boundary=np.r_[True,days[1:]!=days[:-1]]
    boundary[offsets[:-1][n>0]]=True
    starts=np.flatnonzero(boundary)
    session_owner=np.searchsorted(offsets,starts,side='right')-1
    key=session_owner*65536+days[starts].astype(np.int64)
    counts=np.diff(np.r_[starts,len(days)]).astype(np.int64)
    sums=np.add.reduceat(ratings,starts).astype(np.int64)
    moved_session=np.searchsorted(starts,moved,side='right')-1
    counts-=np.bincount(moved_session,minlength=len(starts))
    sums-=np.bincount(moved_session,weights=ratings[moved],minlength=len(starts)).astype(np.int64)
    totals=np.zeros(d['users'],dtype=np.float64)
    active=np.flatnonzero(n)
    totals[active]=np.add.reduceat(ratings,offsets[active])
    totals-=np.bincount(moved_user,weights=ratings[moved],minlength=d['users'])
    user_mean=(totals/np.maximum(remain,1)).astype(np.float32)
    qkey=qu.astype(np.int64)*65536+qd
    where=np.searchsorted(key,qkey)
    exists=where<len(key)
    exists[exists]&=key[where[exists]]==qkey[exists]
    expected_mean=np.zeros(len(mapping),dtype=np.float32)
    take=where[exists]
    if np.any(counts[take]<=0): raise ValueError('removed a whole rating day')
    expected_mean[exists]=sums[take].astype(np.float32)/counts[take].astype(np.float32)-user_mean[qu[exists]]
    names=['vf004_log_user_cnt','vf050_log_day','vf084_same_day_mean_rating']
    errors={}
    for name,expected in zip(names,[expected_count,expected_day,expected_mean]):
        source=np.load(Path(a.checks)/(name+'.npy'),allow_pickle=False)
        actual=source[mapping]
        error=float(np.max(np.abs(actual-expected)))
        errors[name]=error
        print(name,'max_error',error,flush=True)
        if error>5e-7: raise ValueError('independent split/order check failed: '+name)
    report=dict(rows=len(mapping),source_rows=len(source),moved=len(moved),metadata_max_errors=errors,
                mapping_sha256=digest(a.mapping+'.u32'),moved_sha256=digest(a.mapping+'.moved.u32'),
                dataset_sha256=digest(a.data),probe_rating_values_used=False,
                original_probe_excluded_from_source_training=True,
                check_hashes={name:digest(Path(a.checks)/(name+'.npy')) for name in names})
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
