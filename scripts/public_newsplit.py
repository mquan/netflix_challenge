#!/usr/bin/env python3
"""Reproduce the public project's trainx/probex split without target leakage.

This is a DIFFERENT evaluation set from the original full Probe. A small,
rating-independent subset of Probe is explicitly promoted into training.
All remaining ProbeX values stay out of every base-model fit. The purpose
is fitting a matching production blender for public newer model families.
Never report its score as original-Probe CV.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load, HEADER
from production_dataset import keys
from streaming_blend import digest


def choose_moved(d, original_source_map):
    # Source original Probe is numeric-movie ordered, with raw probe.txt user
    # ordering inside movies. It is NOT the sorted (movie, user) probex order.
    inverse=np.argsort(original_source_map)
    q=d['probe'][inverse]
    users=q['user']; counts=np.bincount(users,minlength=d['users'])
    unique,first=np.unique(users,return_index=True)
    order_users=unique[np.argsort(first,kind='stable')]
    by_user=np.argsort(users,kind='stable')
    query_offsets=np.r_[0,np.cumsum(counts)]
    rng=np.random.default_rng(42)
    check=np.random.default_rng(42)
    if check.bit_generator.random_raw()!=0xc621fbcd16d92688: raise ValueError('PCG64 reference mismatch')
    moved=[]
    for u in order_users:
        if counts[u]<=1: continue
        rows=by_user[query_offsets[u]:query_offsets[u+1]]
        lo,hi=map(int,d['train_offsets'][u:u+2])
        history=d['train_days'][lo:hi]
        qd=q['day'][rows]
        pos=np.searchsorted(history,qd)
        exists=pos<len(history)
        exists[exists]&=history[pos[exists]]==qd[exists]
        candidates=rows[~exists]
        if len(candidates):
            chosen=candidates[int(rng.choice(len(candidates)))]
            if rng.random()<.9: moved.append(int(inverse[chosen]))
    return np.sort(np.asarray(moved,dtype=np.uint32))


def build(source, old_map, raw_map, output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    d=load(source)
    original_map=np.fromfile(old_map,dtype='<u4')
    if not np.array_equal(np.sort(original_map),np.arange(len(d['probe']))): raise ValueError('invalid old map')
    raw=np.fromfile(raw_map,dtype='<u4')
    if len(raw)!=d['users'] or len(np.unique(raw))!=len(raw): raise ValueError('invalid raw-user map')
    moved=choose_moved(d,original_map)
    retained=np.ones(len(d['probe']),dtype=bool);retained[moved]=False
    original_rows=np.flatnonzero(retained).astype('<u4')
    probe=d['probe'][retained].copy()
    source_order=np.lexsort((raw[probe['user']],probe['item']))
    mapping=np.argsort(source_order).astype('<u4')
    moved.tofile(root/'moved-original-probe.u32')
    original_rows.tofile(root/'original-probe-rows.u32')
    mapping.tofile(root/'processed-to-probex.u32')
    n=np.diff(d['train_offsets']).astype(np.int64)
    owner=np.repeat(np.arange(d['users'],dtype=np.uint32),n)
    merged=np.concatenate((keys(owner,d['train_items'],d['train_days']),
                           keys(d['probe']['user'][moved],d['probe']['item'][moved],d['probe']['day'][moved])))
    del owner
    permutation=np.argsort(merged,kind='stable')
    merged=merged[permutation]
    ratings=np.concatenate((d['train_ratings'],d['probe']['rating'][moved]))[permutation]
    del permutation
    counts=n+np.bincount(d['probe']['user'][moved],minlength=d['users'])
    offsets=np.r_[0,np.cumsum(counts)].astype('<u8')
    h=np.fromfile(source,dtype=HEADER,count=1)
    h['train_count']=len(ratings);h['probe_count']=len(probe);h['mean']=float(ratings.astype(np.float64).mean())
    out=root/'trainx-probex.bin'
    with out.open('xb') as stream:
        for array in (h,offsets,(merged&65535).astype('<u2'),((merged>>16)&65535).astype('<u2'),ratings,
                      probe,d['implicit_offsets'],d['implicit_items'],d['implicit_days']):
            array.tofile(stream)
    report=dict(source_sha256=digest(source),dataset_sha256=digest(out),moved=len(moved),
                probe_count=len(probe),training_count=len(ratings),selection_uses_rating_values=False,
                original_rows_sha256=digest(root/'original-probe-rows.u32'),mapping_sha256=digest(root/'processed-to-probex.u32'),
                metric='Alternative ProbeX holdout; NOT original full-Probe CV',
                train_description='Original training plus predeclared rating-independent subset of original Probe',
                held_description='Every remaining ProbeX value excluded from predictor training')
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--original-map',required=True)
    ap.add_argument('--raw-user-map',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    print(json.dumps(build(a.data,a.original_map,a.raw_user_map,a.output)),flush=True)


if __name__=='__main__': main()
