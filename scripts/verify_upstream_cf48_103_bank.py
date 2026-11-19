#!/usr/bin/env python3
"""Independently check every paired103 column against its retained/raw source."""
from pathlib import Path
import numpy as np
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-cf48-103-bank'
REPORT=u.ROOT/'experiments/upstream-cf48-103-bank'
OLD=u.ROOT/'artifacts/upstream-extension-bank'
DEV=u.ROOT/'artifacts/upstream-cf48-103'

def verify():
    path=REPORT/'manifest.json';m=u.read(path)
    for collection in [m['code_sha256'],m['source_sha256']]:
        for name,digest in collection.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Source hash mismatch: '+name)
    for name,digest in m['files'].items():
        if u.sha(RUN/name)!=digest:raise ValueError('Bank hash mismatch: '+name)
    final=u.read(u.ROOT/'experiments/upstream-remote-cf48-final/paired-receipt.json')
    for name,digest in final['outputs'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Resumed CF48 output differs from intake receipt')
    expected=['tsvdx4-60o','tsvdx4-q16tb','dnn-24__asym-16p75','dnn-24__dnn-16',
        'dnn-24__dnn-16p50__asym-16p75','mf-61__asym-16','mf-61__asym-16_ep02',
        'mf-61__asym-16_ep07','rbmx2-0','tsvdx4-l32nb_ep03','cfnade-48_ep25','cfnade-48_ep95']
    if m['fixed_additions']!=expected or [r['name'] for r in m['paired_additions']]!=expected:
        raise ValueError('Fixed paired103 predictor order mismatch')
    counts={}
    for split,rows in [('probex',1128164),('qual',2817131)]:
        z=np.load(RUN/(split+'-design.npy'),mmap_mode='r')
        f=np.load(RUN/(split+'-features.npy'),mmap_mode='r')
        oz=np.load(OLD/(split+'-design.npy'),mmap_mode='r')
        of=np.load(OLD/(split+'-features.npy'),mmap_mode='r')
        if z.shape!=(rows,179) or f.shape!=(rows,239):raise ValueError('Shape mismatch')
        if z.dtype!=np.float64 or f.dtype!=np.float32:raise ValueError('Precision mismatch')
        raw=[np.load(u.ROOT/r['files'][split],mmap_mode='r') for r in m['paired_additions']]
        dz=np.load(DEV/'design.npy',mmap_mode='r') if split=='probex' else None
        df=np.load(DEV/'features.npy',mmap_mode='r') if split=='probex' else None
        for lo in range(0,rows,16384):
            hi=min(lo+16384,rows)
            if not np.array_equal(z[lo:hi,:166],oz[lo:hi,:166]) or not np.array_equal(z[lo:hi,-1],oz[lo:hi,-1]):
                raise ValueError('Retained ridge columns/intercept changed')
            if not np.array_equal(f[lo:hi,:227],of[lo:hi]):raise ValueError('Retained context changed')
            for j,a in enumerate(raw):
                if a.shape!=(rows,) or a.dtype!=np.float32:raise ValueError('Raw prediction malformed')
                clipped=np.minimum(np.maximum(a[lo:hi],np.float32(0)),np.float32(6))
                centered=clipped-np.float32(3.6)
                if not np.isfinite(clipped).all() or not np.array_equal(z[lo:hi,166+j],clipped) or not np.array_equal(f[lo:hi,227+j],centered):
                    raise ValueError('Raw column transform mismatch')
                if split=='probex' and (not np.array_equal(dz[lo:hi,166+j],clipped) or not np.array_equal(df[lo:hi,227+j],centered)):
                    raise ValueError('New columns differ from originating development cache')
        counts[split]=rows
    for name in ['labels','users']:
        if u.sha(RUN/(name+'.npy'))!=u.sha(OLD/(name+'.npy')) or not np.array_equal(np.load(RUN/(name+'.npy')),np.load(DEV/(name+'.npy'))):
            raise ValueError('Known row identity mismatch')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    result=dict(state='complete',manifest_sha256=u.sha(path),verifier_sha256=u.sha(Path(__file__)),
        counts=counts,prediction_columns=103,all_retained_columns_checked=True,all_twelve_raw_transforms_checked=True,
        added_known_columns_bitwise_match_development=True,retained_parent_realization_may_differ_from_development=True,
        cf48_final_native_hashes_match_verified_intake=True,qualifying_labels_read=False,
        scope='Exact paired-bank input/output hashes and every retained/appended column checked. Previously disclosed paired91 parent differences retained. No new training, fitted blend replay, or qualifying score claimed.')
    u.write(REPORT/'column-verification.json',result);print(result,flush=True)

if __name__=='__main__':verify()
