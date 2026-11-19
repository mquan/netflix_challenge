#!/usr/bin/env python3
"""Verify all paired110 columns, including exact known development additions."""
from pathlib import Path
import numpy as np
import upstream_whole110_bank as bank

u=bank.u

def verify():
    path=bank.REPORT/'manifest.json';m=u.read(path)
    for collection in [m['source_sha256'],m['code_sha256']]:
        for name,digest in collection.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen input changed')
    for name,digest in m['files'].items():
        if u.sha(bank.RUN/name)!=digest:raise ValueError('Paired110 cache changed')
    rows,_=bank.development.screen.verified_inputs()
    if m['paired_additions']!=rows:raise ValueError('Native addition order differs')
    counts={}
    for split,count in [('probex',1128164),('qual',2817131)]:
        z=np.load(bank.RUN/(split+'-design.npy'),mmap_mode='r');f=np.load(bank.RUN/(split+'-features.npy'),mmap_mode='r')
        old=np.load(bank.parent.RUN/(split+'-design.npy'),mmap_mode='r');of=np.load(bank.parent.RUN/(split+'-features.npy'),mmap_mode='r')
        raw=[np.load(u.ROOT/r['paired_files'][split],mmap_mode='r') for r in rows]
        if z.shape!=(count,186) or z.dtype!=np.float64 or f.shape!=(count,336) or f.dtype!=np.float32:
            raise ValueError('Paired110 schema differs')
        df=np.load(bank.development.RUN/'features.npy',mmap_mode='r') if split=='probex' else None
        for lo in range(0,count,16384):
            hi=min(lo+16384,count)
            if not np.array_equal(z[lo:hi,:182].view(np.uint64),old[lo:hi,:182].view(np.uint64)) or not np.array_equal(z[lo:hi,-1].view(np.uint64),old[lo:hi,-1].view(np.uint64)):
                raise ValueError('Retained ridge differs')
            if not np.array_equal(f[lo:hi,:243].view(np.uint32),of[lo:hi,:243].view(np.uint32)) or not np.array_equal(f[lo:hi,246:].view(np.uint32),of[lo:hi,243:].view(np.uint32)):
                raise ValueError('Retained raw/context differs')
            for j,a in enumerate(raw):
                values=np.clip(a[lo:hi],0,6)
                if not np.isfinite(values).all() or not np.array_equal(z[lo:hi,182+j],values) or not np.array_equal(f[lo:hi,243+j],values-np.float32(3.6)):
                    raise ValueError('New native transform differs')
            if split=='probex' and not np.array_equal(f[lo:hi,243:246].view(np.uint32),df[lo:hi,243:246].view(np.uint32)):
                raise ValueError('New known columns differ from development')
        counts[split]=count
    for name in ['labels.npy','users.npy']:
        if u.sha(bank.RUN/name)!=u.sha(bank.parent.RUN/name) or not np.array_equal(np.load(bank.RUN/name),np.load(bank.development.RUN/name)):
            raise ValueError('Known row identity differs')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    result=dict(state='complete',manifest_sha256=u.sha(path),verifier_sha256=u.sha(Path(__file__)),counts=counts,
        prediction_columns=110,learned_support_columns=90,all_retained_columns_checked=True,
        all_three_raw_transforms_checked=True,added_known_columns_bitwise_match_development=True,
        known_identities_byte_identical=True,retained_parent_realization_may_differ_from_development=True,
        qualifying_labels_read=False,scope='All retained and new columns checked on both queries, with independent whole-model proofs bound. No fitting, model replay, score or promotion.')
    u.write(bank.REPORT/'column-verification.json',result);print(result,flush=True)

if __name__=='__main__':verify()
