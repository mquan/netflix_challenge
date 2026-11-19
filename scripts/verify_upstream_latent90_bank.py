#!/usr/bin/env python3
"""Check every retained and appended paired90 column and known identity."""
from pathlib import Path
import numpy as np
import upstream_latent90_bank as bank

u=bank.u

def verify():
    path=bank.REPORT/'manifest.json';m=u.read(path)
    for collection in [m['source_sha256'],m['code_sha256']]:
        for name,digest in collection.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound input changed')
    for name,digest in m['files'].items():
        if u.sha(bank.RUN/name)!=digest:raise ValueError('Paired cache changed')
    if m['new_support_names']!=[f'tx20lat.{kind}{j:02}' for kind in ['u','i'] for j in range(20)]:raise ValueError('Original component order differs')
    paths,_=bank.development.tx_inputs();counts={}
    for split,rows in [('probex',1128164),('qual',2817131)]:
        f=np.load(bank.RUN/(split+'-features.npy'),mmap_mode='r')
        old=np.load(bank.parent.RUN/(split+'-features.npy'),mmap_mode='r')
        raw=[np.load(p,mmap_mode='r') for p in paths[split]]
        df=np.load(bank.development.RUN/'features.npy',mmap_mode='r') if split=='probex' else None
        if f.shape!=(rows,333) or f.dtype!=np.float32:raise ValueError('Schema differs')
        for lo in range(0,rows,16384):
            hi=min(lo+16384,rows)
            if not np.array_equal(f[lo:hi,:293].view(np.uint32),old[lo:hi].view(np.uint32)):raise ValueError('Retained columns differ')
            for j,a in enumerate(raw):
                if not np.array_equal(f[lo:hi,293+j].view(np.uint32),a[lo:hi].view(np.uint32)) or not np.isfinite(a[lo:hi]).all():
                    raise ValueError('Original Tx component differs')
            if split=='probex' and not np.array_equal(f[lo:hi,293:].view(np.uint32),df[lo:hi,293:].view(np.uint32)):
                raise ValueError('New components differ from originating development cache')
        counts[split]=rows
    for name in ['probex-design.npy','qual-design.npy','labels.npy','users.npy']:
        if u.sha(bank.RUN/name)!=u.sha(bank.parent.RUN/name):raise ValueError('Retained ridge/identity changed')
    for name in ['labels.npy','users.npy']:
        if not np.array_equal(np.load(bank.RUN/name),np.load(bank.development.RUN/name)):raise ValueError('Known row identity differs')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    result=dict(state='complete',manifest_sha256=u.sha(path),verifier_sha256=u.sha(Path(__file__)),counts=counts,
        prediction_columns=107,learned_support_columns=90,all_retained_columns_checked=True,
        all_forty_new_support_columns_bitwise_checked=True,added_known_columns_bitwise_match_development=True,
        ridge_and_known_identities_byte_identical=True,retained_parent_realization_may_differ_from_development=True,
        qualifying_labels_read=False,scope='Every retained293/appended40 feature column checked on both query sets. Unchanged ridge and known identities; previous all-row native gather proofs retained. Prior paired-parent realization differences disclosed. No model fit, inference replay or qualifying score claimed.')
    u.write(bank.REPORT/'column-verification.json',result);print(result,flush=True)

if __name__=='__main__':verify()
