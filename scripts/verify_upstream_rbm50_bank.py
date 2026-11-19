#!/usr/bin/env python3
"""Check every paired own107/RBM50 column against its actual native source."""
from pathlib import Path
import numpy as np
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-rbm50-bank'
REPORT=u.ROOT/'experiments/upstream-rbm50-bank'
OLD=u.ROOT/'artifacts/upstream-cf48-103-bank'
DEV=u.ROOT/'artifacts/upstream-rbm50-features'

def verify():
    import upstream_latent90_early_support as support
    path=REPORT/'manifest.json';m=u.read(path)
    for collection in [m['code_sha256'],m['source_sha256']]:
        for name,digest in collection.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source changed: '+name)
    for name,digest in m['files'].items():
        if u.sha(RUN/name)!=digest:raise ValueError('Bank changed: '+name)
    expected=['s3-attn16','s3-attn16_ep08','cfnade-96_ep10','tsvdx5-60']
    if [r['name'] for r in m['paired_additions']]!=expected or m['support_names']!=[f'rbm50lat.u{j:02}' for j in range(50)]:
        raise ValueError('Appended schema differs')
    counts={}
    for split,rows in [('probex',1128164),('qual',2817131)]:
        z=np.load(RUN/(split+'-design.npy'),mmap_mode='r');f=np.load(RUN/(split+'-features.npy'),mmap_mode='r')
        oz=np.load(OLD/(split+'-design.npy'),mmap_mode='r');of=np.load(OLD/(split+'-features.npy'),mmap_mode='r')
        if z.shape!=(rows,183) or f.shape!=(rows,293) or z.dtype!=np.float64 or f.dtype!=np.float32:
            raise ValueError('Shape/precision mismatch')
        raw=[np.load(u.ROOT/r['paired_files'][split],mmap_mode='r') for r in m['paired_additions']]
        latent=[np.load(support.OUT/f'{name}.{split}.npy',mmap_mode='r') for name in m['support_names']]
        dz=np.load(DEV/'design.npy',mmap_mode='r') if split=='probex' else None
        df=np.load(DEV/'features.npy',mmap_mode='r') if split=='probex' else None
        for lo in range(0,rows,16384):
            hi=min(lo+16384,rows)
            if not np.array_equal(z[lo:hi,:178],oz[lo:hi,:178]) or not np.array_equal(z[lo:hi,-1],oz[lo:hi,-1]):
                raise ValueError('Retained ridge changed')
            if not np.array_equal(f[lo:hi,:239].view(np.uint32),of[lo:hi].view(np.uint32)):
                raise ValueError('Retained context changed')
            for j,a in enumerate(raw):
                clipped=np.minimum(np.maximum(a[lo:hi],np.float32(0)),np.float32(6));centered=clipped-np.float32(3.6)
                if not np.isfinite(clipped).all() or not np.array_equal(z[lo:hi,178+j],clipped) or not np.array_equal(f[lo:hi,239+j],centered):
                    raise ValueError('Raw transform differs')
                if split=='probex' and (not np.array_equal(dz[lo:hi,178+j],clipped) or not np.array_equal(df[lo:hi,239+j],centered)):
                    raise ValueError('New prediction differs from originating development cache')
            for j,a in enumerate(latent):
                if not np.isfinite(a[lo:hi]).all() or not np.array_equal(f[lo:hi,243+j].view(np.uint32),a[lo:hi].view(np.uint32)):
                    raise ValueError('Original latent values differ')
                if split=='probex' and not np.array_equal(f[lo:hi,243+j].view(np.uint32),df[lo:hi,243+j].view(np.uint32)):
                    raise ValueError('Latent differs from originating development cache')
        counts[split]=rows
    for name in ['labels','users']:
        if u.sha(RUN/(name+'.npy'))!=u.sha(OLD/(name+'.npy')) or not np.array_equal(np.load(RUN/(name+'.npy')),np.load(DEV/(name+'.npy'))):
            raise ValueError('Known identity mismatch')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    result=dict(state='complete',manifest_sha256=u.sha(path),verifier_sha256=u.sha(Path(__file__)),counts=counts,
        prediction_columns=107,support_columns=50,all_retained_columns_checked=True,all_four_raw_transforms_checked=True,
        all_fifty_support_columns_bitwise_checked=True,added_known_columns_bitwise_match_development=True,
        retained_parent_realization_may_differ_from_development=True,qualifying_labels_read=False,
        scope='Every paired ridge/nonlinear column and known identity checked. Native50 support gathers previously independently replayed on all rows. Prior paired-parent realization differences retained. No model fitting, inference replay or qualifying score claimed.')
    u.write(REPORT/'column-verification.json',result);print(result,flush=True)

if __name__=='__main__':verify()
