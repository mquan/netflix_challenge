#!/usr/bin/env python3
"""Paired own107/item154 known features with streamed qualifying context."""
from pathlib import Path
import numpy as np
import upstream_latent90_bank as parent
import upstream_item154_features as development
import upstream_item64_support as support

u=parent.u
RUN=u.ROOT/'artifacts/upstream-item154-stream-bank'
REPORT=u.ROOT/'experiments/upstream-item154-stream-bank'

def qual_chunk(lo,hi):
    z=np.load(parent.RUN/'qual-design.npy',mmap_mode='r')
    old=np.load(parent.RUN/'qual-features.npy',mmap_mode='r')
    added=np.load(support.RUN/'qual.npy',mmap_mode='r')
    return z[lo:hi],np.column_stack([old[lo:hi],added[lo:hi]]).astype(np.float32)

def build():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing item154 paired evidence preserved')
    pm=u.read(parent.REPORT/'manifest.json');pp=parent.REPORT/'column-verification.json';proof=u.read(pp)
    if proof['state']!='complete' or proof['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json'):raise ValueError('Verified paired90 required')
    dm=u.read(development.REPORT/'manifest.json');dp=development.REPORT/'cache-column-verification.json';checked=u.read(dp)
    if checked['state']!='complete_cache_columns' or checked['manifest_sha256']!=u.sha(development.REPORT/'manifest.json') or dm['nonlinear_features']!=397:
        raise ValueError('Verified original item154 development schema required')
    _,sources=development.checked_support()
    for collection in [pm['source_sha256'],pm['code_sha256']]:
        for name,d in collection.items():
            if name in sources and sources[name]!=d:raise ValueError('Conflicting source lineage')
            sources[name]=d
    for name,d in pm['files'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=d
    for p in [parent.REPORT/'manifest.json',pp,development.REPORT/'manifest.json',dp]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,d in sources.items():
        if u.sha(u.ROOT/name)!=d:raise ValueError('Original own context source differs')
    RUN.mkdir();REPORT.mkdir()
    for name in ['probex-design.npy','labels.npy','users.npy']:u.link_or_copy(parent.RUN/name,RUN/name)
    old=np.load(parent.RUN/'probex-features.npy',mmap_mode='r');extra=np.load(support.RUN/'probex.npy',mmap_mode='r')
    f=np.lib.format.open_memmap(RUN/'probex-features.npy',mode='w+',dtype=np.float32,shape=(1128164,397))
    for lo in range(0,len(f),8192):f[lo:lo+8192,:333]=old[lo:lo+8192];f[lo:lo+8192,333:]=extra[lo:lo+8192]
    f.flush();del f
    # Verify every retained/new known column against the paired parent and
    # development additions. Prior paired-parent realization differences stay explicit.
    f=np.load(RUN/'probex-features.npy',mmap_mode='r');df=np.load(development.RUN/'features.npy',mmap_mode='r')
    for lo in range(0,len(f),16384):
        if not np.array_equal(f[lo:lo+16384,:333].view(np.uint32),old[lo:lo+16384].view(np.uint32)) or not np.array_equal(f[lo:lo+16384,333:].view(np.uint32),df[lo:lo+16384,333:].view(np.uint32)):
            raise ValueError('Known paired/item context columns differ')
    oq=np.load(parent.RUN/'qual-features.npy',mmap_mode='r');eq=np.load(support.RUN/'qual.npy',mmap_mode='r')
    for lo in range(0,2817131,65536):
        hi=min(lo+65536,2817131);z,f=qual_chunk(lo,hi)
        if z.shape!=(hi-lo,183) or not np.array_equal(f[:,:333].view(np.uint32),oq[lo:hi].view(np.uint32)) or not np.array_equal(f[:,333:].view(np.uint32),eq[lo:hi].view(np.uint32)):
            raise ValueError('Every streamed qualifying context column must match')
    for name in ['labels.npy','users.npy']:
        if not np.array_equal(np.load(RUN/name),np.load(development.RUN/name)):raise ValueError('Known row identity differs')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    for name,d in sources.items():
        if u.sha(u.ROOT/name)!=d:raise ValueError('Original source changed during verification')
    u.write(REPORT/'manifest.json',dict(state='complete_verified_streaming_bank',rows=1128164,qualifying_rows=2817131,
        prediction_columns=107,design_columns=183,nonlinear_features=397,learned_support_columns=154,
        source_sha256=sources,code_sha256={str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))},
        files={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,retained_parent_realization_may_differ_from_development=True,
        scope='Preserve paired107/all90 and append original independently verified64 own item components. Qualifying construction is streamed; previous paired-parent differences remain disclosed.'))
    u.write(REPORT/'column-verification.json',dict(state='complete',manifest_sha256=u.sha(REPORT/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),
        all_retained_columns_checked=True,all_item_components_checked=True,added_known_columns_bitwise_match_development=True,
        all_streamed_qualifying_columns_compared_bitwise=True,qualifying_labels_read=False,
        scope='Every paired known/qualifying context column checked. Separate covariance/eigenpair and original factor/query gather proof bound. No fitting or new accuracy claimed.'))

if __name__=='__main__':build()
