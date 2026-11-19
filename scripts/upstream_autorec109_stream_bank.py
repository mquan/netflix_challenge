#!/usr/bin/env python3
"""Verified paired109 known cache and exact streamed qualifying transforms."""
from pathlib import Path
import numpy as np
import upstream_autorec109_bank as original

u=original.u
RUN=u.ROOT/'artifacts/upstream-autorec109-stream-bank'
REPORT=u.ROOT/'experiments/upstream-autorec109-stream-bank'

def qual_chunk(lo,hi):
    parent=original.parent.RUN
    z=np.load(parent/'qual-design.npy',mmap_mode='r');f=np.load(parent/'qual-features.npy',mmap_mode='r')
    rows=original.development.screen.selected()
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['qual'],mmap_mode='r')[lo:hi],0,6) for r in rows])
    return (np.column_stack([z[lo:hi,:182],extra,z[lo:hi,-1]]),
            np.column_stack([f[lo:hi,:243],extra-np.float32(3.6),f[lo:hi,243:]]).astype(np.float32))

def build():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing streaming-bank evidence preserved')
    m=u.read(original.REPORT/'manifest.json');p=u.read(original.REPORT/'column-verification.json')
    if p['state']!='complete' or p['manifest_sha256']!=u.sha(original.REPORT/'manifest.json'):
        raise ValueError('Actual fully checked paired109 source required')
    sources={**m['source_sha256'],**m['code_sha256']}
    for name,d in sources.items():
        if u.sha(u.ROOT/name)!=d:raise ValueError('Original source changed')
    for path in [original.REPORT/'manifest.json',original.REPORT/'column-verification.json']:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    RUN.mkdir();REPORT.mkdir()
    for name in ['probex-design.npy','probex-features.npy','labels.npy','users.npy']:
        if u.sha(original.RUN/name)!=m['files'][name]:raise ValueError('Original known cache differs')
        u.link_or_copy(original.RUN/name,RUN/name)
    # Compare every streamed row with the separately verified materialized
    # representation before retiring the unused qualifying matrices.
    z=np.load(original.RUN/'qual-design.npy',mmap_mode='r');f=np.load(original.RUN/'qual-features.npy',mmap_mode='r')
    for lo in range(0,len(z),65536):
        hi=min(lo+65536,len(z));a,b=qual_chunk(lo,hi)
        if not np.array_equal(a.view(np.uint64),z[lo:hi].view(np.uint64)) or not np.array_equal(b.view(np.uint32),f[lo:hi].view(np.uint32)):
            raise ValueError('Streamed qualifying transform differs bitwise')
    for name,d in sources.items():
        if u.sha(u.ROOT/name)!=d:raise ValueError('Source changed during verification')
    manifest=dict(state='complete_verified_streaming_bank',rows=1128164,qualifying_rows=2817131,prediction_columns=109,
        design_columns=185,nonlinear_features=335,learned_support_columns=90,source_sha256=sources,
        code_sha256={str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))},
        files={p.name:u.sha(p) for p in RUN.iterdir()},paired_additions=original.development.screen.selected(),
        qualifying_storage='Exact chunk construction from preserved paired90 and fixed own snapshot5/10 raw vectors. Materialized qualifying matrices are not required inputs.',
        qualifying_labels_read=False,retained_parent_realization_may_differ_from_development=True)
    u.write(REPORT/'manifest.json',manifest)
    u.write(REPORT/'column-verification.json',dict(state='complete',manifest_sha256=u.sha(REPORT/'manifest.json'),
        verifier_sha256=u.sha(Path(__file__)),all_retained_columns_checked=True,all_two_raw_transforms_checked=True,
        added_known_columns_bitwise_match_development=True,all_streamed_qualifying_columns_compared_bitwise=True,
        rows=2817131,qualifying_labels_read=False,scope='Every streamed qualifying column matches the independently checked materialized bank bitwise. Known matrices are byte-identical hardlinks. Original factors/predictions remain preserved; no fitting or qualifying outcomes.'))

if __name__=='__main__':build()
