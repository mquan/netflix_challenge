#!/usr/bin/env python3
"""Full original391 plus128 local and67 remote candidate operators."""
from pathlib import Path
import numpy as np
import upstream_complete_trajectory_ops as original

COUNTS, RAW391 = original.COUNTS, original.RAW391
sha, relative, merge_bindings = original.sha, original.relative, original.merge_bindings
partition, extend, native_rows = original.partition, original.extend, original.native_rows


def scalar(root, row, query, lo, hi):
    if 'known_row_map' not in row or query == 'qual':
        return original.scalar(root,row,query,lo,hi)
    if query != 'probex' or not 0 <= lo <= hi <= COUNTS[query]:
        raise ValueError('Complete canonical query bounds required')
    values=np.load(Path(root)/relative(row['paired_files'][query]),mmap_mode='r',allow_pickle=False)
    mapping=np.load(Path(root)/relative(row['known_row_map']),mmap_mode='r',allow_pickle=False)
    if values.shape!=(1760494,) or values.dtype!=np.dtype('<f4') or mapping.shape!=(1128164,) or mapping.dtype!=np.dtype('<i8'):
        raise ValueError('Full original native query and canonical projection required')
    selected=mapping[lo:hi]
    if np.any(selected<0) or np.any(selected>=len(values)):raise ValueError('Complete native projection bounds required')
    raw=values[selected]
    if not np.isfinite(raw).all():raise ValueError('Finite original remote predictions required before clipping')
    return np.clip(raw,0,6).astype('<f4')
