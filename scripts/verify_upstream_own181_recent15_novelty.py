#!/usr/bin/env python3
"""Check every old/new canonical raw payload for exact duplicate additions."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
from pathlib import Path
import numpy as np
import upstream_own181_recent15_data as data

u=data.u

def main():
    path=data.REPORT/'canonical-novelty-verification.json'
    if path.exists():raise ValueError('Existing complete raw-payload check retained')
    manifest=u.read(data.REPORT/'manifest.json');data.check(manifest['source_sha256'])
    old=np.load(data.original.RUN/'design.npy',mmap_mode='r')
    # Original design: raw62, metadata5, interactions14x5, further raw104,
    # then intercept. Do not mistake interaction/context columns for predictors.
    positions=list(range(62))+list(range(137,241))
    hashes=[hashlib.sha256() for _ in range(166)]
    new=[hashlib.sha256() for _ in range(15)]
    arrays=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in manifest['recent_rows']]
    for lo in range(0,len(old),8192):
        hi=min(lo+8192,len(old));raw=old[lo:hi,positions];cast=raw.astype(np.float32)
        if not np.array_equal(raw,cast.astype(np.float64)):raise ValueError('Original raw predictors must be exactly representable in their native float32')
        for j,h in enumerate(hashes):h.update(cast[:,j].tobytes(order='C'))
        for j,h in enumerate(new):h.update(np.clip(arrays[j][lo:hi],0,6).tobytes(order='C'))
    old_keys=[h.hexdigest() for h in hashes];new_keys=[h.hexdigest() for h in new]
    collisions=[dict(name=manifest['recent_rows'][i]['name'],original_raw_column=j,original_design_column=positions[j]) for i,h in enumerate(new_keys) for j,key in enumerate(old_keys) if h==key]
    if len(set(new_keys))!=15:raise ValueError('Recent canonical clipped predictors duplicate one another')
    data.check(manifest['source_sha256'])
    u.write(path,dict(state='complete_all_original_and_recent_canonical_raw_payload_fingerprints',
        manifest_sha256=u.sha(data.REPORT/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),rows=len(old),
        original_design_raw_columns=positions,original_raw_payload_sha256=old_keys,recent_raw_payload_sha256=new_keys,
        recent_names=[r['name'] for r in manifest['recent_rows']],exact_canonical_matches_to_original=collisions,
        all_fifteen_canonical_additions_distinct_from_original=not collisions,qualifying_labels_read=False,
        scope='Complete canonical clipped scalar payload equality only. No claim of statistical independence or full qualifying-pair novelty against old166; recent-fifteen full-pair duplicate proof remains separately bound.'))
    print('Complete canonical raw novelty: fifteen additions;',len(collisions),'exact matches to old166',flush=True)

if __name__=='__main__':main()
