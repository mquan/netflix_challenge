#!/usr/bin/env python3
"""Independently verify all retained cache paths, contents and original manifests."""
from pathlib import Path
import hashlib
import json
import datetime as dt
import numpy as np
import upstream_exact_cache_links as source
u=source.u

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block=stream.read(1<<20)
            if not block:break
            h.update(block)
    return h.hexdigest()

def verify():
    report=source.REPORT
    if (report/'independent-verification.json').exists():raise ValueError('Existing independent storage proof preserved')
    receipt=u.read(report/'receipt.json');plan=u.read(report/'plan.json')
    if (receipt['state']!='complete_exact_inactive_cache_storage_shared' or receipt['plan_sha256']!=digest(report/'plan.json') or
        not receipt['all_original_paths_bytes_and_manifest_hashes_preserved'] or receipt['datasets_models_checkpoints_predictions_and_distinct_caches_removed'] is not False):
        raise ValueError('Complete exact-path storage receipt required')
    expected=[(f'artifacts/{a}/{n}',f'artifacts/{b}/{n}') for a,b,n in source.PAIRS]
    if [(r['canonical'],r['target']) for r in receipt['records']]!=expected:raise ValueError('Only the four declared inactive cache pairs allowed')
    records=[]
    for record in receipt['records']:
        a,b=[u.ROOT/record[k] for k in ['canonical','target']]
        if not record['full_byte_comparison_passed'] or not record['both_original_paths_and_hashes_preserved'] or record['candidate_open_handles']!=0:
            raise ValueError('Original complete comparison/open-handle proof required')
        if not a.is_file() or not b.is_file():raise ValueError('Original cache path missing')
        sa,sb=a.stat(),b.stat()
        if (sa.st_dev,sa.st_ino)!=(sb.st_dev,sb.st_ino) or sa.st_size!=record['canonical_before']['size']:
            raise ValueError('Retained path/size/shared storage differs')
        for path,st,key in [(a,sa,'canonical_before'),(b,sb,'target_before')]:
            before=record[key]
            if any(getattr(st,'st_'+field)!=before[field] for field in ['mode','uid','gid']):raise ValueError('Original permissions/ownership changed')
            if digest(path)!=record['sha256']:raise ValueError('Original full file bytes/hash differ')
        aa,bb=np.load(a,mmap_mode='r'),np.load(b,mmap_mode='r')
        if aa.shape!=bb.shape or aa.dtype!=bb.dtype or not np.shares_memory(aa,aa):raise ValueError('Retained array headers differ')
        records.append(dict(canonical=record['canonical'],target=record['target'],sha256=record['sha256'],shape=list(aa.shape),dtype=str(aa.dtype),both_original_paths_and_hashes_verified=True,shared_storage_inode_verified=True))
    for name,value in receipt['source_sha256'].items():
        if digest(u.ROOT/name)!=value:raise ValueError('Original manifest or storage source changed')
    u.write(report/'independent-verification.json',dict(state='complete_independent_exact_cache_path_and_content_verification',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        receipt_sha256=digest(report/'receipt.json'),verifier_sha256=digest(Path(__file__)),records=records,
        all_original_manifest_hashes_unchanged=True,all_cache_values_and_paths_preserved=True,qualifying_labels_read=False,accuracy_claimed=False))
    print('Four retained cache pairs independently rehashed; paths, array schemas and manifests unchanged',flush=True)
if __name__=='__main__':verify()
