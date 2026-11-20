#!/usr/bin/env python3
"""Independently reopen every completed retired-public archive and verify restoration."""
import argparse
from pathlib import Path
import upstream_repro as u
import upstream_public_stats_archive as helpers

def verify(namespace):
    if namespace not in ['upstream-retired-public-stats-archive','upstream-retired-public-features-archive','upstream-retired-public-prediction-archive-v2']:raise ValueError('Explicit retired-public transaction required')
    report=u.ROOT/'experiments'/namespace;source=report/'verification.json';proof=u.read(source)
    if proof.get('original_bytes_discarded_without_archive')!=0:raise ValueError('Every original byte must remain archived')
    records=[]
    for row in proof['records']:
        original=Path(row['original']);archive=Path(row['archive'])
        if not any(original.is_relative_to(helpers.OLD/'artifacts'/name) for name in ['public-bank','public-new']):raise ValueError('Archive origin outside explicit retired public roots')
        if archive!=original.with_name(original.name+'.zst') or archive.is_symlink() or not archive.is_file():raise ValueError('Exact original archive location required')
        if helpers.u.sha(archive)!=row['archive_sha256'] or archive.stat().st_size!=row['archive_bytes']:raise ValueError('Actual archive bytes changed')
        sha,count=helpers.decode_digest(archive)
        if sha!=row['original_sha256'] or count!=row['original_bytes']:raise ValueError('Full exact original-byte restoration fails')
        if original.exists() and u.sha(original)!=row['original_sha256']:raise ValueError('Restored historical path differs')
        if not row['uncompressed_cache_retired'] or not row['exact_full_decode_sha256_verified']:raise ValueError('Original complete transaction receipt required')
        records.append(dict(original=str(original),archive_sha256=row['archive_sha256'],full_original_sha256=sha,full_original_bytes=count,exact_original_bytes_recoverable=True))
    u.write(report/'independent-verification.json',dict(state='complete_independent_all_retired_public_archive_restorations',
        verification_sha256=u.sha(source),verifier_sha256=u.sha(Path(__file__)),records=records,archives_reopened=len(records),
        all_original_bytes_independently_recoverable=True,own_training_inputs_modified=False))
    print(namespace,len(records),'full archives independently reopened; all original bytes recoverable',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('namespace');args=p.parse_args();verify(args.namespace)
