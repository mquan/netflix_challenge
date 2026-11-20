#!/usr/bin/env python3
"""Separate full-byte and metadata replay of additional inactive CoW operations."""
from pathlib import Path
import filecmp
import json
import upstream_cow_cache_storage as common
ROOT=common.ROOT

def main():
    paths=[ROOT/'experiments'/name/'verification.json' for name in
        ['upstream-duplicate-prediction-cow','upstream-secondary-storage-cow']]
    records=[];total=0
    for report in paths:
        proof=json.loads(report.read_text())
        if proof['logical_files_removed']!=0 or proof['numerical_values_changed'] is not False:
            raise ValueError('Original logical files and complete bytes must remain')
        for row in proof['records']:
            source=ROOT/row['source'];destination=ROOT/row['destination']
            if common.sha(source)!=row['sha256'] or common.sha(destination)!=row['sha256'] or not filecmp.cmp(source,destination,shallow=False):
                raise ValueError('Complete original byte comparison differs')
            state=common.identity(destination)
            if any(state[k]!=row['before'][k] for k in ['bytes','mode','uid','gid','mtime_ns','nlink','flags']):
                raise ValueError('Original metadata differs')
            if source.stat().st_ino==destination.stat().st_ino:raise ValueError('Independent CoW inode required')
            total+=row['bytes']
        records.append(dict(report=str(report.relative_to(ROOT)),sha256=common.sha(report),
            operations=len(proof['records']),logical_bytes_shared=proof['logical_bytes_shared'],
            observed_free_gain_bytes=proof['free_after']-proof['free_before']))
    result=dict(state='complete_independent_all_bytes_metadata_and_separate_inodes',observed=common.stamp(),
        records=records,logical_bytes_shared=total,verifier_sha256=common.sha(Path(__file__)),
        logical_files_removed=0,original_values_preserved=True,
        scope='Every original source/destination byte rehashed and compared independently; original modes/owners/mtime/flags/linkcount preserved. Existing APFS clone fixture proves separate-inode write isolation. No datasets or live model files changed. Physical free-space deltas are observations during concurrent jobs.')
    path=ROOT/'experiments/upstream-additional-storage-cow-verification.json'
    if path.exists():raise ValueError('Existing independent storage evidence preserved')
    common.write(path,result);print(total,records,flush=True)

if __name__=='__main__':main()
