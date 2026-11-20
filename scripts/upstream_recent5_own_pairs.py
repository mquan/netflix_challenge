#!/usr/bin/env python3
"""Verify all five newly available own scalar predictors; no qualifying outcomes."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import math
from pathlib import Path
import numpy as np
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-recent5-own-pairs'
EXPECTED=['s3-dnn24','s3-dnn24_ep03','s3-attn32','s3-attn32_ep06','s3-rbm200']

def main():
    if REPORT.exists():raise ValueError('Existing complete own-pair inventory preserved')
    REPORT.mkdir();sources={};rows=[]
    def expect(key,digest):
        if key in sources and sources[key]!=digest:raise ValueError('Conflicting original own source')
        sources[key]=digest
    u.write(REPORT/'plan.json',dict(columns=EXPECTED,
        rule='All newly complete remoteDNN24 two/attention32 two plus localRBM200 terminal, selected by completed own producer coverage. Verify all pairs, no subset/weight search, no fitting or qualifying outcomes. Remote full-probe files remain remote; separate exporter provenance is explicit.'))
    for job in ['s3-dnn24','s3-attn32']:
        path=u.ROOT/f'experiments/upstream-remote15h-return/{job}-v3/local-verification.json';r=u.read(path)
        if (r['state']!='complete_local_compact_remote15h_return_check' or r['source_pin']!=u.PIN or
            not r['original_recipe_trace_and_immutable_export_verified'] or
            not r['local_full_known_map_and_training_disjointness_recomputed'] or
            not r['native_qualifying_npy_bytes_reconstructed_exactly'] or
            not r['native_full_probe_projection_recomputed_by_separate_remote_export'] or
            r['qualifying_labels_read'] is not False or r['public_prediction_values_read'] is not False):raise ValueError('Own remote full producer/transfer verification required')
        for key,digest in r['bindings'].items():expect(key,digest)
        expect(str(path.relative_to(u.ROOT)),u.sha(path))
        rows.extend(dict(name=row['name'],paired_files=row['paired_files'],origin=job,
            native_full_probe_present_locally=False,known_projection_scope='Separate original remote exporter plus local complete canonical map/disjointness replay') for row in r['rows'])
    path=u.ROOT/'experiments/upstream-split3-next3/s3-rbm200-verification.json';r=u.read(path)
    if (r['state']!='complete_original_two_phase_split3_model' or r['source_pin']!=u.PIN or r['epochs_per_phase']!=25 or
        not r['both_full_epoch_traces'] or not r['full_known_projection_recomputed'] or
        not r['known_training_disjointness_bound_to_unchanged_original_data'] or not r['qualifying_targets_zero'] or
        r['qualifying_labels_read'] is not False or r['public_prediction_values_read'] is not False):raise ValueError('Full own local25/25 native proof required')
    for key,digest in r['bindings'].items():expect(key,digest)
    expect(str(path.relative_to(u.ROOT)),u.sha(path))
    for row in r['projected_pairs']:
        for query in ['probex','qual']:expect(row[query],row[query+'_sha256'])
        rows.append(dict(name=row['name'],paired_files={k:row[k] for k in ['probex','qual']},origin='s3-rbm200',
            native_full_probe_present_locally=True,known_projection_scope='Local complete original native probe projection independently recomputed'))
    if [row['name'] for row in rows]!=EXPECTED:raise ValueError('All five fixed own columns required')
    labels=u.FRESH/'data/probex/ratings.npy';expect(str(labels.relative_to(u.ROOT)),u.sha(labels))
    expect(str(Path(__file__).relative_to(u.ROOT)),u.sha(Path(__file__)))
    identities={};cache={}
    for key,digest in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=digest:raise ValueError('Original producer/transfer lineage changed: '+key)
        identities[key]=identity
    y=np.load(labels);vectors=[]
    for row in rows:
        arrays={}
        for query,count in [('probex',1128164),('qual',2817131)]:
            p=u.ROOT/row['paired_files'][query];a=np.load(p,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Complete finite scalar pair required')
            key=str(p.relative_to(u.ROOT))
            if key not in sources:expect(key,u.sha(p))
            arrays[query]=a
        for previous in vectors:
            if all(np.array_equal(previous[k].view('u4'),arrays[k].view('u4')) for k in arrays):raise ValueError('Exact duplicate new predictor pair')
        vectors.append(arrays)
        row['known_raw_rmse']=math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(np.clip(arrays['probex'].astype(np.float64),1.05,4.95),y))/len(y))
    for key,identity in identities.items():
        s=(u.ROOT/key).stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=identity:raise ValueError('Original lineage changed during inventory')
    u.write(REPORT/'verification.json',dict(state='complete_five_fresh_own_pairs_and_original_producer_lineage',rows=rows,
        source_sha256=sources,qualifying_labels_read=False,public_prediction_values_read=False,models_fitted=0,
        scope='Five own newly completed pairs, complete shape/finite/duplicate checks and original producer/transfer lineage. Known raw scores are diagnostics, not meta CV or qualifying scores. No source subset selected.'))
    print([(row['name'],row['known_raw_rmse']) for row in rows],flush=True)

if __name__=='__main__':main()
