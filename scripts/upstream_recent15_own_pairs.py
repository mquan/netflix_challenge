#!/usr/bin/env python3
"""Bind all fifteen completed own pairs, including the first two remote extensions."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
from pathlib import Path
import numpy as np
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-recent15-own-pairs'

def main():
    if REPORT.exists():raise ValueError('Existing complete fifteen-pair proof preserved')
    REPORT.mkdir();base_path=u.ROOT/'experiments/upstream-recent10-own-pairs-v2/verification.json';base=u.read(base_path)
    if base['state']!='complete_ten_fresh_own_pairs_and_original_producer_lineage' or len(base['rows'])!=10:raise ValueError('Complete actual first ten own pairs required')
    rows=list(base['rows']);sources=dict(base['source_sha256'])
    for job,count in [('s3-bknbrx-100w',2),('s3-mfrbmx-60h',3)]:
        path=u.ROOT/'experiments/upstream-remote15h-ext-return'/(job+'-v1')/'local-verification.json';p=u.read(path)
        if (p['state']!='complete_local_compact_remote15h_extension_return_check' or p['original_full_epochs_each']!=12 or len(p['rows'])!=count or
            not p['original_recipe_trace_and_immutable_export_verified'] or not p['native_qualifying_npy_bytes_reconstructed_exactly'] or
            not p['local_full_known_map_and_training_disjointness_recomputed'] or p['qualifying_labels_read'] or p['public_prediction_values_read']):raise ValueError('Actual full original extension and lossless local pairs required')
        for key,h in p['bindings'].items():
            if key in sources and sources[key]!=h:raise ValueError('Conflicting original own pair source')
            sources[key]=h
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        rows.extend(dict(name=r['name'],paired_files=r['paired_files'],origin=job,native_full_probe_present_locally=False,
            known_projection_scope='Separate original remote full-probe exporter plus local complete canonical map/disjointness and original native qualifying bytes') for r in p['rows'])
    if len(rows)!=15 or len({r['name'] for r in rows})!=15:raise ValueError('Every unique declared own column required')
    for p in [base_path,Path(__file__),u.ROOT/'experiments/upstream-all-pipeline-coverage/registry.json',u.ROOT/'experiments/upstream-remote15h/plan.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    identities={};cache={}
    for key,h in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=h:raise ValueError('Actual original source/prediction bytes changed')
        identities[key]=identity
    seen=set()
    for row in rows:
        hashes=[]
        for query,count in [('probex',1128164),('qual',2817131)]:
            a=np.load(u.ROOT/row['paired_files'][query],mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Every complete finite own pair required')
            hashes.append(hashlib.sha256(memoryview(a).cast('B')).hexdigest())
        if tuple(hashes) in seen:raise ValueError('Exact duplicate full own pair')
        seen.add(tuple(hashes));row['full_pair_payload_sha256']=dict(zip(['probex','qual'],hashes))
    for key,expected in identities.items():
        s=(u.ROOT/key).stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=expected:raise ValueError('Original own lineage changed during pair admission')
    u.write(REPORT/'verification.json',dict(state='complete_fifteen_fresh_own_pairs_and_original_producer_lineage',rows=rows,
        source_sha256=sources,qualifying_labels_read=False,public_prediction_values_read=False,models_fitted=0,
        scope='All fifteen newly complete fixed own pairs. Exact original recipes, complete phases, transfer/source lineage and full finite pair payloads. No full remote Probe3 vectors locally, model fitting, outcome selection or new ensemble score.'))
    print('Fifteen complete own pairs and full payload hashes verified; five new extension pairs ready separately from frozen own176 study',flush=True)

if __name__=='__main__':main()
