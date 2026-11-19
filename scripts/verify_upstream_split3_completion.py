#!/usr/bin/env python3
"""Separate source/output/projection completion check for the original split3 pilot."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_split3_pilot as pilot

u=pilot.u


def verify():
    receipt_path=pilot.REPORT/'paired-receipt.json';receipt=u.read(receipt_path)
    plan_path=pilot.REPORT/'plan.json';plan=u.read(plan_path);data_path=pilot.REPORT/'data-verification.json';data=u.read(data_path)
    if (receipt['state']!='complete' or receipt['job']!=pilot.JOB or receipt['source_pin']!=u.PIN or
            receipt['plan_sha256']!=u.sha(plan_path) or receipt['data_verification_sha256']!=u.sha(data_path) or
            plan['implementation_sha256']!=pilot.implementations() or receipt['command']!=pilot.COMMAND or
            receipt['model_environment']!=pilot.MODEL_ENV or receipt['qualifying_labels_read'] is not False or
            receipt['public_prediction_values_read'] is not False or not data['known_rows_absent_from_training']):
        raise ValueError('Complete original pilot provenance required')
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [receipt_path,plan_path,data_path,Path(__file__)]}
    for name,digest in plan['source_files_sha256'].items():
        path=pilot.TREE/name
        if u.sha(path)!=digest:raise ValueError('Source code/configuration changed')
        bindings[str(path.relative_to(u.ROOT))]=digest
    for name,digest in plan['input_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original data/build binding changed')
        bindings[name]=digest
    for name,digest in plan['binary_sha256'].items():
        path=pilot.TREE/'target/release'/name
        if u.sha(path)!=digest:raise ValueError('Native executable changed')
        bindings[str(path.relative_to(u.ROOT))]=digest
    for name,digest in data['data_sha256'].items():
        path=pilot.RUN/name
        if u.sha(path)!=digest:raise ValueError('Generated split3 data changed')
        bindings[str(path.relative_to(u.ROOT))]=digest
    if not set(plan['original_registry_job']['outputs']).issubset(receipt['outputs']):raise ValueError('Original outputs incomplete')
    for name,digest in receipt['outputs'].items():
        path=pilot.TREE/name
        if u.sha(path)!=digest:raise ValueError('Completed native output changed: '+name)
        bindings[str(path.relative_to(u.ROOT))]=digest
    source=pilot.arrays(pilot.TREE/'data/probe_3',labels=True);known=pilot.arrays(u.FRESH/'data/probex',labels=True)
    mapping=pilot.projection(source,known);saved=pilot.RUN/'probex-row-map.npy'
    if u.sha(saved)!=data['row_map_sha256'] or not np.array_equal(np.load(saved),mapping):raise ValueError('Actual query projection differs')
    if np.any(np.load(pilot.TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets not zero')
    for split,rows in [('probe_3',len(source[0])),('qual',2817131)]:
        for epoch in range(1,13):
            path=pilot.TREE/f'preds_3/{pilot.JOB}_ep{epoch:02}.{split}.npy';a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.shape!=(rows,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Original snapshot invalid')
        a=np.load(pilot.TREE/f'preds_3/{pilot.JOB}.{split}.npy');b=np.load(pilot.TREE/f'preds_3/{pilot.JOB}_ep12.{split}.npy')
        if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):raise ValueError('Terminal/final epoch differ')
    records=[]
    if [r['name'] for r in receipt['projected_pairs']]!=[pilot.JOB,pilot.JOB+'_ep08']:raise ValueError('Fixed published pair inventory differs')
    for row in receipt['projected_pairs']:
        actual=np.load(u.ROOT/row['probex']);native=np.load(pilot.TREE/f"preds_3/{row['name']}.probe_3.npy",mmap_mode='r')
        if not np.array_equal(actual.view(np.uint32),native[mapping].view(np.uint32)):
            raise ValueError('Projected own prediction differs')
        for split in ['probex','qual']:
            if u.sha(u.ROOT/row[split])!=row[split+'_sha256']:raise ValueError('Paired vector hash differs')
            bindings[row[split]]=row[split+'_sha256']
        score=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(actual,known[3]))/len(actual))
        records.append(dict(name=row['name'],known_probex_rows=len(actual),standalone_known_probex_rmse=score))
    result=dict(state='complete_original_two_phase_split3_pilot',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        verifier_sha256=u.sha(Path(__file__)),paired_receipt_sha256=u.sha(receipt_path),bindings=bindings,
        native_outputs_verified=len(receipt['outputs']),native_epochs_per_phase=12,
        all24_query_snapshots_verified=True,terminal_equals_epoch12_both_phases=True,
        full_known_projection_recomputed=True,records=records,qualifying_targets_zero=True,qualifying_labels_read=False,
        scope='Separate unchanged source/binary/data/output binding,24 snapshot shape/finiteness, terminal/final-epoch and exact known-row projection checks. Training is not rerun. The earlier full training-disjointness check is bound through unchanged generated data hashes. Standalone known-ProbeX diagnostics are not qualifying scores or blend promotions.')
    u.write(pilot.REPORT/'completion-verification.json',result);print({k:v for k,v in result.items() if k!='bindings'})


if __name__=='__main__':verify()
