#!/usr/bin/env python3
"""Separately verify complete original split3 jobs and every projected query."""
import argparse
from pathlib import Path
import re
import time
import numpy as np
import upstream_split3_next3 as batch

u=batch.u

def verify(name):
    if name not in batch.JOBS:raise ValueError('Unknown original model')
    plan=batch.prepare();rp=batch.REPORT/(name+'-receipt.json');r=u.read(rp)
    kind,params=batch.JOBS[name]
    expected=['./target/release/fit',kind,name,'-p','pipeline-3.toml','--final','--preds','preds_3']
    if (r['state']!='complete_pending_independent_verification' or r['job']!=name or
        r['source_pin']!=u.PIN or r['plan_sha256']!=u.sha(batch.REPORT/'plan.json') or
        r['command']!=expected or r['model_environment']!=params or
        r['qualifying_labels_read'] is not False or r['public_prediction_values_read'] is not False):
        raise ValueError('Original completion provenance differs')
    epochs=batch.EPOCHS[name];log=batch.REPORT/(name+'.log')
    trace=[int(v) for v in re.findall(r'^Epoch\s+(\d+)\s*\|',log.read_text(),re.M)]
    if trace!=list(range(1,epochs+1))*2:raise ValueError('Full two-phase epoch trace differs')
    bindings=dict(plan['bindings'])
    for path in [rp,log,batch.REPORT/'plan.json',Path(__file__)]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    if not set(plan['original_registry_jobs'][name]['outputs']).issubset(r['outputs']):raise ValueError('Declared output missing')
    for rel,digest in r['outputs'].items():
        path=batch.TREE/rel
        if u.sha(path)!=digest:raise ValueError('Native output changed: '+rel)
        bindings[str(path.relative_to(u.ROOT))]=digest
    source=batch.pilot.arrays(batch.TREE/'data/probe_3',labels=True)
    known=batch.pilot.arrays(u.FRESH/'data/probex',labels=True)
    mapping=batch.pilot.projection(source,known);saved=batch.pilot.RUN/'probex-row-map.npy'
    if not np.array_equal(mapping,np.load(saved)):raise ValueError('Exact known-row map differs')
    for path in (u.FRESH/'data/probex').glob('*.npy'):bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    queries=0
    for split,count in [('probe_3',1760494),('qual',2817131)]:
        for path in (batch.TREE/'preds_3').glob(name+'*.'+split+'.npy'):
            if str(path.relative_to(batch.TREE)) not in r['outputs']:raise ValueError('Unbound query output')
            a=np.load(path,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Malformed native query')
            queries+=1
    columns=[n for n in plan['fixed_published_outputs'] if n==name or n.startswith(name+'_ep')]
    if [row['name'] for row in r['projected_pairs']]!=columns:raise ValueError('Published column set differs')
    for row in r['projected_pairs']:
        actual=np.load(u.ROOT/row['probex'],mmap_mode='r')
        native=np.load(batch.TREE/f"preds_3/{row['name']}.probe_3.npy",mmap_mode='r')
        if actual.shape!=(1128164,) or actual.dtype!=np.float32 or not np.array_equal(actual.view(np.uint32),native[mapping].view(np.uint32)):
            raise ValueError('Actual known projection differs')
        for split in ['probex','qual']:
            path=u.ROOT/row[split]
            if u.sha(path)!=row[split+'_sha256']:raise ValueError('Paired query changed')
            bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    if np.any(np.load(batch.TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    proof=dict(state='complete_original_two_phase_split3_model',observed=batch.stamp(),job=name,
        source_pin=u.PIN,epochs_per_phase=epochs,both_full_epoch_traces=True,native_outputs_verified=len(r['outputs']),
        finite_query_vectors_verified=queries,full_known_projection_recomputed=True,
        known_training_disjointness_bound_to_unchanged_original_data=True,qualifying_targets_zero=True,
        qualifying_labels_read=False,public_prediction_values_read=False,bindings=bindings,
        projected_pairs=r['projected_pairs'],
        scope='Separate unchanged source/binary/data/output hashes, both original full epoch traces, finite native queries and every projected known row checked. Training is not rerun; no blend fitting or qualifying score claimed.')
    u.write(batch.REPORT/(name+'-verification.json'),proof);print(name,proof['state'],flush=True)

def watch():
    remaining=set(batch.JOBS)
    while remaining:
        for name in batch.JOBS:
            if name not in remaining:continue
            path=batch.REPORT/(name+'-receipt.json')
            if path.exists():verify(name);remaining.remove(name)
        if remaining:
            u.write(batch.REPORT/'verification-status.json',dict(state='waiting_for_original_native_completion',observed=batch.stamp(),remaining=sorted(remaining)))
            time.sleep(30)
    u.write(batch.REPORT/'verification-status.json',dict(state='all_three_original_models_independently_verified',observed=batch.stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--job',choices=list(batch.JOBS));a=p.parse_args()
    if a.watch:watch()
    elif a.job:verify(a.job)
    else:p.error('Choose --watch or --job')
