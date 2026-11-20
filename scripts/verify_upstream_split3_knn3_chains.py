#!/usr/bin/env python3
"""Separate full native-chain lineage, phase-build and every known projection checks."""
import argparse
from pathlib import Path
import re
import time
import numpy as np
import upstream_split3_knn3_chains as producer
u=producer.u

def verify(name):
    if name not in producer.JOBS:raise ValueError('Unknown fixed original chain')
    plan=producer.prepare();path=producer.REPORT/(name+'-receipt.json');r=u.read(path)
    expected=['./target/release/fit','chain',name,'-p','pipeline-3.toml','--final','--preds','preds_3']
    if (r['state']!='complete_original_chain_pending_independent_verification' or r['job']!=name or r['parent']!=producer.JOBS[name] or
        r['source_pin']!=u.PIN or r['command']!=expected or r['plan_sha256']!=u.sha(producer.REPORT/'plan.json') or
        r['qualifying_labels_read'] is not False or r['public_prediction_values_read'] is not False):raise ValueError('Original chain completion lineage differs')
    log=producer.REPORT/(name+'.log');text=log.read_text()
    phases=[line.strip() for line in text.splitlines() if line.strip() in {'train_3 => probe_3','fulltrain => qual'}]
    builds=re.findall(r'KNN3: (\d+) items with neighbors, (\d+) total weights',text)
    if phases!=['train_3 => probe_3','fulltrain => qual'] or len(builds)!=2 or '(no_fulltrain)' in text or 'skipped, files supplied by hand' in text:
        raise ValueError('Both complete original native neighbor constructions required')
    if [list(map(int,m)) for m in builds]!=r['full_native_phase_build_counts']:raise ValueError('Original build receipts differ')
    producer.phase_trace(text)  # Independently source-bound original config/default check.
    bindings=dict(plan['bindings'])
    for p in [path,log,producer.REPORT/'plan.json',Path(__file__)]:bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    if not set(plan['original_registry_jobs'][name]['outputs']).issubset(r['outputs']):raise ValueError('Original declared output missing')
    for n,h in r['outputs'].items():
        p=producer.TREE/n
        if u.sha(p)!=h:raise ValueError('Original native output changed')
        bindings[str(p.relative_to(u.ROOT))]=h
    source=producer.pilot.arrays(producer.TREE/'data/probe_3',labels=True)
    known=producer.pilot.arrays(u.FRESH/'data/probex',labels=True)
    mapping=producer.pilot.projection(source,known)
    if not np.array_equal(mapping,np.load(producer.pilot.RUN/'probex-row-map.npy')):raise ValueError('Every original query identity map differs')
    pair=r['projected_pair'];actual=np.load(u.ROOT/pair['probex'],mmap_mode='r');native=np.load(producer.TREE/f'preds_3/{name}.probe_3.npy',mmap_mode='r')
    if actual.shape!=(1128164,) or actual.dtype!=np.float32 or not np.array_equal(actual.view(np.uint32),native[mapping].view(np.uint32)):raise ValueError('Full known projection differs')
    for query,count in [('probe_3',1760494),('qual',2817131)]:
        values=np.load(producer.TREE/f'preds_3/{name}.{query}.npy',mmap_mode='r')
        if values.shape!=(count,) or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Complete finite native queries required')
    for query in ['probex','qual']:
        p=u.ROOT/pair[query]
        if u.sha(p)!=pair[query+'_sha256']:raise ValueError('Actual paired vector changed')
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    if np.any(np.load(producer.TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    for n,h in bindings.items():
        if u.sha(u.ROOT/n)!=h:raise ValueError('Bound parent/data/source/output changed')
    u.write(producer.REPORT/(name+'-verification.json'),dict(state='complete_original_two_phase_knn3_chain',observed=producer.stamp(),job=name,parent=producer.JOBS[name],source_pin=u.PIN,
        both_full_native_neighbor_builds_verified=True,original_default_configuration_verified=True,known_training_disjointness_bound_to_unchanged_original_data=True,
        complete_parent_terminal_inputs_bound=True,full_known_projection_recomputed=True,qualifying_targets_zero=True,projected_pairs=[pair],bindings=bindings,
        qualifying_labels_read=False,public_prediction_values_read=False,accuracy_claimed=False,
        scope='Independent source/binary/base/data/output and full two-phase original KNN3 build checks. All native query values and every known row verified. Raw source-model completion, not a global ensemble or qualifying score.'))
    print(name,'independently verified',flush=True)

def watch():
    remaining=set(producer.JOBS)
    while remaining:
        for name in producer.JOBS:
            if name in remaining and (producer.REPORT/(name+'-receipt.json')).exists():verify(name);remaining.remove(name)
        if remaining:
            u.write(producer.REPORT/'verification-status.json',dict(state='waiting_for_actual_full_native_knn3_completion',remaining=sorted(remaining),observed=producer.stamp()));time.sleep(30)
    u.write(producer.REPORT/'verification-status.json',dict(state='both_original_knn3_chains_independently_verified',observed=producer.stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--job',choices=list(producer.JOBS));a=p.parse_args()
    if a.watch:watch()
    elif a.job:verify(a.job)
    else:p.error('Choose --watch or --job')
