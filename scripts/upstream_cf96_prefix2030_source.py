#!/usr/bin/env python3
"""Freeze two own original-source prefix pairs from the verified own transfer."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import upstream_cf96_prefix10_screen as old
u=old.u
RUN=u.ROOT/'artifacts/upstream-cf96-prefix2030-source'
REPORT=u.ROOT/'experiments/upstream-cf96-prefix2030-source'
NEW=['cfnade-96_ep20','cfnade-96_ep30']

def verified_inputs():
    path=REPORT/'paired-receipt.json';r=u.read(path)
    if (r['state']!='complete_verified_own_published_prefix20_30_pairs' or r['source_pin']!=u.PIN or
        r['prediction_epochs']!=[20,30] or r['original_capacity']!=dict(hidden=320,rank=96,configured_epochs_per_phase=95) or
        r['full95_model_complete'] is not False or r['qualifying_labels_read'] is not False or
        r['public_prediction_values_read'] is not False):raise ValueError('Exact complete own prefix evidence required')
    sources=dict(r['source_sha256']);sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Own prefix lineage changed')
    if [row['name'] for row in r['rows']]!=NEW:raise ValueError('Exactly both fixed own prefixes required')
    return r['rows'],sources

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing prefix evidence preserved')
    manifest=u.read(old.BUNDLE/'manifest.json');origin=u.read(old.native.REPORT/'plan.json')
    preflight=u.ROOT/'experiments/upstream-remote9h-cf96/local-preflight-verification.json';checked=u.read(preflight)
    if (manifest['source_pin']!=u.PIN or manifest['source_id']!=origin['source_id'] or
        manifest['data_id']!=origin['data_id'] or origin['hidden']!=320 or origin['rank']!=96 or
        origin['epochs_per_phase']!=95 or u.sha(old.BUNDLE/'origin-plan.json')!=u.sha(old.native.REPORT/'plan.json') or
        checked['manifest_sha256']!=u.sha(old.BUNDLE/'manifest.json') or
        not checked['all84_tensors_per_phase_verified'] or not checked['native_original_source_parity_passed']):
        raise ValueError('Previously independently verified own original320/96/95 package required')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(fixed_additions=NEW,epochs=[20,30],configured_epochs_each=95,
        rule='Both fixed decade prefixes after existingep10, from our own previously verified33/31 transfer. Reverify every member, original data,84-array states and native arithmetic parity. These are original source side outputs beyond the released declared-column schema. Full95/95 source trajectory continues unchanged; no early stopping, qualifying selection or whole-model completion claim.',
        qualifying_labels_read=False,public_prediction_values_read=False))
    with (REPORT/'package-reverification.log').open('x') as log:
        subprocess.run([sys.executable,str(u.ROOT/'scripts/upstream_remote_cf96.py'),'verify-bundle',
            '--bundle',str(old.BUNDLE),'--data-root',str(u.FRESH/'data')],cwd=u.ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    paths=[Path(__file__),old.BUNDLE/'manifest.json',old.BUNDLE/'origin-plan.json',old.BUNDLE/'source-verification.json',
        old.native.REPORT/'plan.json',preflight,u.ROOT/'scripts/upstream_remote_cf96.py',REPORT/'plan.json',REPORT/'package-reverification.log']
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths};rows=[]
    for epoch,name in zip([20,30],NEW):
        pairs={}
        for query,phase,count in [('probex','trainx',1128164),('qual','fulltrain',2817131)]:
            filename=name+'.'+query+'.npy';path=old.native.TREE/'preds_new'/filename
            digest=manifest['phases'][phase]['prediction_sha256'][filename]
            if manifest['phases'][phase]['epoch']<epoch or manifest['files']['native-source/preds_new/'+filename]!=digest or u.sha(path)!=digest:
                raise ValueError('Original source-published prefix binding differs')
            a=np.load(path,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Full native prefix vector differs')
            target=RUN/filename;u.link_or_copy(path,target)
            if u.sha(target)!=digest:raise ValueError('Frozen own prefix differs')
            pairs[query]=str(target.relative_to(u.ROOT))
            for p in [path,target]:sources[str(p.relative_to(u.ROOT))]=digest
        rows.append(dict(name=name,paired_files=pairs,duplicate_of=None))
    u.write(REPORT/'paired-receipt.json',dict(state='complete_verified_own_published_prefix20_30_pairs',rows=rows,
        source_pin=u.PIN,source_id=manifest['source_id'],data_id=manifest['data_id'],source_sha256=sources,
        captured_trajectory_epochs={k:v['epoch'] for k,v in manifest['phases'].items()},prediction_epochs=[20,30],
        original_capacity=dict(hidden=320,rank=96,configured_epochs_per_phase=95),
        full95_model_complete=False,epoch20_30_inference_rerun=False,qualifying_labels_read=False,
        public_prediction_values_read=False,
        scope='Both own complete source-published prefix pairs. Actual own33/31 state/data/arithmetic package reverified. Prefix model tensors were pruned; no new prefix inference replay is claimed. Full original95/95 training continues. No qualifying score, public-column parity or whole95 model completion claim.'))
    verified_inputs()

if __name__=='__main__':main()
