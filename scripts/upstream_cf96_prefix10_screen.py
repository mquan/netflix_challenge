#!/usr/bin/env python3
"""Verify our published CF96 epoch10 pair and screen its fixed own105 addition."""
from pathlib import Path
import math
import subprocess
import sys
import numpy as np
import upstream_split3_105 as temporal
import upstream_cfnade_parallel as native
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-cf96-prefix10-screen'
REPORT=u.ROOT/'experiments/upstream-cf96-prefix10-screen'
NAME='cfnade-96_ep10'
BUNDLE=u.ROOT/'artifacts/remote9h-cf96-transfer/cfnade96-package'

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing CF96 prefix screen preserved')
    manifest=u.read(BUNDLE/'manifest.json');origin=u.read(native.REPORT/'plan.json')
    if (manifest['source_pin']!=u.PIN or manifest['source_id']!=origin['source_id'] or
            manifest['data_id']!=origin['data_id'] or origin['hidden']!=320 or origin['rank']!=96 or
            origin['epochs_per_phase']!=95 or u.sha(BUNDLE/'origin-plan.json')!=u.sha(native.REPORT/'plan.json')):
        raise ValueError('Exact original320/96/95 trajectory required')
    preflight=u.ROOT/'experiments/upstream-remote9h-cf96/local-preflight-verification.json';checked=u.read(preflight)
    if checked['manifest_sha256']!=u.sha(BUNDLE/'manifest.json') or not checked['all84_tensors_per_phase_verified'] or not checked['native_original_source_parity_passed']:
        raise ValueError('Previously verified own package required')
    REPORT.mkdir();RUN.mkdir()
    # Rehash every package member, both84-array states and actual input datasets;
    # rerun the original/checkpointed synthetic native arithmetic parity proof.
    with (REPORT/'package-reverification.log').open('w') as log:
        subprocess.run([sys.executable,str(u.ROOT/'scripts/upstream_remote_cf96.py'),'verify-bundle',
            '--bundle',str(BUNDLE),'--data-root',str(u.FRESH/'data')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [BUNDLE/'manifest.json',BUNDLE/'origin-plan.json',
        BUNDLE/'source-verification.json',native.REPORT/'plan.json',preflight,Path(__file__),
        u.ROOT/'scripts/upstream_remote_cf96.py',REPORT/'package-reverification.log']}
    pairs={}
    for split,train,rows in [('probex','trainx',1128164),('qual','fulltrain',2817131)]:
        name=f'{NAME}.{split}.npy';path=native.TREE/'preds_new'/name
        digest=manifest['phases'][train]['prediction_sha256'][name]
        if manifest['phases'][train]['epoch']<10 or u.sha(path)!=digest or manifest['files']['native-source/preds_new/'+name]!=digest:
            raise ValueError('Current snapshot differs from earlier own-package hash')
        a=np.load(path,mmap_mode='r')
        if a.shape!=(rows,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Malformed own CF96 snapshot')
        sources[str(path.relative_to(u.ROOT))]=digest
        target=RUN/name;u.link_or_copy(path,target)
        if u.sha(target)!=digest:raise ValueError('Frozen prefix copy differs')
        pairs[split]=str(target.relative_to(u.ROOT));sources[pairs[split]]=digest
    u.write(REPORT/'paired-receipt.json',dict(state='verified_complete_published_epoch10_pair',name=NAME,pairs=pairs,
        source_pin=u.PIN,source_id=manifest['source_id'],data_id=manifest['data_id'],source_sha256=sources,
        captured_trajectory_epochs={k:v['epoch'] for k,v in manifest['phases'].items()},
        original_capacity=dict(hidden=320,rank=96,configured_epochs_per_phase=95),prediction_epoch=10,
        full95_model_complete=False,epoch10_inference_rerun=False,qualifying_labels_read=False,
        scope='Both original source-published epoch10 query vectors complete. Match the independently verified own33/31 package, whose states, actual data and arithmetic parity were rechecked. Epoch10 tensors were pruned; no new epoch10 inference replay claimed. Full95-epoch native training remains pending.'))
    temporal.configure();parent=temporal.n.verify_prepared();cacheproof=temporal.REPORT/'cache-column-verification.json'
    if u.read(cacheproof)['manifest_sha256']!=u.sha(temporal.REPORT/'manifest.json'):raise ValueError('Own105 cache proof mismatch')
    for p in [REPORT/'paired-receipt.json',temporal.REPORT/'manifest.json',cacheproof]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in parent['prepared_sha256'].items():sources[str((temporal.RUN/name).relative_to(u.ROOT))]=digest
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.clip(np.load(u.ROOT/pairs['probex']),0,6)[:,None]
    u.write(REPORT/'plan.json',dict(ready_unused=[dict(name=NAME,paired_files=pairs,duplicate_of=None)],fixed_additions=[NAME],
        rule='One fixed public-schema epoch10 addition to own105. Same user folds and ridge penalty; no qualifying outcomes or snapshot search. Whole95-epoch model remains incomplete.',
        qualifying_labels_read=False,public_prediction_values_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0]));records=[]
    for arm,result in scored.items():
        path=RUN/(arm+'-ridge.npy');np.save(path,result['prediction'])
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-12:raise ValueError('Own105 ridge control differs')
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(path),
            fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==fold],y[groups//2==fold]) for fold in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=[NAME],source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),full95_model_complete=False,qualifying_labels_read=False,
        scope='Known-ProbeX ridge diagnostic for our complete source-published CF96 epoch10 pair. No nonlinear/qualifying promotion or completed95-epoch-model claim.'))
    print(records,flush=True)

if __name__=='__main__':main()
