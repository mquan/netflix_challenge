#!/usr/bin/env python3
"""Fixed all-two addition: published own CF96 ep10 and completed temporal60."""
from pathlib import Path
import math
import numpy as np
import upstream_split3_105 as temporal
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-ready107-screen'
REPORT=u.ROOT/'experiments/upstream-ready107-screen'
NEW=['cfnade-96_ep10','tsvdx5-60']

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing all-two screen preserved')
    temporal.configure();parent=temporal.n.verify_prepared()
    proofpath=temporal.REPORT/'cache-column-verification.json'
    if u.read(proofpath)['manifest_sha256']!=u.sha(temporal.REPORT/'manifest.json'):raise ValueError('Own105 proof mismatch')
    cfpath=u.ROOT/'experiments/upstream-cf96-prefix10-screen/paired-receipt.json';cf=u.read(cfpath)
    tpath=u.ROOT/'experiments/upstream-tsvdx5-60-completion/verification.json';tp=u.read(tpath)
    if cf['state']!='verified_complete_published_epoch10_pair' or cf['prediction_epoch']!=10 or cf['original_capacity']!={'hidden':320,'rank':96,'configured_epochs_per_phase':95}:
        raise ValueError('Verified original-capacity CF96 snapshot required')
    if tp['state']!='complete_original_temporal_native_outputs' or tp['epochs_per_phase']!=23:
        raise ValueError('Complete original temporal23/23 verification required')
    sources={**cf['source_sha256'],**tp['bindings']}
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Verified native input changed: '+name)
    for p in [cfpath,tpath,proofpath,temporal.REPORT/'manifest.json',Path(__file__)]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in parent['prepared_sha256'].items():sources[str((temporal.RUN/name).relative_to(u.ROOT))]=digest
    rows=[dict(name=NEW[0],paired_files=cf['pairs'],duplicate_of=None),
        dict(name=NEW[1],paired_files={s:str((u.FRESH/'preds_new'/f'{NEW[1]}.{s}.npy').relative_to(u.ROOT)) for s in ['probex','qual']},duplicate_of=None)]
    for row in rows:
        for split,count in [('probex',1128164),('qual',2817131)]:
            path=u.ROOT/row['paired_files'][split];a=np.load(path,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid paired source vector')
            sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex']),0,6) for r in rows])
    if old.shape!=(len(y),181):raise ValueError('Own105 ridge schema mismatch')
    RUN.mkdir();REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(ready_unused=rows,fixed_additions=NEW,
        rule='Append both independently verified public-schema columns together. Fixed control/all-two arms, original user folds and ridge penalty; no snapshot/subset search or qualifying outcomes.',
        cf96_full95_model_complete=False,tsvdx5_60_full23_23_model_complete=True,qualifying_labels_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1]));records=[]
    for arm,result in scored.items():
        path=RUN/(arm+'-ridge.npy');np.save(path,result['prediction'])
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-12:raise ValueError('Own105 control mismatch')
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(path),
            fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=NEW,source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Own107 known-ProbeX ridge diagnostic only: complete source-published CF96 epoch10 pair plus complete original temporal23/23 pair. No nonlinear or qualifying promotion.'))
    print(records,flush=True)

if __name__=='__main__':main()
