#!/usr/bin/env python3
"""Fixed own original CF96 epoch20/30 prefix addition to own107/item154."""
from pathlib import Path
import math
import numpy as np
import upstream_item154_features as temporal
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-cf96-prefix2030-screen'
REPORT=u.ROOT/'experiments/upstream-cf96-prefix2030-screen'
import upstream_cf96_prefix2030_source as prefix
NEW=prefix.NEW

def verified_inputs():return prefix.verified_inputs()

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing CF96 prefix screen preserved')
    temporal.configure();parent=temporal.n.verify_prepared();rows,sources=verified_inputs()
    if parent['support_columns']!=154 or parent['prediction_columns']!=107:raise ValueError('Complete own107/item154 parent required')
    proofpath=temporal.REPORT/'cache-column-verification.json'
    if u.read(proofpath)['manifest_sha256']!=u.sha(temporal.REPORT/'manifest.json'):raise ValueError('Parent proof differs')
    for p in [proofpath,temporal.REPORT/'manifest.json',Path(__file__)]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in parent['prepared_sha256'].items():sources[str((temporal.RUN/name).relative_to(u.ROOT))]=digest
    for row in rows:
        for split,count in [('probex',1128164),('qual',2817131)]:
            p=u.ROOT/row['paired_files'][split];a=np.load(p,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Native pair shape/precision/finiteness differs')
            if u.sha(p)!=sources[str(p.relative_to(u.ROOT))]:raise ValueError('Paired source changed')
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex']),0,6) for r in rows])
    if old.shape!=(len(y),183):raise ValueError('Own107 ridge schema differs')
    RUN.mkdir();REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(ready_unused=rows,fixed_additions=NEW,
        rule='Fixed control and BOTH own originalCF96 epoch20/30 side outputs from the reverified own33/31 package. Original320/96/95 trajectory continues unchanged, incomplete95/95 explicitly disclosed. Retain existingep10. Full original user folds and ridge penalty; no subset search or qualifying outcomes.',qualifying_labels_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1]));records=[]
    for arm,result in scored.items():
        p=RUN/(arm+'-ridge.npy');np.save(p,result['prediction'])
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-12:raise ValueError('Parent control differs')
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(p),fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=NEW,source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Known-ProbeX diagnostic on fixed ownCF96 epoch20/30 side outputs; original95/95 training incomplete and continuing. No nonlinear or qualifying promotion.'))
    print(records,flush=True)

if __name__=='__main__':main()
